# MCP OAuth 2.1 Authentication Support — PKCE (Proof Key for Code Exchange) Based
#
# Implements RFC 7636 PKCE flow explicitly:
#   1. Generate code_verifier + code_challenge (S256)
#   2. Send code_challenge in authorization request
#   3. Send code_verifier in token exchange
#   4. Support both public clients (PKCE only) and confidential clients (PKCE + secret)

import asyncio
import base64
import hashlib
import json
import os
import secrets
import string
import time
import webbrowser
from pathlib import Path
from typing import Optional

from authlib.integrations.requests_client import OAuth2Session
from authlib.oauth2 import OAuth2Client
import secrets

import httpx

from src import logger

# MCP SDK OAuth imports
try:
    from mcp.client.auth import OAuthClientProvider, TokenStorage
    from mcp.client.auth.oauth2 import (
        OAuthClientMetadata,
        OAuthToken,
        OAuthClientInformationFull,
    )
    from mcp.shared.auth import OAuthMetadata

    OAUTH_AVAILABLE = True
except ImportError:
    OAUTH_AVAILABLE = False
    logger.warning("MCP OAuth modules not available. Update mcp SDK.")


# Well-known OAuth provider presets
OAUTH_PRESETS = {
    "github": {
        "authorization_endpoint": "https://github.com/login/oauth/authorize",
        "token_endpoint": "https://github.com/login/oauth/access_token",
        "default_scopes": "read:user repo read:project read:packages",
        "issuer": "https://github.com",
    },
    "google-drive": {
        "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_endpoint": "https://oauth2.googleapis.com/token",
        "default_scopes": "https://www.googleapis.com/auth/drive.readonly",
        "issuer": "https://accounts.google.com",
    },
    "gmail": {
        "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_endpoint": "https://oauth2.googleapis.com/token",
        "default_scopes": "https://www.googleapis.com/auth/gmail.readonly",
        "issuer": "https://accounts.google.com",
    },
    "google-calendar": {
        "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_endpoint": "https://oauth2.googleapis.com/token",
        "default_scopes": "https://www.googleapis.com/auth/calendar.readonly",
        "issuer": "https://accounts.google.com",
    },
    "notion": {
        "authorization_endpoint": "https://api.notion.com/v1/oauth/authorize",
        "token_endpoint": "https://api.notion.com/v1/oauth/token",
        "default_scopes": "read_content read_user",
        "issuer": "https://api.notion.com",
    },
}


# ---------------------------------------------------------------------------
# PKCE Parameter Generation (RFC 7636)
# ---------------------------------------------------------------------------


class PKCEParams:
    """
    Generate and hold PKCE parameters.

    - code_verifier: 128-char random string from unreserved characters
    - code_challenge: Base64-URL-encoded SHA256 of code_verifier (S256)
    """

    def __init__(self, code_verifier: Optional[str] = None):
        if code_verifier:
            self.code_verifier = code_verifier
        else:
            # RFC 7636 §4.1 — 43–128 unreserved characters
            unreserved = string.ascii_letters + string.digits + "-._~"
            self.code_verifier = "".join(secrets.choice(unreserved) for _ in range(128))

        digest = hashlib.sha256(self.code_verifier.encode("ascii")).digest()
        self.code_challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
        self.code_challenge_method = "S256"

    def to_dict(self) -> dict:
        return {
            "code_verifier": self.code_verifier,
            "code_challenge": self.code_challenge,
            "code_challenge_method": self.code_challenge_method,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PKCEParams":
        return cls(code_verifier=data["code_verifier"])


# ---------------------------------------------------------------------------
# File-based Token & PKCE State Storage
# ---------------------------------------------------------------------------


class FileTokenStorage:
    """
    File-based token storage implementing MCP SDK's TokenStorage protocol.

    Stores OAuth tokens, client registration info, and PKCE state as JSON
    files in ~/.mcp/tokens/<server_name>/.

    Token validity is tracked via a metadata file that records
    the save timestamp and expires_in value from the token response.
    """

    EXPIRY_GRACE_SECONDS = 60

    def __init__(self, server_name: str, base_dir: Optional[str] = None):
        self.server_name = server_name
        self.storage_dir = Path(base_dir or os.path.expanduser("~/.mcp/tokens")) / server_name
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._tokens_path = self.storage_dir / "tokens.json"
        self._client_info_path = self.storage_dir / "client_info.json"
        self._metadata_path = self.storage_dir / "token_metadata.json"
        self._pkce_state_path = self.storage_dir / "pkce_state.json"

    # -- PKCE state persistence --

    def save_pkce_state(self, pkce: PKCEParams, state: str) -> None:
        """Persist PKCE code_verifier + OAuth state between auth request and callback."""
        data = {**pkce.to_dict(), "state": state, "created_at": time.time()}
        try:
            self._pkce_state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.debug(f"Saved PKCE state for {self.server_name}")
        except Exception as e:
            logger.error(f"Failed to save PKCE state for {self.server_name}: {e}")

    def load_pkce_state(self) -> Optional[dict]:
        """Load persisted PKCE state (code_verifier + state). Returns None if missing/stale."""
        if not self._pkce_state_path.exists():
            return None
        try:
            data = json.loads(self._pkce_state_path.read_text(encoding="utf-8"))
            # PKCE state older than 10 minutes is stale
            if time.time() - data.get("created_at", 0) > 600:
                self.clear_pkce_state()
                return None
            return data
        except Exception:
            return None

    def clear_pkce_state(self) -> None:
        """Delete persisted PKCE state after token exchange."""
        if self._pkce_state_path.exists():
            try:
                self._pkce_state_path.unlink()
            except Exception:
                pass

    # -- Token metadata --

    def _load_metadata(self) -> Optional[dict]:
        if not self._metadata_path.exists():
            return None
        try:
            return json.loads(self._metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_metadata(self, expires_in: Optional[int]) -> None:
        metadata = {"saved_at": time.time(), "expires_in": expires_in}
        try:
            self._metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to save token metadata for {self.server_name}: {e}")

    def is_token_expired(self) -> bool:
        """Check whether the stored token has expired."""
        if not self._tokens_path.exists():
            return True

        metadata = self._load_metadata()
        if metadata and metadata.get("saved_at") and metadata.get("expires_in"):
            elapsed = time.time() - metadata["saved_at"]
            if elapsed >= (metadata["expires_in"] - self.EXPIRY_GRACE_SECONDS):
                logger.info(f"Token for {self.server_name} has expired (elapsed={elapsed:.0f}s, expires_in={metadata['expires_in']}s)")
                return True
            return False

        # Fallback: read expires_in from the token file + file mtime
        try:
            data = json.loads(self._tokens_path.read_text(encoding="utf-8"))
            expires_in = data.get("expires_in")
            if expires_in is not None:
                elapsed = time.time() - self._tokens_path.stat().st_mtime
                if elapsed >= (expires_in - self.EXPIRY_GRACE_SECONDS):
                    logger.info(f"Token for {self.server_name} has expired (fallback, elapsed={elapsed:.0f}s)")
                    return True
                return False
        except Exception:
            pass

        return False

    def clear_tokens(self) -> None:
        """Delete stored tokens and metadata to force re-authentication."""
        for path in (self._tokens_path, self._metadata_path):
            if path.exists():
                try:
                    path.unlink()
                    logger.info(f"Deleted {path.name} for {self.server_name}")
                except Exception as e:
                    logger.error(f"Failed to delete {path.name} for {self.server_name}: {e}")

    # -- TokenStorage protocol (MCP SDK) --

    async def get_tokens(self) -> "OAuthToken | None":
        if not self._tokens_path.exists():
            return None
        if self.is_token_expired():
            logger.warning(f"Stored token for {self.server_name} has expired. Clearing.")
            self.clear_tokens()
            return None
        try:
            data = json.loads(self._tokens_path.read_text(encoding="utf-8"))
            return OAuthToken.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to load OAuth tokens: {e}")
            return None

    async def set_tokens(self, tokens: "OAuthToken") -> None:
        try:
            self._tokens_path.write_text(tokens.model_dump_json(indent=2), encoding="utf-8")
            expires_in = getattr(tokens, "expires_in", None)
            self._save_metadata(expires_in)
            logger.info(f"Saved OAuth tokens for {self.server_name}" + (f" (expires_in={expires_in}s)" if expires_in else ""))
        except Exception as e:
            logger.error(f"Failed to save OAuth tokens: {e}")

    async def get_client_info(self) -> "OAuthClientInformationFull | None":
        if not self._client_info_path.exists():
            return None
        try:
            data = json.loads(self._client_info_path.read_text(encoding="utf-8"))
            return OAuthClientInformationFull.model_validate(data)
        except Exception:
            return None

    async def set_client_info(self, client_info: "OAuthClientInformationFull") -> None:
        try:
            self._client_info_path.write_text(client_info.model_dump_json(indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save client info: {e}")


# ---------------------------------------------------------------------------
# Local OAuth Callback Server
# ---------------------------------------------------------------------------


def _build_callback_handler(redirect_port: int):
    """Build a local HTTP server that captures the OAuth callback."""

    async def callback_handler() -> tuple[str, str | None]:
        auth_code_future: asyncio.Future[tuple[str, str | None]] = asyncio.get_event_loop().create_future()

        from aiohttp import web

        async def handle_callback(request: web.Request) -> web.Response:
            code = request.query.get("code")
            state = request.query.get("state")
            error = request.query.get("error")

            if error:
                auth_code_future.set_exception(RuntimeError(f"OAuth error: {error} - {request.query.get('error_description', '')}"))
                return web.Response(
                    text=f"<html><body><h1>Authentication Failed</h1><p>Error: {error}</p><p>You can close this window.</p></body></html>",
                    content_type="text/html",
                )

            if not code:
                auth_code_future.set_exception(RuntimeError("No authorization code received"))
                return web.Response(
                    text="<html><body><h1>Error</h1><p>No authorization code received.</p></body></html>",
                    content_type="text/html",
                )

            auth_code_future.set_result((code, state))
            return web.Response(
                text="<html><body><h1>Authentication Successful!</h1><p>You can close this window and return to the application.</p></body></html>",
                content_type="text/html",
            )

        app = web.Application()
        app.router.add_get("/callback", handle_callback)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "localhost", redirect_port)
        await site.start()
        logger.info(f"OAuth callback server listening on http://localhost:{redirect_port}/callback")

        try:
            result = await asyncio.wait_for(auth_code_future, timeout=300.0)
            return result
        finally:
            await runner.cleanup()

    return callback_handler


# ---------------------------------------------------------------------------
# PKCE OAuth Provider — explicit PKCE flow
# ---------------------------------------------------------------------------


class PKCEOAuthProvider:
    """
    OAuth 2.1 provider with explicit PKCE (RFC 7636) support.

    Handles the full Authorization Code + PKCE flow:
      1. Generate code_verifier / code_challenge
      2. Build authorization URL with code_challenge + S256
      3. Open browser, wait for callback
      4. Exchange authorization code + code_verifier for tokens
      5. Persist tokens via FileTokenStorage

    Supports:
      - Public clients (no client_secret, PKCE only)
      - Confidential clients (client_secret + PKCE)
      - Token refresh with PKCE
    """

    def __init__(
        self,
        server_name: str,
        authorization_endpoint: str,
        token_endpoint: str,
        client_id: str,
        redirect_uri: str,
        storage: FileTokenStorage,
        scopes: Optional[str] = None,
        client_secret: Optional[str] = None,
        issuer: Optional[str] = None,
        redirect_port: int = 3000,
        timeout: float = 300.0,
    ):
        self.server_name = server_name
        self.authorization_endpoint = authorization_endpoint
        self.token_endpoint = token_endpoint
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.scopes = scopes
        self.issuer = issuer
        self.storage = storage
        self.redirect_port = redirect_port
        self.timeout = timeout

        self._callback_handler = _build_callback_handler(redirect_port)

    @property
    def is_public_client(self) -> bool:
        """Public client = no client_secret, relies entirely on PKCE."""
        return not self.client_secret

    async def get_valid_token(self) -> Optional[str]:
        """
        Return a valid access token, refreshing or re-authenticating as needed.

        Flow:
          1. Check stored token → return if valid
          2. Try refresh_token if available
          3. Fall back to full PKCE authorization flow
        """
        tokens = await self.storage.get_tokens()

        if tokens and not self.storage.is_token_expired():
            return tokens.access_token

        # Try refresh
        if tokens and getattr(tokens, "refresh_token", None):
            refreshed = await self._refresh_token(tokens.refresh_token)
            if refreshed:
                return refreshed.access_token

        # Full PKCE auth flow
        new_tokens = await self._authorize()
        if new_tokens:
            return new_tokens.access_token

        return None

    async def _authorize(self) -> Optional["OAuthToken"]:
        """
        Run the full PKCE authorization code flow.

        1. Generate PKCE params (code_verifier, code_challenge)
        2. Build authorization URL with code_challenge
        3. Open browser, start local callback server
        4. Exchange auth code + code_verifier for tokens
        """
        pkce = PKCEParams()
        state = secrets.token_urlsafe(32)

        # Persist PKCE state in case the process restarts between auth and callback
        self.storage.save_pkce_state(pkce, state)

        # Build authorization URL
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "code_challenge": pkce.code_challenge,
            "code_challenge_method": pkce.code_challenge_method,
            "state": state,
        }
        if self.scopes:
            params["scope"] = self.scopes

        if self.server_name.lower() == "notion":
            params["owner"] = "user"

        auth_url = f"{self.authorization_endpoint}?{_urlencode(params)}"
        logger.info(f"[PKCE] Opening browser for authorization: {auth_url}")
        webbrowser.open(auth_url)

        # Wait for callback
        auth_code, returned_state = await self._callback_handler()

        # Validate state to prevent CSRF
        if returned_state != state:
            logger.error(f"[PKCE] State mismatch: expected={state}, got={returned_state}")
            self.storage.clear_pkce_state()
            raise RuntimeError("OAuth state mismatch — possible CSRF attack")

        # Exchange code + code_verifier for tokens
        tokens = await self._exchange_code(auth_code, pkce.code_verifier)
        self.storage.clear_pkce_state()
        return tokens

    async def _exchange_code(self, auth_code: str, code_verifier: str) -> Optional["OAuthToken"]:
        """
        Exchange authorization code for tokens, including code_verifier (PKCE §4.5).
        """
        data = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "code_verifier": code_verifier,
        }

        # Confidential clients include client_secret
        if self.client_secret:
            data["client_secret"] = self.client_secret

        headers = {"Accept": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.token_endpoint, data=data, headers=headers)
                resp.raise_for_status()

                token_data = resp.json()

                # GitHub returns token in non-standard format sometimes
                if "access_token" not in token_data:
                    logger.error(f"[PKCE] Token response missing access_token: {token_data}")
                    return None

                token = OAuthToken.model_validate(token_data)
                await self.storage.set_tokens(token)
                logger.info(f"[PKCE] Token exchange successful for {self.server_name}")
                return token

        except httpx.HTTPStatusError as e:
            logger.error(f"[PKCE] Token exchange failed ({e.response.status_code}): {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"[PKCE] Token exchange error: {e}")
            return None

    async def _refresh_token(self, refresh_token: str) -> Optional["OAuthToken"]:
        """
        Refresh an expired access token using the refresh_token grant.

        PKCE is not required for refresh, but client_id is always sent.
        """
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret

        headers = {"Accept": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.token_endpoint, data=data, headers=headers)
                resp.raise_for_status()

                token_data = resp.json()
                if "access_token" not in token_data:
                    logger.warning(f"[PKCE] Refresh response missing access_token")
                    return None

                token = OAuthToken.model_validate(token_data)
                await self.storage.set_tokens(token)
                logger.info(f"[PKCE] Token refresh successful for {self.server_name}")
                return token

        except Exception as e:
            logger.warning(f"[PKCE] Token refresh failed for {self.server_name}: {e}")
            return None


# ---------------------------------------------------------------------------
# httpx Auth adapter — bridges PKCEOAuthProvider into MCP SDK transport
# ---------------------------------------------------------------------------


class PKCEAuth(httpx.Auth):
    """
    httpx.Auth adapter that injects a Bearer token obtained via PKCE flow.

    Used by MCP SDK's SSE/HTTP transports which accept an httpx.Auth instance.
    """

    def __init__(self, provider: PKCEOAuthProvider):
        self._provider = provider
        self._token: Optional[str] = None

    async def _ensure_token(self):
        self._token = await self._provider.get_valid_token()

    def sync_auth_flow(self, request: httpx.Request):
        """Synchronous auth flow — runs async token acquisition in a new loop."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(asyncio.run, self._ensure_token()).result()
        else:
            asyncio.run(self._ensure_token())

        if self._token:
            request.headers["Authorization"] = f"Bearer {self._token}"
        yield request

    async def async_auth_flow(self, request: httpx.Request):
        """Async auth flow — directly awaits token acquisition."""
        await self._ensure_token()
        if self._token:
            request.headers["Authorization"] = f"Bearer {self._token}"
        yield request


# ---------------------------------------------------------------------------
# Legacy SDK-based provider (for standard MCP dynamic registration)
# ---------------------------------------------------------------------------


class _DynamicRegistrationProvider(OAuthClientProvider):
    """
    Thin wrapper around MCP SDK's OAuthClientProvider for servers that
    support dynamic client registration (no pre-registered credentials).

    The SDK handles PKCE internally for this path.
    """

    pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _urlencode(params: dict) -> str:
    """URL-encode query parameters."""
    from urllib.parse import urlencode

    return urlencode(params)


async def create_oauth_provider(config) -> "httpx.Auth":
    """
    Create an OAuth auth provider from an MCPServerConfig.

    Routing:
      - If config has oauth_client_id → PKCEOAuthProvider (explicit PKCE flow)
        - With client_secret → confidential client + PKCE
        - Without client_secret → public client, PKCE only
      - Otherwise → MCP SDK's OAuthClientProvider (dynamic registration, SDK-managed PKCE)

    Args:
        config: MCPServerConfig with oauth_enabled=True

    Returns:
        httpx.Auth instance for use with MCP transports
    """
    if not OAUTH_AVAILABLE:
        raise RuntimeError("MCP OAuth modules not available. Update mcp SDK: pip install --upgrade mcp")

    redirect_port = config.oauth_redirect_port or 3000
    redirect_uri = f"http://localhost:{redirect_port}/callback"
    storage = FileTokenStorage(server_name=config.name)

    # --- Explicit PKCE flow for pre-registered clients ---
    if config.oauth_client_id:
        client_type = "public" if not config.oauth_client_secret else "confidential"
        logger.info(f"[PKCE] Creating {client_type} OAuth provider for '{config.name}' (client_id={config.oauth_client_id[:8]}...)")

        provider = PKCEOAuthProvider(
            server_name=config.name,
            authorization_endpoint=config.oauth_authorization_endpoint,
            token_endpoint=config.oauth_token_endpoint,
            client_id=config.oauth_client_id,
            client_secret=config.oauth_client_secret,
            redirect_uri=redirect_uri,
            scopes=config.oauth_scopes,
            issuer=config.oauth_issuer,
            storage=storage,
            redirect_port=redirect_port,
            timeout=config.timeout,
        )

        return PKCEAuth(provider)

    # --- Standard MCP dynamic registration (SDK handles PKCE internally) ---
    logger.info(f"Creating dynamic-registration OAuth provider for '{config.name}'")

    async def redirect_handler(authorization_url: str) -> None:
        logger.info(f"Opening browser for OAuth authorization: {authorization_url}")
        webbrowser.open(authorization_url)

    callback_handler = _build_callback_handler(redirect_port)

    client_metadata = OAuthClientMetadata(
        redirect_uris=[redirect_uri],
        token_endpoint_auth_method="none",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        client_name=config.oauth_client_name or "AI Companion MCP Client",
        scope=config.oauth_scopes,
    )

    provider = _DynamicRegistrationProvider(
        server_url=config.url,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
        timeout=config.timeout,
    )

    return provider
