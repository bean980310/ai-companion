# MCP OAuth 2.1 Authentication Support
# Provides FileTokenStorage, PreRegisteredOAuthProvider,
# and helper for creating OAuthClientProvider

import asyncio
import json
import os
import time
import webbrowser
from pathlib import Path
from typing import Optional

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


class FileTokenStorage:
    """
    File-based token storage implementing MCP SDK's TokenStorage protocol.

    Stores OAuth tokens and client registration info as JSON files
    in ~/.mcp/tokens/<server_name>/.

    Token validity is tracked via a metadata file that records
    the save timestamp and expires_in value from the token response.
    """

    # Grace period in seconds — treat tokens as expired this much earlier
    # to avoid using tokens that are about to expire mid-request.
    EXPIRY_GRACE_SECONDS = 60

    def __init__(self, server_name: str, base_dir: Optional[str] = None):
        self.server_name = server_name
        self.storage_dir = Path(base_dir or os.path.expanduser("~/.mcp/tokens")) / server_name
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._tokens_path = self.storage_dir / "tokens.json"
        self._client_info_path = self.storage_dir / "client_info.json"
        self._metadata_path = self.storage_dir / "token_metadata.json"

    def _load_metadata(self) -> Optional[dict]:
        """Load token metadata (saved_at, expires_in)."""
        if not self._metadata_path.exists():
            return None
        try:
            return json.loads(self._metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_metadata(self, expires_in: Optional[int]) -> None:
        """Save token metadata alongside the token file."""
        metadata = {
            "saved_at": time.time(),
            "expires_in": expires_in,
        }
        try:
            self._metadata_path.write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"Failed to save token metadata for {self.server_name}: {e}")

    def is_token_expired(self) -> bool:
        """
        Check whether the stored token has expired.

        Uses the saved_at timestamp + expires_in from the token response.
        If metadata is missing, falls back to the token file's mtime
        combined with the expires_in parsed from the token itself.

        Returns True if expired or if validity cannot be determined
        (missing metadata and no expires_in in token).
        """
        if not self._tokens_path.exists():
            return True

        metadata = self._load_metadata()

        # Try metadata first (most reliable)
        if metadata and metadata.get("saved_at") and metadata.get("expires_in"):
            saved_at = metadata["saved_at"]
            expires_in = metadata["expires_in"]
            elapsed = time.time() - saved_at
            if elapsed >= (expires_in - self.EXPIRY_GRACE_SECONDS):
                logger.info(
                    f"Token for {self.server_name} has expired "
                    f"(elapsed={elapsed:.0f}s, expires_in={expires_in}s)"
                )
                return True
            return False

        # Fallback: read expires_in from the token file + file mtime
        try:
            data = json.loads(self._tokens_path.read_text(encoding="utf-8"))
            expires_in = data.get("expires_in")
            if expires_in is not None:
                file_mtime = self._tokens_path.stat().st_mtime
                elapsed = time.time() - file_mtime
                if elapsed >= (expires_in - self.EXPIRY_GRACE_SECONDS):
                    logger.info(
                        f"Token for {self.server_name} has expired (fallback check, "
                        f"elapsed={elapsed:.0f}s, expires_in={expires_in}s)"
                    )
                    return True
                return False
        except Exception:
            pass

        # Cannot determine expiry — treat as valid to avoid unnecessary re-auth
        # (the server will reject it if actually expired)
        return False

    def clear_tokens(self) -> None:
        """Delete stored tokens, metadata, and client info to force re-authentication."""
        for path in (self._tokens_path, self._metadata_path):
            if path.exists():
                try:
                    path.unlink()
                    logger.info(f"Deleted {path.name} for {self.server_name}")
                except Exception as e:
                    logger.error(f"Failed to delete {path.name} for {self.server_name}: {e}")

    async def get_tokens(self) -> "OAuthToken | None":
        """Load stored OAuth tokens, returning None if expired."""
        if not self._tokens_path.exists():
            return None

        # Validate token expiry before returning
        if self.is_token_expired():
            logger.warning(
                f"Stored token for {self.server_name} has expired. "
                f"Clearing tokens to trigger re-authentication."
            )
            self.clear_tokens()
            return None

        try:
            data = json.loads(self._tokens_path.read_text(encoding="utf-8"))
            return OAuthToken.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to load OAuth tokens: {e}")
            return None

    async def set_tokens(self, tokens: "OAuthToken") -> None:
        """Persist OAuth tokens with expiry metadata."""
        try:
            self._tokens_path.write_text(tokens.model_dump_json(indent=2), encoding="utf-8")

            # Save metadata for expiry tracking
            expires_in = getattr(tokens, "expires_in", None)
            self._save_metadata(expires_in)

            logger.info(
                f"Saved OAuth tokens for {self.server_name}"
                + (f" (expires_in={expires_in}s)" if expires_in else "")
            )
        except Exception as e:
            logger.error(f"Failed to save OAuth tokens: {e}")

    async def get_client_info(self) -> "OAuthClientInformationFull | None":
        """Load stored client registration info."""
        if not self._client_info_path.exists():
            return None
        try:
            data = json.loads(self._client_info_path.read_text(encoding="utf-8"))
            return OAuthClientInformationFull.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to load client info: {e}")
            return None

    async def set_client_info(self, client_info: "OAuthClientInformationFull") -> None:
        """Persist client registration info."""
        try:
            self._client_info_path.write_text(client_info.model_dump_json(indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save client info: {e}")


class PreRegisteredOAuthProvider(OAuthClientProvider):
    """
    OAuth provider for external providers (GitHub, Google, etc.)
    that require pre-registered client credentials.

    Extends OAuthClientProvider to bypass dynamic client registration
    while keeping the Authorization Code + PKCE flow intact.
    """

    def __init__(
        self,
        server_url: str,
        client_metadata: "OAuthClientMetadata",
        storage: "FileTokenStorage",
        redirect_handler,
        callback_handler,
        timeout: float,
        client_id: str,
        client_secret: str,
        issuer: str,
        authorization_endpoint: str,
        token_endpoint: str,
    ):
        super().__init__(
            server_url=server_url,
            client_metadata=client_metadata,
            storage=storage,
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
            timeout=timeout,
        )
        # Pre-registered client info — bypasses dynamic registration
        self._fixed_client_info = OAuthClientInformationFull(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uris=client_metadata.redirect_uris,
            token_endpoint_auth_method=client_metadata.token_endpoint_auth_method,
            grant_types=client_metadata.grant_types,
            response_types=client_metadata.response_types,
            scope=client_metadata.scope,
        )
        # Pre-configured OAuth metadata for the external provider
        self._fixed_oauth_metadata = OAuthMetadata(
            issuer=issuer,
            authorization_endpoint=authorization_endpoint,
            token_endpoint=token_endpoint,
        )

    async def _initialize(self) -> None:
        """Load stored tokens and inject pre-configured client info & metadata."""
        self.context.current_tokens = await self.context.storage.get_tokens()
        self.context.client_info = self._fixed_client_info
        self.context.oauth_metadata = self._fixed_oauth_metadata
        self._initialized = True


def _build_callback_handler(redirect_port: int):
    """Build a reusable OAuth callback handler."""

    async def callback_handler() -> tuple[str, str | None]:
        """Wait for the OAuth callback and return (auth_code, state)."""
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


async def create_oauth_provider(config) -> "OAuthClientProvider":
    """
    Create an OAuthClientProvider from an MCPServerConfig.

    If the config has oauth_client_id set, creates a PreRegisteredOAuthProvider
    for external OAuth providers (GitHub, Google, etc.) that require
    pre-registered client credentials.

    Otherwise, creates a standard OAuthClientProvider that uses
    MCP's dynamic client registration.

    Args:
        config: MCPServerConfig with oauth_enabled=True

    Returns:
        OAuthClientProvider instance (httpx.Auth subclass)
    """
    if not OAUTH_AVAILABLE:
        raise RuntimeError("MCP OAuth modules not available. Update mcp SDK: pip install --upgrade mcp")

    redirect_port = config.oauth_redirect_port or 3000
    redirect_uri = f"http://localhost:{redirect_port}/callback"

    storage = FileTokenStorage(server_name=config.name)

    # Authorization redirect: open browser
    async def redirect_handler(authorization_url: str) -> None:
        logger.info(f"Opening browser for OAuth authorization: {authorization_url}")
        webbrowser.open(authorization_url)

    callback_handler = _build_callback_handler(redirect_port)

    # External OAuth provider (GitHub, Google, etc.)
    if config.oauth_client_id:
        auth_method = "client_secret_post"
        client_metadata = OAuthClientMetadata(
            redirect_uris=[redirect_uri],
            token_endpoint_auth_method=auth_method,
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            client_name=config.oauth_client_name or "AI Companion MCP Client",
            scope=config.oauth_scopes,
        )

        provider = PreRegisteredOAuthProvider(
            server_url=config.url,
            client_metadata=client_metadata,
            storage=storage,
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
            timeout=config.timeout,
            client_id=config.oauth_client_id,
            client_secret=config.oauth_client_secret or "",
            issuer=config.oauth_issuer,
            authorization_endpoint=config.oauth_authorization_endpoint,
            token_endpoint=config.oauth_token_endpoint,
        )
        return provider

    # Standard MCP OAuth (dynamic client registration)
    client_metadata = OAuthClientMetadata(
        redirect_uris=[redirect_uri],
        token_endpoint_auth_method="none",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        client_name=config.oauth_client_name or "AI Companion MCP Client",
        scope=config.oauth_scopes,
    )

    provider = OAuthClientProvider(
        server_url=config.url,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
        timeout=config.timeout,
    )

    return provider
