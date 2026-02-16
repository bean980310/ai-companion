# MCP OAuth 2.1 Authentication Support
# Provides FileTokenStorage and helper for creating OAuthClientProvider

import asyncio
import json
import os
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
    OAUTH_AVAILABLE = True
except ImportError:
    OAUTH_AVAILABLE = False
    logger.warning("MCP OAuth modules not available. Update mcp SDK.")


class FileTokenStorage:
    """
    File-based token storage implementing MCP SDK's TokenStorage protocol.

    Stores OAuth tokens and client registration info as JSON files
    in ~/.mcp/tokens/<server_name>/.
    """

    def __init__(self, server_name: str, base_dir: Optional[str] = None):
        self.storage_dir = Path(
            base_dir or os.path.expanduser("~/.mcp/tokens")
        ) / server_name
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._tokens_path = self.storage_dir / "tokens.json"
        self._client_info_path = self.storage_dir / "client_info.json"

    async def get_tokens(self) -> "OAuthToken | None":
        """Load stored OAuth tokens."""
        if not self._tokens_path.exists():
            return None
        try:
            data = json.loads(self._tokens_path.read_text(encoding="utf-8"))
            return OAuthToken.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to load OAuth tokens: {e}")
            return None

    async def set_tokens(self, tokens: "OAuthToken") -> None:
        """Persist OAuth tokens."""
        try:
            self._tokens_path.write_text(
                tokens.model_dump_json(indent=2),
                encoding="utf-8"
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
            self._client_info_path.write_text(
                client_info.model_dump_json(indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"Failed to save client info: {e}")


async def create_oauth_provider(config) -> "OAuthClientProvider":
    """
    Create an OAuthClientProvider from an MCPServerConfig.

    This sets up the full OAuth 2.1 Authorization Code + PKCE flow:
    - Opens the user's browser for authorization
    - Runs a local HTTP server to receive the callback
    - Stores tokens locally for reuse

    Args:
        config: MCPServerConfig with oauth_enabled=True

    Returns:
        OAuthClientProvider instance (httpx.Auth subclass)
    """
    if not OAUTH_AVAILABLE:
        raise RuntimeError(
            "MCP OAuth modules not available. "
            "Update mcp SDK: pip install --upgrade mcp"
        )

    redirect_port = config.oauth_redirect_port or 3000
    redirect_uri = f"http://localhost:{redirect_port}/callback"

    client_metadata = OAuthClientMetadata(
        redirect_uris=[redirect_uri],
        token_endpoint_auth_method="none",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        client_name=config.oauth_client_name or "AI Companion MCP Client",
        scope=config.oauth_scopes,
    )

    storage = FileTokenStorage(server_name=config.name)

    # Authorization redirect: open browser
    async def redirect_handler(authorization_url: str) -> None:
        logger.info(f"Opening browser for OAuth authorization: {authorization_url}")
        webbrowser.open(authorization_url)

    # Callback: run a temporary local HTTP server to receive the auth code
    async def callback_handler() -> tuple[str, str | None]:
        """Wait for the OAuth callback and return (auth_code, state)."""
        auth_code_future: asyncio.Future[tuple[str, str | None]] = asyncio.get_event_loop().create_future()

        from aiohttp import web

        async def handle_callback(request: web.Request) -> web.Response:
            code = request.query.get("code")
            state = request.query.get("state")
            error = request.query.get("error")

            if error:
                auth_code_future.set_exception(
                    RuntimeError(f"OAuth error: {error} - {request.query.get('error_description', '')}")
                )
                return web.Response(
                    text="<html><body><h1>Authentication Failed</h1>"
                         f"<p>Error: {error}</p>"
                         "<p>You can close this window.</p></body></html>",
                    content_type="text/html"
                )

            if not code:
                auth_code_future.set_exception(
                    RuntimeError("No authorization code received")
                )
                return web.Response(
                    text="<html><body><h1>Error</h1>"
                         "<p>No authorization code received.</p></body></html>",
                    content_type="text/html"
                )

            auth_code_future.set_result((code, state))
            return web.Response(
                text="<html><body><h1>Authentication Successful!</h1>"
                     "<p>You can close this window and return to the application.</p></body></html>",
                content_type="text/html"
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

    provider = OAuthClientProvider(
        server_url=config.url,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
        timeout=config.timeout,
    )

    return provider
