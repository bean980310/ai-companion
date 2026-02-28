# MCP Client Manager
# Manages connections to external MCP servers and tool execution

import asyncio
import json
import os
import traceback
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
from contextlib import AsyncExitStack

from src import logger

from .models import (
    MCPServerConfig,
    MCPTool,
    MCPToolResult,
    MCPToolParameter,
    MCPTransportType,
    MCPResource,
    MCPPrompt
)

# MCP SDK imports
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP SDK not available. Install with: pip install mcp")

from .oauth import create_oauth_provider

class MCPClientManager:
    """
    Manages multiple MCP server connections and provides unified tool access.

    This class handles:
    - Connecting to MCP servers (SSE, HTTP, STDIO)
    - Discovering available tools from servers
    - Executing tools and returning results
    - Managing server configurations
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the MCP Client Manager.

        Args:
            config_path: Path to the MCP servers configuration file.
                        Defaults to 'mcp_servers.json' in the project root.
        """
        self.config_path = config_path or Path.home() / ".ai-companion" / "mcp_servers.json"
        self.servers: Dict[str, MCPServerConfig] = {}
        self.sessions: Dict[str, ClientSession] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.prompts: Dict[str, MCPPrompt] = {}
        self._connected_servers: set = set()

        # Load saved configurations
        self._load_config()

    def _load_config(self):
        """Load server configurations from file"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for server_data in data.get("servers", []):
                        config = MCPServerConfig.from_dict(server_data)
                        self.servers[config.name] = config
                logger.info(f"Loaded {len(self.servers)} MCP server configurations")
            except Exception as e:
                logger.error(f"Error loading MCP config: {e}")

    def _save_config(self):
        """Save server configurations to file"""
        try:
            data = {
                "servers": [server.to_dict() for server in self.servers.values()]
            }
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.servers)} MCP server configurations")
        except Exception as e:
            logger.error(f"Error saving MCP config: {e}")

    def add_server(
        self,
        name: str,
        url: str,
        transport: str = "sse",
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        description: str = "",
        oauth_enabled: bool = True,
        oauth_client_id: Optional[str] = None,
        oauth_client_secret: Optional[str] = None,
        oauth_authorization_endpoint: Optional[str] = None,
        oauth_token_endpoint: Optional[str] = None,
        oauth_scopes: Optional[str] = None,
        save: bool = True
    ) -> MCPServerConfig:
        """
        Add a new MCP server configuration.

        Args:
            name: Unique name for the server
            url: Server URL (for SSE/HTTP) or command (for STDIO)
            transport: Transport type ('sse', 'http', or 'stdio')
            api_key: Optional API key for authentication
            headers: Optional HTTP headers
            timeout: Connection timeout in seconds
            description: Human-readable description
            oauth_enabled: Whether to use OAuth authentication
            oauth_client_id: Pre-registered OAuth client ID (for GitHub, Google, etc.)
            oauth_client_secret: Pre-registered OAuth client secret
            oauth_authorization_endpoint: OAuth authorization endpoint URL
            oauth_token_endpoint: OAuth token endpoint URL
            oauth_scopes: Space-separated OAuth scopes
            save: Whether to persist the configuration

        Returns:
            The created MCPServerConfig
        """
        config = MCPServerConfig(
            name=name,
            url=url,
            transport=MCPTransportType(transport),
            api_key=api_key,
            headers=headers or {},
            timeout=timeout,
            description=description,
            oauth_enabled=oauth_enabled,
            oauth_client_name="AI Companion MCP Client",
            oauth_client_id=oauth_client_id,
            oauth_client_secret=oauth_client_secret,
            oauth_authorization_endpoint=oauth_authorization_endpoint,
            oauth_token_endpoint=oauth_token_endpoint,
            oauth_scopes=oauth_scopes,
        )
        self.servers[name] = config

        if save:
            self._save_config()

        logger.info(f"Added MCP server: {name} ({url})")
        return config

    def remove_server(self, name: str, save: bool = True) -> bool:
        """
        Remove an MCP server configuration.

        Args:
            name: Name of the server to remove
            save: Whether to persist the change

        Returns:
            True if removed, False if not found
        """
        if name in self.servers:
            # Disconnect if connected
            if name in self._connected_servers:
                asyncio.create_task(self.disconnect(name))

            del self.servers[name]
            # Remove associated tools
            self.tools = {
                k: v for k, v in self.tools.items()
                if v.server_name != name
            }

            if save:
                self._save_config()

            logger.info(f"Removed MCP server: {name}")
            return True
        return False

    def get_server(self, name: str) -> Optional[MCPServerConfig]:
        """Get a server configuration by name"""
        return self.servers.get(name)

    def list_servers(self) -> List[MCPServerConfig]:
        """List all configured servers"""
        return list(self.servers.values())

    async def connect(self, server_name: str) -> bool:
        """
        Connect to an MCP server and discover its tools.

        Args:
            server_name: Name of the server to connect to

        Returns:
            True if connected successfully
        """
        if not MCP_AVAILABLE:
            logger.error("MCP SDK not available")
            return False

        config = self.servers.get(server_name)
        if not config:
            logger.error(f"Server not found: {server_name}")
            return False

        if not config.enabled:
            logger.warning(f"Server {server_name} is disabled")
            return False

        try:
            if config.transport == MCPTransportType.SSE:
                await self._connect_sse(config)
            elif config.transport == MCPTransportType.STDIO:
                await self._connect_stdio(config)
            elif config.transport == MCPTransportType.HTTP:
                await self._connect_http(config)
            else:
                logger.error(f"Unsupported transport: {config.transport}")
                return False

            self._connected_servers.add(server_name)
            logger.info(f"Connected to MCP server: {server_name}")
            return True

        except Exception as e:
            logger.error(f"Error connecting to {server_name}: {e}\n\n{traceback.format_exc()}")
            return False

    async def _connect_sse(self, config: MCPServerConfig):
        """Connect to an SSE-based MCP server"""
        headers = config.headers.copy()
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"

        # OAuth 2.1 authentication
        auth = None
        if config.oauth_enabled:
            auth = await create_oauth_provider(config)

        async with sse_client(config.url, headers=headers, auth=auth) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self.sessions[config.name] = session
                await self._discover_capabilities(config.name, session)

    async def _connect_stdio(self, config: MCPServerConfig):
        """Connect to a STDIO-based MCP server"""
        # For STDIO, url is the command, headers can contain args
        command = config.url
        # Handle args gently (could be list or string)
        args_raw = config.headers.get("args")
        if isinstance(args_raw, list):
            args = [str(a) for a in args_raw]
        elif isinstance(args_raw, str):
            args = args_raw.split()
        else:
            args = []

        # Handle env gently (could be dict)
        env_raw = config.headers.get("env")
        if isinstance(env_raw, dict):
            env = os.environ.copy()
            env.update({str(k): str(v) for k, v in env_raw.items()})
        else:
            env = None

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self.sessions[config.name] = session
                await self._discover_capabilities(config.name, session)

    async def _connect_http(self, config: MCPServerConfig):
        """Connect to an HTTP-based MCP server"""
        headers = config.headers.copy()

        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"

        # OAuth 2.1 authentication
        http_client = None
        if config.oauth_enabled:
            auth = await create_oauth_provider(config)
            http_client = create_mcp_http_client(headers=headers, auth=auth)

        async with streamable_http_client(config.url, http_client=http_client) as (
            read, write, _get_session_id
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self.sessions[config.name] = session
                await self._discover_capabilities(config.name, session)

    async def _discover_capabilities(self, server_name: str, session: ClientSession):
        """Discover tools, resources, and prompts from a connected server"""
        # Discover tools
        try:
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                params = []
                input_schema = tool.inputSchema or {}

                # Parse parameters from input schema
                properties = input_schema.get("properties", {})
                required = input_schema.get("required", [])

                for param_name, param_info in properties.items():
                    params.append(MCPToolParameter(
                        name=param_name,
                        type=param_info.get("type", "string"),
                        description=param_info.get("description", ""),
                        required=param_name in required,
                        default=param_info.get("default"),
                        enum=param_info.get("enum")
                    ))

                mcp_tool = MCPTool(
                    name=tool.name,
                    description=tool.description or "",
                    server_name=server_name,
                    parameters=params,
                    input_schema=input_schema
                )
                self.tools[mcp_tool.full_name] = mcp_tool
                logger.debug(f"Discovered tool: {mcp_tool.full_name}")

            logger.info(f"Discovered {len(tools_result.tools)} tools from {server_name}")
        except Exception as e:
            logger.warning(f"Error discovering tools from {server_name}: {e}")

        # Discover resources
        try:
            resources_result = await session.list_resources()
            for resource in resources_result.resources:
                mcp_resource = MCPResource(
                    uri=str(resource.uri),
                    name=resource.name,
                    description=resource.description or "",
                    mime_type=resource.mimeType or "application/json",
                    server_name=server_name
                )
                self.resources[f"{server_name}__{resource.name}"] = mcp_resource
            logger.info(f"Discovered {len(resources_result.resources)} resources from {server_name}")
        except Exception as e:
            logger.debug(f"No resources available from {server_name}: {e}")

        # Discover prompts
        try:
            prompts_result = await session.list_prompts()
            for prompt in prompts_result.prompts:
                args = []
                for arg in prompt.arguments or []:
                    args.append(MCPToolParameter(
                        name=arg.name,
                        type="string",
                        description=arg.description or "",
                        required=arg.required or False
                    ))
                mcp_prompt = MCPPrompt(
                    name=prompt.name,
                    description=prompt.description or "",
                    arguments=args,
                    server_name=server_name
                )
                self.prompts[f"{server_name}__{prompt.name}"] = mcp_prompt
            logger.info(f"Discovered {len(prompts_result.prompts)} prompts from {server_name}")
        except Exception as e:
            logger.debug(f"No prompts available from {server_name}: {e}")

    async def disconnect(self, server_name: str):
        """Disconnect from an MCP server"""
        if server_name in self.sessions:
            try:
                # Session cleanup
                del self.sessions[server_name]
            except Exception as e:
                logger.warning(f"Error disconnecting from {server_name}: {e}")

        self._connected_servers.discard(server_name)

        # Remove associated tools
        self.tools = {
            k: v for k, v in self.tools.items()
            if v.server_name != server_name
        }

        logger.info(f"Disconnected from MCP server: {server_name}")

    async def connect_all(self) -> Dict[str, bool]:
        """
        Connect to all enabled servers.

        Returns:
            Dict mapping server names to connection success status
        """
        results = {}
        for name, config in self.servers.items():
            if config.enabled:
                results[name] = await self.connect(name)
        return results

    async def disconnect_all(self):
        """Disconnect from all servers"""
        for server_name in list(self._connected_servers):
            await self.disconnect(server_name)

    def list_tools(self, server_name: Optional[str] = None) -> List[MCPTool]:
        """
        List available tools, optionally filtered by server.

        Args:
            server_name: Optional server name to filter by

        Returns:
            List of available tools
        """
        if server_name:
            return [t for t in self.tools.values() if t.server_name == server_name]
        return list(self.tools.values())

    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """
        Get a tool by its full name (server__tool).

        Args:
            tool_name: Full tool name or just tool name (searches all servers)

        Returns:
            The MCPTool if found
        """
        # Try exact match first
        if tool_name in self.tools:
            return self.tools[tool_name]

        # Try finding by short name
        for full_name, tool in self.tools.items():
            if tool.name == tool_name:
                return tool

        return None

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        server_name: Optional[str] = None
    ) -> MCPToolResult:
        """
        Call an MCP tool with the given arguments.

        Args:
            tool_name: Name of the tool to call
            arguments: Dictionary of arguments for the tool
            server_name: Optional server name (required if tool_name is ambiguous)

        Returns:
            MCPToolResult containing the result or error
        """
        if not MCP_AVAILABLE:
            return MCPToolResult(
                tool_name=tool_name,
                server_name=server_name or "unknown",
                success=False,
                error="MCP SDK not available"
            )

        # Find the tool
        tool = None
        if server_name:
            full_name = f"{server_name}__{tool_name}"
            tool = self.tools.get(full_name)
        else:
            tool = self.get_tool(tool_name)

        if not tool:
            return MCPToolResult(
                tool_name=tool_name,
                server_name=server_name or "unknown",
                success=False,
                error=f"Tool not found: {tool_name}"
            )

        # Check if server is connected
        if tool.server_name not in self.sessions:
            # Try to connect
            if not await self.connect(tool.server_name):
                return MCPToolResult(
                    tool_name=tool_name,
                    server_name=tool.server_name,
                    success=False,
                    error=f"Cannot connect to server: {tool.server_name}"
                )

        session = self.sessions[tool.server_name]

        try:
            result = await session.call_tool(tool.name, arguments)

            # Process the result
            content = []
            content_type = "text"

            for item in result.content:
                if hasattr(item, "text"):
                    content.append(item.text)
                elif hasattr(item, "data"):
                    # Image or binary data
                    content.append(item.data)
                    content_type = "image" if hasattr(item, "mimeType") and "image" in item.mimeType else "binary"
                else:
                    content.append(str(item))

            # Join text content or return first item for non-text
            final_content = "\n".join(content) if content_type == "text" else content[0] if content else None

            return MCPToolResult(
                tool_name=tool.name,
                server_name=tool.server_name,
                success=not result.isError if hasattr(result, "isError") else True,
                content=final_content,
                content_type=content_type
            )

        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return MCPToolResult(
                tool_name=tool.name,
                server_name=tool.server_name,
                success=False,
                error=str(e)
            )

    async def read_resource(self, uri: str, server_name: str) -> Optional[str]:
        """
        Read a resource from an MCP server.

        Args:
            uri: Resource URI
            server_name: Server name

        Returns:
            Resource content or None
        """
        if server_name not in self.sessions:
            logger.error(f"Server not connected: {server_name}")
            return None

        try:
            session = self.sessions[server_name]
            result = await session.read_resource(uri)
            if result.contents:
                return result.contents[0].text if hasattr(result.contents[0], "text") else str(result.contents[0])
            return None
        except Exception as e:
            logger.error(f"Error reading resource {uri}: {e}")
            return None

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: Dict[str, str],
        server_name: str
    ) -> Optional[str]:
        """
        Get a prompt from an MCP server.

        Args:
            prompt_name: Name of the prompt
            arguments: Arguments for the prompt
            server_name: Server name

        Returns:
            Rendered prompt or None
        """
        if server_name not in self.sessions:
            logger.error(f"Server not connected: {server_name}")
            return None

        try:
            session = self.sessions[server_name]
            result = await session.get_prompt(prompt_name, arguments)
            if result.messages:
                return result.messages[0].content.text if hasattr(result.messages[0].content, "text") else str(result.messages[0].content)
            return None
        except Exception as e:
            logger.error(f"Error getting prompt {prompt_name}: {e}")
            return None

    def is_connected(self, server_name: str) -> bool:
        """Check if a server is connected"""
        return server_name in self._connected_servers

    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """
        Get tools in a format suitable for LLM function calling.

        Returns:
            List of tool definitions in OpenAI function format
        """
        tools = []
        for tool in self.tools.values():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.full_name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            })
        return tools


# Global instance for easy access
_mcp_client_manager: Optional[MCPClientManager] = None


def get_mcp_client_manager() -> MCPClientManager:
    """Get the global MCP client manager instance"""
    global _mcp_client_manager
    if _mcp_client_manager is None:
        _mcp_client_manager = MCPClientManager()
    return _mcp_client_manager
