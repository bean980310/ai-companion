# MCP Client Page
# Provides UI for managing MCP server connections and calling external tools

import gradio as gr
import asyncio
import json
from typing import Any, Dict, List, Optional

from ai_companion_core import logger

from src.common.translations import translation_manager, _
from src.common_blocks import create_page_header, get_language_code


# Import MCP client components
from src.mcp.client import MCPClientManager, MCPServerConfig, MCPTool, MCPToolResult
from src.mcp.client.manager import get_mcp_client_manager
from src.mcp.client.oauth import OAUTH_PRESETS


def run_async(coro):
    """Helper to run async functions in sync context"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # Create a new thread for the coroutine
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        return loop.run_until_complete(coro)


# Initialize MCP client manager
mcp_manager = get_mcp_client_manager()


def add_mcp_server(name: str, url: str, transport: str, api_key: str, timeout: float, description: str, command: str, args: str, env: str, oauth_enabled: bool, oauth_client_id: str, oauth_client_secret: str, oauth_issuer: str, oauth_authorization_endpoint: str, oauth_token_endpoint: str, oauth_scopes: str) -> tuple:
    """Add a new MCP server configuration"""
    is_stdio = transport == "stdio"

    if not name:
        return (gr.update(), "Error: Name is required", gr.update())
    if is_stdio and not command:
        return (gr.update(), "Error: Command is required for STDIO transport", gr.update())
    if not is_stdio and not url:
        return (gr.update(), "Error: URL is required for SSE/HTTP transport", gr.update())

    try:
        import shlex

        # Parse args string into list
        args_list = shlex.split(args) if args else []
        # Parse env string (KEY=VALUE per line) into dict
        env_dict = None
        if env and env.strip():
            env_dict = {}
            for line in env.strip().splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    env_dict[k.strip()] = v.strip()

        mcp_manager.add_server(
            name=name,
            transport=transport,
            url=url if not is_stdio else None,
            api_key=api_key if api_key and not is_stdio else None,
            command=command if is_stdio else None,
            args=args_list if is_stdio else None,
            env=env_dict if is_stdio else None,
            timeout=timeout,
            description=description,
            oauth_enabled=oauth_enabled if not is_stdio else False,
            oauth_client_id=oauth_client_id if oauth_client_id else None,
            oauth_client_secret=oauth_client_secret if oauth_client_secret else None,
            oauth_issuer=oauth_issuer if oauth_issuer else None,
            oauth_authorization_endpoint=oauth_authorization_endpoint if oauth_authorization_endpoint else None,
            oauth_token_endpoint=oauth_token_endpoint if oauth_token_endpoint else None,
            oauth_scopes=oauth_scopes if oauth_scopes else None,
        )

        servers = get_servers_display()
        return (gr.update(value=servers), f"Added server: {name}", gr.update())
    except Exception as e:
        logger.error(f"Error adding server: {e}")
        return (gr.update(), f"Error: {str(e)}", gr.update())


def remove_mcp_server(server_name: str) -> tuple:
    """Remove an MCP server"""
    if not server_name:
        return (gr.update(), "Error: Select a server to remove", gr.update())

    try:
        if mcp_manager.remove_server(server_name):
            servers = get_servers_display()
            return (gr.update(value=servers), f"Removed server: {server_name}", gr.update(choices=get_server_names(), value=None))
        else:
            return (gr.update(), f"Server not found: {server_name}", gr.update())
    except Exception as e:
        logger.error(f"Error removing server: {e}")
        return (gr.update(), f"Error: {str(e)}", gr.update())


def connect_to_server(server_name: str) -> tuple:
    """Connect to an MCP server and discover tools"""
    if not server_name:
        return ("Error: Select a server to connect", gr.update(), gr.update())

    try:
        # Check token validity and report status
        token_status = mcp_manager.get_token_status(server_name)
        reauth_msg = ""
        if token_status["oauth_enabled"] and token_status["is_expired"]:
            reauth_msg = " (expired token cleared — re-authenticating)"
        elif token_status["oauth_enabled"] and not token_status["has_token"]:
            reauth_msg = " (no stored token — authentication required)"

        success = run_async(mcp_manager.connect(server_name))

        if success:
            tools = get_tools_display()
            servers = get_servers_display()
            return (
                f"Connected to {server_name}. Discovered {len(mcp_manager.list_tools(server_name))} tools." + (f"{reauth_msg}" if reauth_msg else ""),
                gr.update(value=servers),
                gr.update(value=tools),
            )
        else:
            return (f"Failed to connect to {server_name}{reauth_msg}", gr.update(), gr.update())
    except Exception as e:
        logger.error(f"Error connecting to server: {e}")
        return (f"Error: {str(e)}", gr.update(), gr.update())


def disconnect_from_server(server_name: str) -> tuple:
    """Disconnect from an MCP server"""
    if not server_name:
        return ("Error: Select a server to disconnect", gr.update(), gr.update())

    try:
        run_async(mcp_manager.disconnect(server_name))
        tools = get_tools_display()
        servers = get_servers_display()
        return (f"Disconnected from {server_name}", gr.update(value=servers), gr.update(value=tools))
    except Exception as e:
        logger.error(f"Error disconnecting: {e}")
        return (f"Error: {str(e)}", gr.update(), gr.update())


def connect_all_servers() -> tuple:
    """Connect to all enabled servers"""
    try:
        results = run_async(mcp_manager.connect_all())
        connected = sum(1 for v in results.values() if v)
        failed = sum(1 for v in results.values() if not v)

        tools = get_tools_display()
        servers = get_servers_display()
        return (f"Connected: {connected}, Failed: {failed}", gr.update(value=servers), gr.update(value=tools))
    except Exception as e:
        logger.error(f"Error connecting all: {e}")
        return (f"Error: {str(e)}", gr.update(), gr.update())


def get_servers_display() -> List[List[str]]:
    """Get server list for display in DataFrame"""
    data = []
    for server_id, server in mcp_manager.servers.items():
        if mcp_manager.is_connected(server_id):
            status = "Connected"
        else:
            # Show token status for OAuth-enabled servers
            token_status = mcp_manager.get_token_status(server_id)
            if token_status["oauth_enabled"]:
                if token_status["is_expired"]:
                    status = "Token Expired"
                elif token_status["has_token"]:
                    status = "Disconnected (Token OK)"
                else:
                    status = "Disconnected (No Token)"
            else:
                status = "Disconnected"
        tool_count = len(mcp_manager.list_tools(server_id)) if mcp_manager.is_connected(server_id) else "-"
        endpoint = server.url if server.url else server.command or ""
        data.append([server.name, endpoint, server.transport.value, status, str(tool_count), server.description])
    return data


def get_tools_display() -> List[List[str]]:
    """Get tools list for display in DataFrame"""
    tools = mcp_manager.list_tools()
    data = []
    for tool in tools:
        params = ", ".join([f"{p.name}: {p.type}" for p in tool.parameters[:3]])
        if len(tool.parameters) > 3:
            params += f", ... (+{len(tool.parameters) - 3})"
        data.append([tool.server_name, tool.name, tool.description[:100] + "..." if len(tool.description) > 100 else tool.description, params])
    return data


def get_server_names() -> List[str]:
    """Get list of server IDs for dropdown"""
    return list(mcp_manager.servers.keys())


def get_tool_names() -> List[str]:
    """Get list of full tool names for dropdown"""
    return [t.full_name for t in mcp_manager.list_tools()]


def refresh_server_list() -> tuple:
    """Refresh the server and tool lists"""
    servers = get_servers_display()
    tools = get_tools_display()
    return (gr.update(value=servers), gr.update(value=tools), gr.update(choices=get_server_names()), gr.update(choices=get_tool_names()))


def get_tool_info(tool_name: str) -> str:
    """Get detailed information about a tool"""
    if not tool_name:
        return "Select a tool to see its details"

    tool = mcp_manager.get_tool(tool_name)
    if not tool:
        return f"Tool not found: {tool_name}"

    info = f"## {tool.name}\n\n"
    info += f"**Server:** {tool.server_name}\n\n"
    info += f"**Description:** {tool.description}\n\n"
    info += "**Parameters:**\n\n"

    if tool.parameters:
        for param in tool.parameters:
            required = "*required*" if param.required else "optional"
            default = f" (default: {param.default})" if param.default is not None else ""
            info += f"- `{param.name}` ({param.type}, {required}){default}\n"
            if param.description:
                info += f"  - {param.description}\n"
            if param.enum:
                info += f"  - Options: {', '.join(param.enum)}\n"
    else:
        info += "- No parameters\n"

    info += "\n**Input Schema:**\n```json\n"
    info += json.dumps(tool.input_schema, indent=2)
    info += "\n```"

    return info


def call_mcp_tool(tool_name: str, arguments_json: str) -> str:
    """Call an MCP tool with the given arguments"""
    if not tool_name:
        return "Error: Select a tool to call"

    try:
        arguments = json.loads(arguments_json) if arguments_json.strip() else {}
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON arguments - {e}"

    try:
        result = run_async(mcp_manager.call_tool(tool_name, arguments))

        if result.success:
            output = f"**Tool:** {result.tool_name}\n"
            output += f"**Server:** {result.server_name}\n"
            output += f"**Status:** Success\n\n"
            output += "**Result:**\n"

            if result.content_type == "text":
                output += f"```\n{result.content}\n```"
            elif result.content_type == "image":
                output += f"[Image data received - {len(result.content) if result.content else 0} bytes]"
            else:
                output += f"```\n{result.content}\n```"

            return output
        else:
            return f"**Error:** {result.error}"

    except Exception as e:
        logger.error(f"Error calling tool: {e}")
        return f"Error: {str(e)}"


def update_tool_selector(server_filter: str) -> gr.update:
    """Update tool selector based on server filter"""
    if server_filter and server_filter != "All Servers":
        tools = [t.full_name for t in mcp_manager.list_tools(server_filter)]
    else:
        tools = get_tool_names()
    return gr.update(choices=tools, value=tools[0] if tools else None)


def generate_arguments_template(tool_name: str) -> str:
    """Generate a JSON template for tool arguments"""
    if not tool_name:
        return "{}"

    tool = mcp_manager.get_tool(tool_name)
    if not tool:
        return "{}"

    template = {}
    for param in tool.parameters:
        if param.enum:
            template[param.name] = param.enum[0]
        elif param.default is not None:
            template[param.name] = param.default
        elif param.type == "string":
            template[param.name] = ""
        elif param.type == "number" or param.type == "integer":
            template[param.name] = 0
        elif param.type == "boolean":
            template[param.name] = False
        elif param.type == "array":
            template[param.name] = []
        elif param.type == "object":
            template[param.name] = {}
        else:
            template[param.name] = None

    return json.dumps(template, indent=2)


# Create the page
with gr.Blocks() as demo:
    # Page Header
    # page_header = create_page_header(page_title_key="mcp_client_title")
    # language_dropdown = page_header.language_dropdown

    gr.Markdown("""
    Connect to external MCP servers and use their tools.
    MCP (Model Context Protocol) allows AI applications to share tools and resources.
    """)

    with gr.Tab("Servers"):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Configured Servers")
                servers_table = gr.DataFrame(headers=["Name", "Endpoint", "Transport", "Status", "Tools", "Description"], value=get_servers_display(), interactive=False, wrap=True)

            with gr.Column(scale=1):
                gr.Markdown("### Add Server")
                server_name_input = gr.Textbox(label="Server Name", placeholder="my-mcp-server")
                transport_dropdown = gr.Dropdown(label="Transport", choices=["sse", "streamable-http", "stdio"], value="sse")
                # SSE / HTTP fields
                with gr.Column() as http_fields_column:
                    server_url_input = gr.Textbox(label="URL", placeholder="http://localhost:8080/mcp/sse")
                    api_key_input = gr.Textbox(label="API Key (optional)", type="password")
                # STDIO fields
                with gr.Column(visible=False) as stdio_fields_column:
                    command_input = gr.Textbox(label="Command", placeholder="npx")
                    args_input = gr.Textbox(label="Arguments", placeholder="-y @example/mcp-server", info="Space-separated arguments")
                    env_input = gr.Textbox(label="Environment Variables", placeholder="API_KEY=your-key\nDEBUG=true", lines=3, info="KEY=VALUE per line")
                timeout_input = gr.Number(label="Timeout (seconds)", value=30.0)
                description_input = gr.Textbox(label="Description", placeholder="Description of this server")
                enable_oauth_checkbox = gr.Checkbox(label="Use OAuth")
                with gr.Column(visible=False) as oauth_details_column:
                    gr.Markdown("**PKCE (Proof Key for Code Exchange)** is used for all OAuth flows. Public clients (no secret) rely on PKCE alone.")
                    oauth_provider_preset = gr.Dropdown(label="OAuth Provider Preset", choices=["(Custom)"] + list(OAUTH_PRESETS.keys()), value="(Custom)", interactive=True)
                    oauth_client_id_input = gr.Textbox(label="OAuth Client ID", placeholder="your-client-id")
                    oauth_client_secret_input = gr.Textbox(label="OAuth Client Secret (optional for public clients)", type="password", placeholder="Leave empty for public client (PKCE only)")
                    oauth_issuer_input = gr.Textbox(label="Issuer", placeholder="https://github.com")
                    oauth_auth_endpoint_input = gr.Textbox(label="Authorization Endpoint", placeholder="https://github.com/login/oauth/authorize")
                    oauth_token_endpoint_input = gr.Textbox(label="Token Endpoint", placeholder="https://github.com/login/oauth/access_token")
                    oauth_scopes_input = gr.Textbox(label="OAuth Scopes", placeholder="read:user repo")
                add_server_btn = gr.Button("Add Server", variant="primary")

        with gr.Row():
            server_select = gr.Dropdown(label="Select Server", choices=get_server_names(), interactive=True)
            connect_btn = gr.Button("Connect", variant="primary")
            disconnect_btn = gr.Button("Disconnect", variant="secondary")
            connect_all_btn = gr.Button("Connect All", variant="secondary")
            remove_server_btn = gr.Button("Remove", variant="stop")
            refresh_btn = gr.Button("Refresh", variant="secondary")

        status_text = gr.Textbox(label="Status", interactive=False)

    with gr.Tab("Tools"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Available Tools")
                server_filter = gr.Dropdown(label="Filter by Server", choices=["All Servers"] + get_server_names(), value="All Servers")
                tools_table = gr.DataFrame(headers=["Server", "Tool", "Description", "Parameters"], value=get_tools_display(), interactive=False, wrap=True)

            with gr.Column(scale=1):
                gr.Markdown("### Call Tool")
                tool_select = gr.Dropdown(label="Select Tool", choices=get_tool_names(), interactive=True)
                tool_info = gr.Markdown("Select a tool to see its details")
                generate_template_btn = gr.Button("Generate Template", variant="secondary")
                arguments_input = gr.Code(label="Arguments (JSON)", language="json", value="{}", lines=10)
                call_tool_btn = gr.Button("Call Tool", variant="primary")
                tool_result = gr.Markdown("Tool result will appear here")

    with gr.Tab("Configuration"):
        gr.Markdown("### MCP Servers Configuration")
        gr.Markdown("""
        You can also configure MCP servers by editing the `mcp_servers.json` file in the project root.

        **Example configuration (SSE / HTTP):**
        ```json
        {
          "mcpServers": {
            "example-server": {
              "name": "Example Server",
              "url": "http://localhost:8080/mcp/sse",
              "transport": "sse",
              "enabled": true,
              "description": "Example MCP server"
            }
          }
        }
        ```

        **Example configuration (STDIO):**
        ```json
        {
          "mcpServers": {
            "my-local-server": {
              "name": "My Local Server",
              "transport": "stdio",
              "command": "npx",
              "args": ["-y", "@example/mcp-server"],
              "env": {"API_KEY": "your-key"},
              "enabled": true,
              "description": "Local MCP server via STDIO"
            }
          }
        }
        ```

        **Transport types:**
        - `sse` - Server-Sent Events (recommended for remote servers) — uses `url`, `headers`
        - `streamable-http` - Streamable HTTP transport — uses `url`, `headers`
        - `stdio` - Standard I/O (for local command-based servers) — uses `command`, `args`, `env`

        **Popular MCP Servers:**
        - Hugging Face Hub: Tools for searching models, datasets, and papers
        - GitHub: Repository and issue management
        - Notion: Document and database access
        - Slack: Message and channel management
        """)

        config_display = gr.Code(label="Current Configuration", language="json", value=json.dumps({"mcpServers": {sid: s.to_dict() for sid, s in mcp_manager.servers.items()}}, indent=2), interactive=False)

    # Transport dropdown toggles HTTP vs STDIO fields
    def on_transport_change(transport_val: str):
        is_stdio = transport_val == "stdio"
        return (
            gr.update(visible=not is_stdio),  # http_fields_column
            gr.update(visible=is_stdio),  # stdio_fields_column
            gr.update(visible=not is_stdio),  # oauth checkbox (not for STDIO)
            gr.update(visible=False),  # oauth details column
        )

    transport_dropdown.change(
        fn=on_transport_change,
        inputs=[transport_dropdown],
        outputs=[http_fields_column, stdio_fields_column, enable_oauth_checkbox, oauth_details_column],
    )

    # OAuth checkbox toggles visibility of external OAuth fields
    enable_oauth_checkbox.change(fn=lambda enabled: gr.update(visible=enabled), inputs=[enable_oauth_checkbox], outputs=[oauth_details_column])

    # OAuth provider preset fills in endpoints automatically
    def apply_oauth_preset(preset_name: str):
        if preset_name == "(Custom)" or preset_name not in OAUTH_PRESETS:
            return (gr.update(), gr.update(), gr.update())
        preset = OAUTH_PRESETS[preset_name]
        return (
            gr.update(value=preset["issuer"]),
            gr.update(value=preset["authorization_endpoint"]),
            gr.update(value=preset["token_endpoint"]),
            gr.update(value=preset["default_scopes"]),
        )

    oauth_provider_preset.change(fn=apply_oauth_preset, inputs=[oauth_provider_preset], outputs=[oauth_issuer_input, oauth_auth_endpoint_input, oauth_token_endpoint_input, oauth_scopes_input])

    # Event handlers
    add_server_btn.click(
        fn=add_mcp_server,
        inputs=[
            server_name_input,
            server_url_input,
            transport_dropdown,
            api_key_input,
            timeout_input,
            description_input,
            command_input,
            args_input,
            env_input,
            enable_oauth_checkbox,
            oauth_client_id_input,
            oauth_client_secret_input,
            oauth_issuer_input,
            oauth_auth_endpoint_input,
            oauth_token_endpoint_input,
            oauth_scopes_input,
        ],
        outputs=[servers_table, status_text, server_select],
    ).then(
        fn=lambda: (
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value="sse"),
            gr.update(value=""),
            gr.update(value=30.0),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=False),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
        ),
        outputs=[server_name_input, server_url_input, transport_dropdown, api_key_input, timeout_input, description_input, command_input, args_input, env_input, enable_oauth_checkbox, oauth_client_id_input, oauth_client_secret_input, oauth_auth_endpoint_input, oauth_token_endpoint_input, oauth_scopes_input],
    ).then(fn=lambda: gr.update(choices=get_server_names()), outputs=[server_select])

    remove_server_btn.click(fn=remove_mcp_server, inputs=[server_select], outputs=[servers_table, status_text, server_select])

    connect_btn.click(fn=connect_to_server, inputs=[server_select], outputs=[status_text, servers_table, tools_table]).then(fn=lambda: gr.update(choices=get_tool_names()), outputs=[tool_select])

    disconnect_btn.click(fn=disconnect_from_server, inputs=[server_select], outputs=[status_text, servers_table, tools_table]).then(fn=lambda: gr.update(choices=get_tool_names()), outputs=[tool_select])

    connect_all_btn.click(fn=connect_all_servers, outputs=[status_text, servers_table, tools_table]).then(fn=lambda: gr.update(choices=get_tool_names()), outputs=[tool_select])

    refresh_btn.click(fn=refresh_server_list, outputs=[servers_table, tools_table, server_select, tool_select])

    server_filter.change(fn=update_tool_selector, inputs=[server_filter], outputs=[tool_select])

    tool_select.change(fn=get_tool_info, inputs=[tool_select], outputs=[tool_info])

    generate_template_btn.click(fn=generate_arguments_template, inputs=[tool_select], outputs=[arguments_input])

    call_tool_btn.click(fn=call_mcp_tool, inputs=[tool_select, arguments_input], outputs=[tool_result])

    # Language change event
    def on_mcp_client_language_change(selected_lang: str):
        lang_code = get_language_code(selected_lang)
        translation_manager.set_language(lang_code)
        return [gr.update(value=f"## {_('mcp_client_title')}"), gr.update(label=_("language_select"), info=_("language_info"))]

    # language_dropdown.change(
    #     fn=on_mcp_client_language_change,
    #     inputs=[language_dropdown],
    #     outputs=[page_header.title, language_dropdown]
    # )


if __name__ == "__main__":
    demo.launch()
