# MCP Client Page
# Provides UI for managing MCP server connections and calling external tools

import gradio as gr
import asyncio
import json
from typing import Any, Dict, List, Optional

from src.common.translations import translation_manager, _
from src.common_blocks import create_page_header, get_language_code
from src import logger

# Import MCP client components
from src.mcp.client import MCPClientManager, MCPServerConfig, MCPTool, MCPToolResult
from src.mcp.client.manager import get_mcp_client_manager


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


def add_mcp_server(
    name: str,
    url: str,
    transport: str,
    api_key: str,
    timeout: float,
    description: str,
    oauth_enabled: bool
) -> tuple:
    """Add a new MCP server configuration"""
    if not name or not url:
        return (
            gr.update(),  # servers list
            "Error: Name and URL are required",
            gr.update()  # tools list
        )

    try:
        mcp_manager.add_server(
            name=name,
            url=url,
            transport=transport,
            api_key=api_key if api_key else None,
            timeout=timeout,
            description=description,
            oauth_enabled=oauth_enabled
        )

        servers = get_servers_display()
        return (
            gr.update(value=servers),
            f"Added server: {name}",
            gr.update()
        )
    except Exception as e:
        logger.error(f"Error adding server: {e}")
        return (
            gr.update(),
            f"Error: {str(e)}",
            gr.update()
        )


def remove_mcp_server(server_name: str) -> tuple:
    """Remove an MCP server"""
    if not server_name:
        return (
            gr.update(),
            "Error: Select a server to remove",
            gr.update()
        )

    try:
        if mcp_manager.remove_server(server_name):
            servers = get_servers_display()
            return (
                gr.update(value=servers),
                f"Removed server: {server_name}",
                gr.update(choices=get_server_names(), value=None)
            )
        else:
            return (
                gr.update(),
                f"Server not found: {server_name}",
                gr.update()
            )
    except Exception as e:
        logger.error(f"Error removing server: {e}")
        return (
            gr.update(),
            f"Error: {str(e)}",
            gr.update()
        )


def connect_to_server(server_name: str) -> tuple:
    """Connect to an MCP server and discover tools"""
    if not server_name:
        return (
            "Error: Select a server to connect",
            gr.update(),
            gr.update()
        )

    try:
        success = run_async(mcp_manager.connect(server_name))

        if success:
            tools = get_tools_display()
            servers = get_servers_display()
            return (
                f"Connected to {server_name}. Discovered {len(mcp_manager.list_tools(server_name))} tools.",
                gr.update(value=servers),
                gr.update(value=tools)
            )
        else:
            return (
                f"Failed to connect to {server_name}",
                gr.update(),
                gr.update()
            )
    except Exception as e:
        logger.error(f"Error connecting to server: {e}")
        return (
            f"Error: {str(e)}",
            gr.update(),
            gr.update()
        )


def disconnect_from_server(server_name: str) -> tuple:
    """Disconnect from an MCP server"""
    if not server_name:
        return (
            "Error: Select a server to disconnect",
            gr.update(),
            gr.update()
        )

    try:
        run_async(mcp_manager.disconnect(server_name))
        tools = get_tools_display()
        servers = get_servers_display()
        return (
            f"Disconnected from {server_name}",
            gr.update(value=servers),
            gr.update(value=tools)
        )
    except Exception as e:
        logger.error(f"Error disconnecting: {e}")
        return (
            f"Error: {str(e)}",
            gr.update(),
            gr.update()
        )


def connect_all_servers() -> tuple:
    """Connect to all enabled servers"""
    try:
        results = run_async(mcp_manager.connect_all())
        connected = sum(1 for v in results.values() if v)
        failed = sum(1 for v in results.values() if not v)

        tools = get_tools_display()
        servers = get_servers_display()
        return (
            f"Connected: {connected}, Failed: {failed}",
            gr.update(value=servers),
            gr.update(value=tools)
        )
    except Exception as e:
        logger.error(f"Error connecting all: {e}")
        return (
            f"Error: {str(e)}",
            gr.update(),
            gr.update()
        )


def get_servers_display() -> List[List[str]]:
    """Get server list for display in DataFrame"""
    servers = mcp_manager.list_servers()
    data = []
    for server in servers:
        status = "Connected" if mcp_manager.is_connected(server.name) else "Disconnected"
        tool_count = len(mcp_manager.list_tools(server.name)) if mcp_manager.is_connected(server.name) else "-"
        data.append([
            server.name,
            server.url,
            server.transport.value,
            status,
            str(tool_count),
            server.description
        ])
    return data


def get_tools_display() -> List[List[str]]:
    """Get tools list for display in DataFrame"""
    tools = mcp_manager.list_tools()
    data = []
    for tool in tools:
        params = ", ".join([f"{p.name}: {p.type}" for p in tool.parameters[:3]])
        if len(tool.parameters) > 3:
            params += f", ... (+{len(tool.parameters) - 3})"
        data.append([
            tool.server_name,
            tool.name,
            tool.description[:100] + "..." if len(tool.description) > 100 else tool.description,
            params
        ])
    return data


def get_server_names() -> List[str]:
    """Get list of server names for dropdown"""
    return [s.name for s in mcp_manager.list_servers()]


def get_tool_names() -> List[str]:
    """Get list of full tool names for dropdown"""
    return [t.full_name for t in mcp_manager.list_tools()]


def refresh_server_list() -> tuple:
    """Refresh the server and tool lists"""
    servers = get_servers_display()
    tools = get_tools_display()
    return (
        gr.update(value=servers),
        gr.update(value=tools),
        gr.update(choices=get_server_names()),
        gr.update(choices=get_tool_names())
    )


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
    page_header = create_page_header(page_title_key="mcp_client_title")
    language_dropdown = page_header.language_dropdown

    gr.Markdown("""
    Connect to external MCP servers and use their tools.
    MCP (Model Context Protocol) allows AI applications to share tools and resources.
    """)

    with gr.Tab("Servers"):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Configured Servers")
                servers_table = gr.DataFrame(
                    headers=["Name", "URL", "Transport", "Status", "Tools", "Description"],
                    value=get_servers_display(),
                    interactive=False,
                    wrap=True
                )

            with gr.Column(scale=1):
                gr.Markdown("### Add Server")
                server_name_input = gr.Textbox(label="Server Name", placeholder="my-mcp-server")
                server_url_input = gr.Textbox(
                    label="URL",
                    placeholder="http://localhost:8080/mcp/sse"
                )
                transport_dropdown = gr.Dropdown(
                    label="Transport",
                    choices=["sse", "streamable-http", "stdio"],
                    value="sse"
                )
                api_key_input = gr.Textbox(
                    label="API Key (optional)",
                    type="password"
                )
                timeout_input = gr.Number(label="Timeout (seconds)", value=30.0)
                description_input = gr.Textbox(
                    label="Description",
                    placeholder="Description of this server"
                )
                enable_oauth_checkbox = gr.Checkbox(
                    label="Use OAuth"
                )
                add_server_btn = gr.Button("Add Server", variant="primary")

        with gr.Row():
            server_select = gr.Dropdown(
                label="Select Server",
                choices=get_server_names(),
                interactive=True
            )
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
                server_filter = gr.Dropdown(
                    label="Filter by Server",
                    choices=["All Servers"] + get_server_names(),
                    value="All Servers"
                )
                tools_table = gr.DataFrame(
                    headers=["Server", "Tool", "Description", "Parameters"],
                    value=get_tools_display(),
                    interactive=False,
                    wrap=True
                )

            with gr.Column(scale=1):
                gr.Markdown("### Call Tool")
                tool_select = gr.Dropdown(
                    label="Select Tool",
                    choices=get_tool_names(),
                    interactive=True
                )
                tool_info = gr.Markdown("Select a tool to see its details")
                generate_template_btn = gr.Button("Generate Template", variant="secondary")
                arguments_input = gr.Code(
                    label="Arguments (JSON)",
                    language="json",
                    value="{}",
                    lines=10
                )
                call_tool_btn = gr.Button("Call Tool", variant="primary")
                tool_result = gr.Markdown("Tool result will appear here")

    with gr.Tab("Configuration"):
        gr.Markdown("### MCP Servers Configuration")
        gr.Markdown("""
        You can also configure MCP servers by editing the `mcp_servers.json` file in the project root.

        **Example configuration:**
        ```json
        {
          "servers": [
            {
              "name": "example-server",
              "url": "http://localhost:8080/mcp/sse",
              "transport": "sse",
              "enabled": true,
              "description": "Example MCP server"
            }
          ]
        }
        ```

        **Transport types:**
        - `sse` - Server-Sent Events (recommended for remote servers)
        - `http` - HTTP transport
        - `stdio` - Standard I/O (for local command-based servers)

        **Popular MCP Servers:**
        - Hugging Face Hub: Tools for searching models, datasets, and papers
        - GitHub: Repository and issue management
        - Notion: Document and database access
        - Slack: Message and channel management
        """)

        config_display = gr.Code(
            label="Current Configuration",
            language="json",
            value=json.dumps(
                {"servers": [s.to_dict() for s in mcp_manager.list_servers()]},
                indent=2
            ),
            interactive=False
        )

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
            enable_oauth_checkbox
        ],
        outputs=[servers_table, status_text, server_select]
    ).then(
        fn=lambda: (gr.update(value=""), gr.update(value=""), gr.update(value="sse"),
                   gr.update(value=""), gr.update(value=30.0), gr.update(value="")),
        outputs=[
            server_name_input, server_url_input, transport_dropdown,
            api_key_input, timeout_input, description_input
        ]
    ).then(
        fn=lambda: gr.update(choices=get_server_names()),
        outputs=[server_select]
    )

    remove_server_btn.click(
        fn=remove_mcp_server,
        inputs=[server_select],
        outputs=[servers_table, status_text, server_select]
    )

    connect_btn.click(
        fn=connect_to_server,
        inputs=[server_select],
        outputs=[status_text, servers_table, tools_table]
    ).then(
        fn=lambda: gr.update(choices=get_tool_names()),
        outputs=[tool_select]
    )

    disconnect_btn.click(
        fn=disconnect_from_server,
        inputs=[server_select],
        outputs=[status_text, servers_table, tools_table]
    ).then(
        fn=lambda: gr.update(choices=get_tool_names()),
        outputs=[tool_select]
    )

    connect_all_btn.click(
        fn=connect_all_servers,
        outputs=[status_text, servers_table, tools_table]
    ).then(
        fn=lambda: gr.update(choices=get_tool_names()),
        outputs=[tool_select]
    )

    refresh_btn.click(
        fn=refresh_server_list,
        outputs=[servers_table, tools_table, server_select, tool_select]
    )

    server_filter.change(
        fn=update_tool_selector,
        inputs=[server_filter],
        outputs=[tool_select]
    )

    tool_select.change(
        fn=get_tool_info,
        inputs=[tool_select],
        outputs=[tool_info]
    )

    generate_template_btn.click(
        fn=generate_arguments_template,
        inputs=[tool_select],
        outputs=[arguments_input]
    )

    call_tool_btn.click(
        fn=call_mcp_tool,
        inputs=[tool_select, arguments_input],
        outputs=[tool_result]
    )

    # Language change event
    def on_mcp_client_language_change(selected_lang: str):
        lang_code = get_language_code(selected_lang)
        translation_manager.set_language(lang_code)
        return [
            gr.update(value=f"## {_('mcp_client_title')}"),
            gr.update(label=_('language_select'), info=_('language_info'))
        ]

    language_dropdown.change(
        fn=on_mcp_client_language_change,
        inputs=[language_dropdown],
        outputs=[page_header.title, language_dropdown]
    )


if __name__ == "__main__":
    demo.launch()
