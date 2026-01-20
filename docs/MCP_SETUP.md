# AI Companion MCP Setup Guide

AI Companion provides both **MCP Server** (expose AI Companion tools to external clients) and **MCP Client** (connect to external MCP servers) capabilities.

## Table of Contents
- [MCP Server](#mcp-server) - Expose AI Companion as an MCP server
- [MCP Client](#mcp-client) - Connect to external MCP servers

---

# MCP Server

AI Companion includes a built-in MCP (Model Context Protocol) server that allows you to use its AI capabilities from MCP-compatible clients like Claude Desktop, Cursor, Cline, and other tools.

## Overview

When you run AI Companion with `mcp_server=True` (enabled by default), the following MCP tools become available:

| Tool | Description |
|------|-------------|
| `chat` | Generate chat completions using various LLM providers |
| `list_models` | List all available AI models (LLM and image) |
| `list_sessions` | List existing chat sessions |
| `get_history` | Retrieve chat history for a session |
| `translate` | Translate text between languages |
| `summarize` | Summarize text in different styles |
| `analyze_image` | Analyze images using vision models |
| `generate_title` | Generate titles for content |
| `generations` | Generate images (from Image Gen page) |

## Starting the MCP Server

The MCP server starts automatically when you run AI Companion:

```bash
python app.py
```

By default, the MCP server will be available at:
- **SSE Endpoint**: `http://localhost:7861/gradio_api/mcp/sse`
- **Web Interface**: `http://localhost:7861/mcp-tools`

### Command Line Options

```bash
# Enable MCP server (default: True)
python app.py --mcp_server True

# Change port
python app.py --port 8080

# Enable remote access
python app.py --listen
```

## Connecting MCP Clients

### Claude Desktop

1. Open Claude Desktop settings
2. Navigate to the MCP configuration section
3. Add the AI Companion server:

```json
{
  "mcpServers": {
    "ai-companion": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "http://localhost:7861/gradio_api/mcp/sse"
      ]
    }
  }
}
```

### Cursor

1. Open Cursor settings
2. Go to MCP configuration
3. Add the server URL: `http://localhost:7861/gradio_api/mcp/sse`

### Claude Code (CLI)

Add to your MCP settings:

```json
{
  "mcpServers": {
    "ai-companion": {
      "command": "npx",
      "args": ["mcp-remote", "http://localhost:7861/gradio_api/mcp/sse"]
    }
  }
}
```

### Cline (VS Code Extension)

1. Open Cline settings in VS Code
2. Add the MCP server configuration
3. Use the SSE endpoint URL

## Using MCP Tools

### Chat Completion

Generate responses using various AI providers:

```
Use the chat tool to ask: "What is the capital of France?"
```

Parameters:
- `message`: The user's input message
- `model`: Model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022')
- `provider`: AI provider ('openai', 'anthropic', 'google-genai', etc.)
- `system_message`: System prompt
- `api_key`: API key for the provider
- `temperature`: Randomness control (0.0-2.0)
- `max_length`: Maximum tokens (-1 for no limit)

### Translation

Translate text between languages:

```
Use the translate tool to translate "Hello, world!" to Korean
```

Parameters:
- `text`: Text to translate
- `source_language`: Source language ('auto' for detection)
- `target_language`: Target language code
- `model`, `provider`, `api_key`: AI configuration

### Summarization

Summarize text in different styles:

```
Use the summarize tool with style 'bullet_points' for this article...
```

Parameters:
- `text`: Text to summarize
- `style`: 'concise', 'detailed', or 'bullet_points'
- `model`, `provider`, `api_key`: AI configuration

### Image Analysis

Analyze images using vision models:

```
Use the analyze_image tool to describe this image
```

Parameters:
- `image_path`: Path to the image file
- `question`: Question about the image
- `model`, `provider`, `api_key`: AI configuration

### Image Generation

Generate images using the existing image generation pipeline:

```
Use the generations tool to create an image of "a sunset over mountains"
```

## Supported Providers

### LLM Providers
- **OpenAI**: GPT-4o, GPT-4, GPT-3.5-Turbo
- **Anthropic**: Claude 3.5, Claude 3
- **Google GenAI**: Gemini Pro, Gemini Ultra
- **Perplexity**: Sonar models
- **X.AI**: Grok models
- **Mistral AI**: Mistral, Mixtral
- **OpenRouter**: Access to multiple providers
- **HuggingFace Inference**: Various open models
- **Local Models**: Transformers, GGUF (llama.cpp), MLX (Apple Silicon)
- **Local Servers**: Ollama, LM Studio

### Image Providers
- **OpenAI**: DALL-E 2, DALL-E 3, GPT-Image
- **Google GenAI**: Gemini Image, Imagen
- **ComfyUI**: Local diffusion models

## API Keys

For cloud providers, you'll need API keys:

1. **Environment Variables** (Recommended):
   ```bash
   export OPENAI_API_KEY="sk-..."
   export ANTHROPIC_API_KEY="sk-ant-..."
   export GOOGLE_API_KEY="..."
   ```

2. **Per-request**: Pass the `api_key` parameter with each tool call

3. **Web Interface**: Enter keys in the API Key fields on the MCP Tools page

## Troubleshooting

### Connection Issues

1. Ensure AI Companion is running
2. Check the port is correct (default: 7861)
3. Verify firewall settings allow connections
4. For remote access, use `--listen` flag

### Tool Not Found

1. Verify MCP server is enabled (`--mcp_server True`)
2. Check the endpoint URL is correct
3. Restart the AI Companion server

### API Errors

1. Verify API keys are valid
2. Check the selected model is available
3. Ensure the provider supports the requested feature

## Web Interface

Access the MCP tools via web browser at:
```
http://localhost:7861/mcp-tools
```

This provides a visual interface for testing all MCP tools before using them programmatically.

## Development

### Adding New MCP Tools

1. Add the function to `src/mcp/tools.py`
2. Include proper type hints and docstrings
3. Register the tool in `app.py` with `api_name`
4. The function will automatically be exposed as an MCP tool

### Tool Requirements

- Functions must have type hints for all parameters
- Docstrings are used as tool descriptions
- Parameter descriptions come from docstring Args section
- Return type annotation determines output format

---

# MCP Client

AI Companion can also connect to external MCP servers and use their tools. This allows you to extend AI Companion's capabilities with tools from other MCP-compatible services.

## Accessing the MCP Client

Navigate to the MCP Client page:
```
http://localhost:7861/mcp-client
```

Or click "MCP Client" in the navigation menu.

## Adding MCP Servers

### Via Web Interface

1. Go to the **Servers** tab
2. Fill in the server details:
   - **Name**: Unique identifier for the server
   - **URL**: Server endpoint (e.g., `http://localhost:8080/mcp/sse`)
   - **Transport**: Connection type (`sse`, `http`, or `stdio`)
   - **API Key**: Optional authentication token
   - **Timeout**: Connection timeout in seconds
   - **Description**: Human-readable description
3. Click **Add Server**

### Via Configuration File

Edit `mcp_servers.json` in the project root:

```json
{
  "servers": [
    {
      "name": "my-server",
      "url": "http://localhost:8080/mcp/sse",
      "transport": "sse",
      "api_key": null,
      "headers": {},
      "timeout": 30.0,
      "enabled": true,
      "description": "My MCP Server"
    }
  ]
}
```

## Connecting to Servers

1. Select a server from the dropdown
2. Click **Connect** to establish connection and discover tools
3. Or click **Connect All** to connect to all enabled servers

Once connected, you'll see:
- Server status changes to "Connected"
- Available tools appear in the **Tools** tab

## Using External Tools

### Via Web Interface

1. Go to the **Tools** tab
2. Select a tool from the dropdown
3. View tool details (parameters, description)
4. Click **Generate Template** to create argument JSON
5. Fill in the arguments
6. Click **Call Tool** to execute

### Tool Information

Each tool displays:
- **Server**: Which MCP server provides the tool
- **Description**: What the tool does
- **Parameters**: Required and optional inputs
- **Input Schema**: JSON schema for arguments

## Transport Types

### SSE (Server-Sent Events)
- Most common for remote servers
- URL format: `http://host:port/path/sse`
- Supports authentication via headers

### HTTP
- Streamable HTTP transport
- URL format: `http://host:port/path/http`
- Good for simple request-response patterns

### STDIO
- For local command-based servers
- URL is the command to run
- Pass arguments via headers: `{"args": "-arg1 value1"}`

## Popular MCP Servers

### Hugging Face Hub
```json
{
  "name": "huggingface",
  "url": "https://huggingface.co/mcp/sse",
  "transport": "sse",
  "description": "Search models, datasets, papers on HF Hub"
}
```

### GitHub (via MCP)
```json
{
  "name": "github",
  "url": "http://localhost:3000/mcp/sse",
  "transport": "sse",
  "api_key": "ghp_your_token",
  "description": "GitHub repository management"
}
```

### Local Filesystem
```json
{
  "name": "filesystem",
  "url": "npx",
  "transport": "stdio",
  "headers": {"args": "@anthropic/mcp-server-filesystem /path/to/dir"},
  "description": "Local file operations"
}
```

## Programmatic Usage

The MCP Client Manager can also be used programmatically:

```python
from src.mcp.client import MCPClientManager

# Get the global manager
from src.mcp.client.manager import get_mcp_client_manager
manager = get_mcp_client_manager()

# Add a server
manager.add_server(
    name="my-server",
    url="http://localhost:8080/mcp/sse",
    transport="sse"
)

# Connect (async)
import asyncio
asyncio.run(manager.connect("my-server"))

# List tools
tools = manager.list_tools()
for tool in tools:
    print(f"{tool.full_name}: {tool.description}")

# Call a tool (async)
result = asyncio.run(manager.call_tool(
    tool_name="search",
    arguments={"query": "hello world"}
))
print(result.content)
```

## Integration with Chat

MCP tools from connected servers can be used in the chat interface:

1. Connect to desired MCP servers
2. In chat, the AI can discover and use available tools
3. Tools are exposed in OpenAI function-calling format

Get tools for LLM:
```python
tools = manager.get_tools_for_llm()
# Returns list of OpenAI-compatible function definitions
```

## Troubleshooting

### Server Won't Connect

1. Verify the server is running
2. Check URL and transport type match the server
3. Ensure API key is correct (if required)
4. Check firewall/network settings

### Tools Not Discovered

1. Verify server supports MCP tool listing
2. Check server logs for errors
3. Try reconnecting

### Tool Call Fails

1. Verify argument format matches schema
2. Check required parameters are provided
3. Review server logs for errors

## Security Considerations

- Store API keys securely (use environment variables)
- Only connect to trusted MCP servers
- Review tool descriptions before calling
- Be cautious with tools that access sensitive data
