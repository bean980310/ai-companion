# MCP Tool-Calling Agent
# Bridges MCP client tools into the ai-companion-llm-backend tool-calling loop.
# The heavy lifting (schema conversion, OpenAI-compatible tool loop) lives in
# the backend; this module only resolves clients and adapts MCP tool results.

from typing import Any, Dict, List, Optional

from ai_companion_core import logger
from ai_companion_core.environ_manager import load_env_variables

from ai_companion_llm_backend import (
    ToolResult,
    build_tool_specs_from_mcp,
    provider_supports_tools,
    run_tool_loop,
    run_tool_loop_responses,
)

from .client.manager import MCPClientManager, get_mcp_client_manager
from .client.models import MCPToolResult
from .runtime import run_mcp_coro

# Providers that expose an OpenAI-compatible chat.completions tool-calling API.
_OPENAI_COMPATIBLE_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openai": {"base_url": None, "env": "OPENAI_API_KEY"},
    "xai": {"base_url": "https://api.x.ai/v1", "env": "XAI_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "env": "OPENROUTER_API_KEY"},
    "mistralai": {"base_url": "https://api.mistral.ai/v1", "env": "MISTRAL_API_KEY"},
    "lmstudio": {"base_url": "http://localhost:1234/v1", "env": "LM_API_KEY", "default_key": "not-needed"},
    "vllm-api": {"base_url": "http://localhost:8000/v1", "env": "VLLM_API_KEY", "default_key": "not-needed"},
}

MAX_TOOL_ITERATIONS = 6

_TOOL_USAGE_HINT = (
    "You can call external tools to answer the user. "
    "Use the provided functions when they help fulfill the request. "
    "After receiving tool results, produce the final answer in the user's language."
)


def _normalize_provider(provider: str) -> str:
    """Map the app's provider ids onto backend provider ids."""
    if provider == "vllm":
        return "vllm-api"
    return provider


def supports_mcp_tools(provider: str) -> bool:
    """Whether the given provider can drive MCP tool calling.

    True when either the app's OpenAI-compatible client map handles it or the
    backend advertises native tool support.
    """
    if not provider:
        return False
    if provider.startswith("custom:"):
        return True
    if provider in _OPENAI_COMPATIBLE_PROVIDERS:
        return True
    return provider_supports_tools(_normalize_provider(provider))


def _content_to_openai(content: Any) -> Any:
    """Convert internal multimodal content into OpenAI chat content format."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: List[Dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            parts.append({"type": "text", "text": item})
            continue
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text" or "text" in item:
            parts.append({"type": "text", "text": item.get("text", "")})
        elif item_type == "image":
            url = item.get("url") or item.get("image_url")
            if url:
                parts.append({"type": "image_url", "image_url": {"url": url}})

    if not parts:
        return ""
    if len(parts) == 1 and parts[0]["type"] == "text":
        return parts[0]["text"]
    return parts


def _build_messages(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    for msg in history:
        role = msg.get("role")
        if role not in ("system", "user", "assistant"):
            continue
        messages.append({"role": role, "content": _content_to_openai(msg.get("content", ""))})
    return messages


def _mcp_result_to_tool_result(result: MCPToolResult) -> ToolResult:
    return ToolResult(
        success=result.success,
        content=result.content,
        error=result.error,
        content_type=result.content_type,
    )


def _resolve_client(provider: str, api_key: Optional[str]):
    import openai

    base_url: Optional[str] = None
    key = api_key

    if provider.startswith("custom:"):
        from src.common.custom_providers import get_custom_provider

        profile = get_custom_provider(provider[len("custom:") :])
        if not profile:
            raise ValueError(f"Unknown custom provider: {provider}")
        base_url = profile.get("base_url")
        key = profile.get("api_key") or key
    else:
        spec = _OPENAI_COMPATIBLE_PROVIDERS.get(provider)
        if spec is None:
            raise ValueError(f"Provider '{provider}' does not support MCP tool calling")
        base_url = spec.get("base_url")
        if not key:
            key = load_env_variables(spec["env"])
        if not key:
            key = spec.get("default_key")

    if not key and provider not in ("lmstudio", "vllm-api"):
        raise ValueError(f"API key is required for provider '{provider}'")

    return openai.OpenAI(api_key=key or "not-needed", base_url=base_url)


def _make_executor(manager: MCPClientManager):
    def executor(tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        result = run_mcp_coro(manager.call_tool(tool_name, arguments))
        return _mcp_result_to_tool_result(result)

    return executor


def _select_tools(manager: MCPClientManager, tool_names: Optional[List[str]]) -> List[Any]:
    all_tools = manager.list_tools()
    if tool_names:
        selected = set(tool_names)
        all_tools = [t for t in all_tools if t.full_name in selected or t.name in selected]
    return all_tools


def run_tool_agent(
    history: List[Dict[str, Any]],
    selected_model: str,
    provider: str,
    api_key: Optional[str] = None,
    tool_names: Optional[List[str]] = None,
    temperature: float = 0.6,
    max_tokens: int = 4096,
    max_iterations: int = MAX_TOOL_ITERATIONS,
    manager: Optional[MCPClientManager] = None,
) -> str:
    """
    Run the backend tool-calling loop using MCP tools.

    Args:
        history: Conversation history in the app's internal message format.
        selected_model: Model id sent to the provider.
        provider: LLM provider id (must be tool-capable).
        api_key: Optional API key; falls back to env / stored profile.
        tool_names: Optional list of MCP tool full names to expose. When None,
            all discovered tools are exposed.
        temperature: Sampling temperature.
        max_tokens: Max response tokens per completion.
        max_iterations: Safety cap on tool-calling rounds.
        manager: Optional MCP client manager (defaults to the global one).

    Returns:
        The assistant's final text answer (already stripped).
    """
    manager = manager or get_mcp_client_manager()

    tools = _select_tools(manager, tool_names)
    if not tools:
        raise RuntimeError("No MCP tools are available. Connect an MCP server and select tools.")

    tool_specs = build_tool_specs_from_mcp(tools)
    client = _resolve_client(provider, api_key)
    messages = _build_messages(history)

    # OpenAI's official endpoint uses the Responses API; the rest use
    # OpenAI-compatible chat.completions.
    if provider == "openai":
        system_prompt = next((str(m.get("content", "")) for m in history if m.get("role") == "system"), None)
        input_items = [m for m in messages if m.get("role") != "system"]
        hint = _TOOL_USAGE_HINT
        if system_prompt:
            system_prompt = system_prompt + "\n\n" + hint
        else:
            system_prompt = hint
        return run_tool_loop_responses(
            client=client,
            model=selected_model,
            input_items=input_items,
            tool_specs=tool_specs,
            executor=_make_executor(manager),
            instructions=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_iterations=max_iterations,
        )

    return run_tool_loop(
        client=client,
        model=selected_model,
        messages=messages,
        tool_specs=tool_specs,
        executor=_make_executor(manager),
        temperature=temperature,
        max_tokens=max_tokens,
        max_iterations=max_iterations,
        system_tool_hint=_TOOL_USAGE_HINT,
    )


def list_selectable_mcp_tools() -> List[Dict[str, str]]:
    """Return connected MCP tools as simple dicts for UI dropdowns."""
    manager = get_mcp_client_manager()
    return [
        {
            "value": t.full_name,
            "label": f"{t.server_name}: {t.name}",
            "description": t.description or "",
        }
        for t in manager.list_tools()
    ]
