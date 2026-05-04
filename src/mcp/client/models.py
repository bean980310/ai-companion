# MCP Client Data Models

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal
from enum import Enum


class MCPTransportType(str, Enum):
    """Supported MCP transport types"""

    SSE = "sse"
    HTTP = "streamable-http"
    STDIO = "stdio"


@dataclass
class MCPServerStreamableHTTPConfig:
    url: str
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class MCPServerSSEConfig:
    url: str
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class MCPServerSTDIOConfig:
    command: str
    args: List[str] = field(default_factory=list)
    env: Optional[Dict[str, str]] = None


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection"""

    name: str
    transport: MCPTransportType = MCPTransportType.SSE
    enabled: bool = True
    description: str = ""
    timeout: float = 30.0
    # SSE / HTTP fields
    url: Optional[str] = None
    api_key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    # STDIO fields
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Optional[Dict[str, str]] = None
    # OAuth 2.1 settings
    oauth_enabled: bool = False
    oauth_client_name: Optional[str] = None
    oauth_scopes: Optional[str] = None
    oauth_redirect_port: int = 3000
    # External OAuth client settings (GitHub, Google, etc.)
    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[str] = None
    oauth_issuer: Optional[str] = None
    oauth_authorization_endpoint: Optional[str] = None
    oauth_token_endpoint: Optional[str] = None
    # Set to False when the authorization server does not support PKCE (RFC 7636).
    oauth_use_pkce: bool = True

    def to_dict(self) -> Dict[str, Any]:
        base = {
            "name": self.name,
            "transport": self.transport.value,
            "enabled": self.enabled,
            "description": self.description,
            "timeout": self.timeout,
        }
        if self.transport == MCPTransportType.STDIO:
            base["command"] = self.command
            base["args"] = self.args
            if self.env is not None:
                base["env"] = self.env
        else:
            base["url"] = self.url
            base["api_key"] = self.api_key
            base["headers"] = self.headers
            base["oauth_enabled"] = self.oauth_enabled
            base["oauth_client_name"] = self.oauth_client_name
            base["oauth_scopes"] = self.oauth_scopes
            base["oauth_redirect_port"] = self.oauth_redirect_port
            base["oauth_client_id"] = self.oauth_client_id
            base["oauth_client_secret"] = self.oauth_client_secret
            base["oauth_issuer"] = self.oauth_issuer
            base["oauth_authorization_endpoint"] = self.oauth_authorization_endpoint
            base["oauth_token_endpoint"] = self.oauth_token_endpoint
            base["oauth_use_pkce"] = self.oauth_use_pkce
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPServerConfig":
        transport = MCPTransportType(data.get("transport", "sse"))
        return cls(
            name=data.get("name", ""),
            transport=transport,
            enabled=data.get("enabled", True),
            description=data.get("description", ""),
            timeout=data.get("timeout", 30.0),
            # SSE / HTTP fields
            url=data.get("url"),
            api_key=data.get("api_key"),
            headers=data.get("headers", {}),
            # STDIO fields
            command=data.get("command"),
            args=data.get("args", []),
            env=data.get("env"),
            # OAuth
            oauth_enabled=data.get("oauth_enabled", False),
            oauth_client_name=data.get("oauth_client_name"),
            oauth_scopes=data.get("oauth_scopes"),
            oauth_redirect_port=data.get("oauth_redirect_port", 3000),
            oauth_client_id=data.get("oauth_client_id"),
            oauth_client_secret=data.get("oauth_client_secret"),
            oauth_issuer=data.get("oauth_issuer"),
            oauth_authorization_endpoint=data.get("oauth_authorization_endpoint"),
            oauth_token_endpoint=data.get("oauth_token_endpoint"),
            oauth_use_pkce=data.get("oauth_use_pkce", True),
        )


@dataclass
class MCPToolParameter:
    """Parameter definition for an MCP tool"""

    name: str
    type: str
    description: str = ""
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None


@dataclass
class MCPTool:
    """Representation of an MCP tool"""

    name: str
    description: str
    server_name: str
    parameters: List[MCPToolParameter] = field(default_factory=list)
    input_schema: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        """Full tool name including server prefix"""
        return f"{self.server_name}__{self.name}"

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "server_name": self.server_name, "parameters": [{"name": p.name, "type": p.type, "description": p.description, "required": p.required, "default": p.default, "enum": p.enum} for p in self.parameters], "input_schema": self.input_schema}


@dataclass
class MCPToolResult:
    """Result from calling an MCP tool"""

    tool_name: str
    server_name: str
    success: bool
    content: Any = None
    error: Optional[str] = None
    content_type: str = "text"  # text, image, json, etc.

    def to_dict(self) -> Dict[str, Any]:
        return {"tool_name": self.tool_name, "server_name": self.server_name, "success": self.success, "content": self.content, "error": self.error, "content_type": self.content_type}


@dataclass
class MCPResource:
    """Representation of an MCP resource"""

    uri: str
    name: str
    description: str = ""
    mime_type: str = "application/json"
    server_name: str = ""


@dataclass
class MCPPrompt:
    """Representation of an MCP prompt template"""

    name: str
    description: str = ""
    arguments: List[MCPToolParameter] = field(default_factory=list)
    server_name: str = ""
