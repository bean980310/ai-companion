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
class MCPServerConfig:
    """Configuration for an MCP server connection"""
    name: str
    url: str
    transport: MCPTransportType = MCPTransportType.SSE
    api_key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    enabled: bool = True
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "transport": self.transport.value,
            "api_key": self.api_key,
            "headers": self.headers,
            "timeout": self.timeout,
            "enabled": self.enabled,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPServerConfig":
        return cls(
            name=data["name"],
            url=data["url"],
            transport=MCPTransportType(data.get("transport", "sse")),
            api_key=data.get("api_key"),
            headers=data.get("headers", {}),
            timeout=data.get("timeout", 30.0),
            enabled=data.get("enabled", True),
            description=data.get("description", "")
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
        return {
            "name": self.name,
            "description": self.description,
            "server_name": self.server_name,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "enum": p.enum
                }
                for p in self.parameters
            ],
            "input_schema": self.input_schema
        }


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
        return {
            "tool_name": self.tool_name,
            "server_name": self.server_name,
            "success": self.success,
            "content": self.content,
            "error": self.error,
            "content_type": self.content_type
        }


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
