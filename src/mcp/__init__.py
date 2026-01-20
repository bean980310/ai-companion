# MCP (Model Context Protocol) Tools for AI Companion
# This module provides MCP tool wrappers for exposing AI Companion functionality
# to MCP clients like Claude Desktop, Cursor, or other MCP-compatible applications.

from .tools import register_mcp_tools

__all__ = ["register_mcp_tools"]
