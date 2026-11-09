"""
Model Context Protocol (MCP) Package for VeriMind.
Exposes VeriMind search, ingestion, and claim verification tools to external AI tools (Claude Desktop, Cursor).
"""
from backend.app.mcp.server import VeriMindMCPServer
from backend.app.mcp.mcp_client import VeriMindMCPClient

__all__ = ["VeriMindMCPServer", "VeriMindMCPClient"]