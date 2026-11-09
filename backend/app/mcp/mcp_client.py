"""
MCP Client Bridge.
Invokes local MCP server instances programmatically from backend workflows.
"""
from typing import Dict, Any
from backend.app.mcp.server import mcp_server

class VeriMindMCPClient:
    async def invoke_mcp_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Calls MCP server tool internally using JSON-RPC standard format."""
        rpc_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": tool_args
            }
        }
        res_json = await mcp_server.handle_json_rpc(dumps_request := str(rpc_request).replace("'", '"'))
        return dict(eval(res_json))

mcp_client = VeriMindMCPClient()