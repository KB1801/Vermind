"""
Standalone Stdio Entry Point for VeriMind MCP Server.
Allows Cursor / Claude Desktop to run `python backend/app/mcp/verimind_mcp_server.py` directly.
"""
import sys
import asyncio
from backend.app.mcp.server import mcp_server

async def run_stdio_mcp_loop():
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        if not line:
            break
        raw_req = line.decode('utf-8').strip()
        if raw_req:
            res = await mcp_server.handle_json_rpc(raw_req)
            sys.stdout.write(res + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    asyncio.run(run_stdio_mcp_loop())