"""
Native Model Context Protocol (MCP) Server Implementation.
Exposes VeriMind RAG tools via standard JSON-RPC 2.0 protocol over stdio / HTTP.
"""
import json
import sys
from typing import Dict, Any, List
from backend.app.retrieval.qdrant_store import qdrant_store
from backend.app.verification.validator import evidence_validator
from backend.app.core.logger import logger

class VeriMindMCPServer:
    def __init__(self):
        self.tools = {
            "search_verimind": self.tool_search_verimind,
            "verify_claim": self.tool_verify_claim,
            "ingest_text": self.tool_ingest_text
        }

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Returns JSON schema for tools advertised by this MCP server."""
        return [
            {
                "name": "search_verimind",
                "description": "Performs Hybrid BM25+Dense retrieval over VeriMind indexed enterprise documents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query or natural language question"},
                        "top_k": {"type": "integer", "default": 5, "description": "Number of relevant evidence chunks to return"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "verify_claim",
                "description": "Decomposes and verifies a factual claim against retrieved evidence chunks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string", "description": "The specific factual claim to verify"}
                    },
                    "required": ["claim"]
                }
            },
            {
                "name": "ingest_text",
                "description": "Ingests raw text into VeriMind vector memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Document title or source label"},
                        "content": {"type": "string", "description": "Body text to index"}
                    },
                    "required": ["title", "content"]
                }
            }
        ]

    async def tool_search_verimind(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        results = qdrant_store.hybrid_search(query, top_k=top_k)
        return {"query": query, "chunks_found": len(results), "chunks": results}

    async def tool_verify_claim(self, claim: str) -> Dict[str, Any]:
        chunks = qdrant_store.hybrid_search(claim, top_k=3)
        claims_res, eii = await evidence_validator.validate_answer_claims(claim, chunks)
        return {
            "claim": claim,
            "evidence_integrity_index": eii,
            "verifications": [c.model_dump() for c in claims_res]
        }

    async def tool_ingest_text(self, title: str, content: str) -> Dict[str, Any]:
        chunk_data = [{"chunk_id": f"mcp_{hash(title)}", "doc_id": title, "text": content}]
        count = qdrant_store.upsert_chunks(chunk_data)
        return {"title": title, "status": "indexed", "chunks_stored": count}

    async def handle_json_rpc(self, request_raw: str) -> str:
        """Processes incoming MCP JSON-RPC 2.0 requests."""
        try:
            req = json.loads(request_raw)
            method = req.get("method")
            req_id = req.get("id")

            if method == "tools/list":
                return json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"tools": self.get_tool_definitions()}})
            
            elif method == "tools/call":
                params = req.get("params", {})
                name = params.get("name")
                args = params.get("arguments", {})

                if name in self.tools:
                    tool_func = self.tools[name]
                    res = await tool_func(**args)
                    return json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res)}]} })
                else:
                    return json.dumps({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Tool '{name}' not found"}})

            return json.dumps({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}})
        except Exception as e:
            return json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {str(e)}"}})

mcp_server = VeriMindMCPServer()