"""
Chat Assistant Pipeline Manager.
Coordinates multi-turn dialogue state with RAG pattern execution.
"""
from typing import Dict, Any
from backend.app.rag_patterns.router import rag_router
from backend.app.core.memory import memory_manager

class ChatAssistant:
    async def handle_chat_message(
        self,
        session_id: str,
        user_message: str,
        tenant_id: str = "default_tenant"
    ) -> Dict[str, Any]:

        # Save user message to history
        memory_manager.add_message(session_id, "user", user_message)

        # Route query through RAG router
        response = await rag_router.route_and_execute(
            query=user_message,
            requested_pattern="auto",
            tenant_id=tenant_id
        )

        # Save assistant response to history
        memory_manager.add_message(session_id, "assistant", response.answer)

        return {
            "session_id": session_id,
            "answer": response.answer,
            "pattern_used": response.pattern_used,
            "evidence_integrity_index": response.evidence_integrity_index,
            "history_length": len(memory_manager.get_history(session_id))
        }

chat_assistant = ChatAssistant()