"""
Conversation Session State & Multi-Turn Memory Manager.
Maintains short-term chat message buffers across user sessions.
"""
from typing import List, Dict, Any

class ConversationMemoryManager:
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].append({"role": role, "content": content})
        
        # Trim older turns to keep memory bounded
        if len(self.sessions[session_id]) > (self.max_turns * 2):
            self.sessions[session_id] = self.sessions[session_id][-(self.max_turns * 2):]

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self.sessions.get(session_id, [])

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

memory_manager = ConversationMemoryManager()