"""
Feedback-to-EVAL_SET Auto-Improvement Loop.
Logs user feedback downvotes and automatically constructs new benchmark evaluation instances.
Appends output safely to data/feedback_log.jsonl without overwriting files.
"""
import json
from pathlib import Path
from typing import Dict, Any, List
from backend.app.core.logger import logger

class FeedbackLoopManager:
    FEEDBACK_PATH = Path("data/feedback_log.jsonl")
    EVAL_SET_PATH = Path("data/eval_set.jsonl")

    def __init__(self):
        self.FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

    def log_user_feedback(
        self,
        query: str,
        answer: str,
        retrieved_chunk_ids: List[str],
        rating: int,  # 1 for thumbs up, -1 for downvote
        comment: str = ""
    ) -> Dict[str, Any]:
        """
        Appends user feedback record to JSONL log.
        Auto-promotes downvoted queries (rating = -1) to training eval sets.
        """
        entry = {
            "query": query,
            "answer": answer,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "rating": rating,
            "comment": comment
        }

        # Safe append mode write
        with open(self.FEEDBACK_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        promoted = False
        if rating < 0:
            promoted = self._promote_to_eval_set(query, retrieved_chunk_ids)

        return {
            "status": "recorded",
            "promoted_to_eval_set": promoted,
            "feedback_path": str(self.FEEDBACK_PATH)
        }

    def _promote_to_eval_set(self, query: str, chunk_ids: List[str]) -> bool:
        eval_entry = {
            "query": query,
            "relevant_chunk_ids": chunk_ids,
            "source": "user_feedback_downvote"
        }
        try:
            with open(self.EVAL_SET_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(eval_entry) + "\n")
            logger.info(f"✓ Downvoted query promoted to {self.EVAL_SET_PATH}")
            return True
        except Exception as e:
            logger.error(f"Failed to promote feedback to eval set: {str(e)}")
            return False

feedback_manager = FeedbackLoopManager()