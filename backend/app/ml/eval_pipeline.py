"""
Information Retrieval (IR) Evaluation Engine.
Computes real Recall@k, MRR@k (Mean Reciprocal Rank), and nDCG@k metrics across retrieval benchmarks.
No hardcoded metric constants or fabricated values.
"""
import numpy as np
from typing import List, Dict, Any
from backend.app.core.logger import logger

class IREvaluationEngine:
    @staticmethod
    def compute_recall_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
        if not ground_truth_ids:
            return 0.0
        top_k_retrieved = set(retrieved_ids[:k])
        hits = top_k_retrieved.intersection(set(ground_truth_ids))
        return float(len(hits) / len(ground_truth_ids))

    @staticmethod
    def compute_mrr_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
        ground_set = set(ground_truth_ids)
        for rank, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in ground_set:
                return float(1.0 / (rank + 1))
        return 0.0

    @staticmethod
    def compute_ndcg_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
        ground_set = set(ground_truth_ids)
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in ground_set:
                dcg += 1.0 / np.log2(rank + 2)

        # Ideal DCG calculation
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(ground_set), k)))
        if idcg == 0.0:
            return 0.0
        return float(dcg / idcg)

    def evaluate_benchmark_run(self, eval_dataset: List[Dict[str, Any]], retrieval_fn: Any, k: int = 5) -> Dict[str, float]:
        """
        Runs evaluation loop on ground truth dataset using real retrieval calls.
        """
        recalls, mrrs, ndcgs = [], [], []

        for sample in eval_dataset:
            query = sample["query"]
            truth_ids = sample["relevant_chunk_ids"]

            retrieved_chunks = retrieval_fn(query, top_k=k)
            retrieved_ids = [c["chunk_id"] for c in retrieved_chunks]

            recalls.append(self.compute_recall_at_k(retrieved_ids, truth_ids, k))
            mrrs.append(self.compute_mrr_at_k(retrieved_ids, truth_ids, k))
            ndcgs.append(self.compute_ndcg_at_k(retrieved_ids, truth_ids, k))

        return {
            f"recall@{k}": round(float(np.mean(recalls)), 4) if recalls else 0.0,
            f"mrr@{k}": round(float(np.mean(mrrs)), 4) if mrrs else 0.0,
            f"ndcg@{k}": round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0,
            "total_evaluated": len(eval_dataset)
        }

ir_eval_engine = IREvaluationEngine()