"""
Locust Load Testing Suite for VeriMind API.
Simulates concurrent user load against vector search and RAG synthesis endpoints.
"""
from locust import HttpUser, task, between

class VeriMindLoadUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def test_search_endpoint(self):
        self.client.post("/api/v1/search/chunks", json={
            "query": "What are VeriMind system requirements?",
            "top_k": 5,
            "tenant_id": "locust_test"
        })

    @task(1)
    def test_rag_query_endpoint(self):
        self.client.post("/api/v1/rag/query", json={
            "query": "Explain atomic claim verification and EII index.",
            "pattern": "auto",
            "top_k": 3,
            "tenant_id": "locust_test"
        })