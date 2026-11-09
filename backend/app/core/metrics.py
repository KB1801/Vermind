"""
Prometheus Observability Metrics Collector.
Exposes counters and histograms for query latencies and vector operations.
"""
from prometheus_client import Counter, Histogram, Gauge

QUERY_COUNTER = Counter('verimind_queries_total', 'Total user query requests processed', ['pattern'])
QUERY_LATENCY = Histogram('verimind_query_latency_seconds', 'Latency of query execution in seconds', ['pattern'])
VECTOR_COUNT = Gauge('verimind_indexed_chunks_total', 'Total active vector chunks in database')