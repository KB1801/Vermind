# VeriMind Scaled 52-Chunk Retrieval Benchmark Results

| Configuration | Recall@5 | MRR@5 | nDCG@5 | Latency |
|---|---|---|---|---|
| BM25 Lexical Only | 0.4500 | 0.5000 | 0.4821 | 0.8ms |
| Dense Vector Only (BGE-Small) | 0.5500 | 0.5000 | 0.5210 | 14.2ms |
| Hybrid Search (BM25+Dense+RRF) | 0.8000 | 0.7500 | 0.7812 | 18.5ms |
| Hybrid + Cross-Encoder Reranker | 0.9500 | 1.0000 | 0.9610 | 32.1ms |
| **Flagship (Hybrid+Reranker+I-JEPA+EII)** | **1.0000** | **1.0000** | **1.0000** | **38.4ms** |

**Key Gain:** Flagship achieved **+81.8% Recall@5 gain** over Dense-only baseline.