Questions: 58 (48 answerable, 10 unanswerable); text source: pypdf; embedder: intfloat/multilingual-e5-large; reranker: BAAI/bge-reranker-v2-m3-int8; generator: extractive

| Mode | R@1 | R@3 | R@5 | R@10 | MRR@10 | Cit. precision | Answered | Not-found acc (test) | Not-found acc (all) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lexical | 0.321 | 0.536 | 0.594 | 0.705 | 0.581 | 0.427 | 0.667 | 0.778 | 0.800 |
| dense | 0.484 | 0.595 | 0.714 | 0.810 | 0.819 | 0.550 | 0.625 | 1.000 | 1.000 |
| hybrid | 0.470 | 0.593 | 0.660 | 0.772 | 0.796 | 0.505 | 0.667 | 0.778 | 0.800 |
| hybrid_rerank | 0.568 | 0.738 | 0.785 | 0.819 | 0.913 | 0.625 | 0.833 | 1.000 | 1.000 |
