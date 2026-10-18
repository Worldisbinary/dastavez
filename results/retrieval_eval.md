Questions: 58 (48 answerable, 10 unanswerable); text source: cascade; embedder: intfloat/multilingual-e5-large; reranker: BAAI/bge-reranker-v2-m3-int8; generator: extractive

| Mode | R@1 | R@3 | R@5 | R@10 | MRR@10 | Cit. precision | Answered | Not-found acc (test) | Not-found acc (all) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lexical | 0.443 | 0.581 | 0.647 | 0.757 | 0.772 | 0.500 | 0.729 | 0.889 | 0.900 |
| dense | 0.477 | 0.722 | 0.791 | 0.858 | 0.873 | 0.597 | 0.646 | 1.000 | 1.000 |
| hybrid | 0.495 | 0.647 | 0.745 | 0.816 | 0.871 | 0.524 | 0.729 | 0.889 | 0.900 |
| hybrid_rerank | 0.575 | 0.798 | 0.848 | 0.867 | 0.953 | 0.667 | 1.000 | 1.000 | 1.000 |
