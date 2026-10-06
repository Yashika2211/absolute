| Method | Recall@50 | NDCG@10 | MRR@50 | New-item Recall@50 | New-item NDCG@10 |
|---|---|---|---|---|---|
| Popularity (14d) | 0.0207 | 0.0052 | 0.0047 | 0.0158 | 0.0035 |
| Recently viewed + popularity | 0.3988 | 0.3796 | 0.3998 | 0.0158 | 0.0035 |
| Two-tower, exact | 0.5191 | 0.3738 | 0.3773 | 0.3009 | 0.1197 |
| Two-tower, FAISS HNSW | 0.5191 | 0.3738 | 0.3773 | 0.3009 | 0.1197 |
| **Two-tower + recent → LightGBM** | 0.5677 | 0.4331 | 0.4369 | 0.3040 | 0.1229 |
