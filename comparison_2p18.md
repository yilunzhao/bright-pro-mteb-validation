# Re-validation on the mteb 2.18.1 rebase (2026-07-14)

PR branch `add-bright-pro-retrieval` rebased onto mteb `main` (2.18.1, includes #4684).
Same configs as the May validation. nDCG@10, MTEB vs BRIGHT-Pro's own harness.

| domain | model | MTEB 2.18.1 | BRIGHT-PRO | Δ |
|---|---|---:|---:|---:|
| biology | gte-qwen2 | 0.51111 | 0.50996 | +0.00115 |
| biology | qwen3-embed | 0.43637 | 0.43537 | +0.00100 |
| biology | reasonir | 0.33809 | 0.33322 | +0.00487 |
| biology | embeddinggemma | 0.40375 | 0.40247 | +0.00128 |
| biology | bm25 | 0.19397 | 0.34663 | -0.15266 |
| earth_science | gte-qwen2 | 0.55730 | 0.55145 | +0.00585 |
| earth_science | qwen3-embed | 0.52582 | 0.51592 | +0.00990 |
| earth_science | reasonir | 0.44151 | 0.44187 | -0.00036 |
| earth_science | embeddinggemma | 0.48646 | 0.47893 | +0.00753 |
| earth_science | bm25 | 0.25198 | 0.40569 | -0.15371 |
| economics | gte-qwen2 | 0.28560 | 0.28374 | +0.00186 |
| economics | qwen3-embed | 0.36599 | 0.36772 | -0.00173 |
| economics | reasonir | 0.26110 | 0.26126 | -0.00016 |
| economics | embeddinggemma | 0.25389 | 0.25365 | +0.00024 |
| economics | bm25 | 0.22313 | 0.32055 | -0.09742 |
| psychology | gte-qwen2 | 0.30203 | 0.28687 | +0.01516 |
| psychology | qwen3-embed | 0.38508 | 0.38607 | -0.00099 |
| psychology | reasonir | 0.27697 | 0.27693 | +0.00004 |
| psychology | embeddinggemma | 0.25812 | 0.25899 | -0.00087 |
| psychology | bm25 | 0.17058 | 0.25457 | -0.08399 |
| robotics | gte-qwen2 | 0.35937 | 0.35703 | +0.00234 |
| robotics | qwen3-embed | 0.40843 | 0.41263 | -0.00420 |
| robotics | reasonir | 0.33139 | 0.32699 | +0.00440 |
| robotics | embeddinggemma | 0.28251 | 0.28729 | -0.00478 |
| robotics | bm25 | 0.22331 | 0.31757 | -0.09426 |
| stackoverflow | gte-qwen2 | 0.33228 | 0.33279 | -0.00051 |
| stackoverflow | qwen3-embed | 0.46043 | 0.46069 | -0.00026 |
| stackoverflow | reasonir | 0.38006 | 0.38097 | -0.00091 |
| stackoverflow | embeddinggemma | 0.29070 | 0.29185 | -0.00115 |
| stackoverflow | bm25 | 0.34426 | 0.33049 | +0.01377 |
| sustainable_living | gte-qwen2 | 0.29411 | 0.29174 | +0.00237 |
| sustainable_living | qwen3-embed | 0.34064 | 0.33986 | +0.00078 |
| sustainable_living | reasonir | 0.27702 | 0.27219 | +0.00483 |
| sustainable_living | embeddinggemma | 0.27227 | 0.27490 | -0.00263 |
| sustainable_living | bm25 | 0.23875 | 0.32029 | -0.08154 |

Dense models (28 cells): mean |Δ| = 0.00293, max |Δ| = 0.01516.

ReasonIR-8B is within ±0.005 on every domain — #4684 (merged) fully resolved the
document-prefix gap root-caused in the May analysis (previously +0.02 ~ +0.031).

BM25 cells reflect the known implementation difference between mteb's `bm25s`
and BRIGHT-Pro's BM25 and are not part of the reproduction claim.

gte-Qwen2-7B-instruct requires loader fixes (bf16, `trust_remote_code=True`,
`config.use_cache=False` on transformers >= 4.56); see `scripts/run_mteb_eval.py`.
Per-cell outputs: `results_2p18/`.
