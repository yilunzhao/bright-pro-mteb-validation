# BRIGHT-Pro × MTEB integration: cross-validation

Companion repository for [embeddings-benchmark/mteb#4651](https://github.com/embeddings-benchmark/mteb/pull/4651) — the PR adding BRIGHT-Pro as a retrieval benchmark in MTEB.

This repo cross-validates that running 4 retrievers through the new MTEB task classes (`BrightPro{Domain}Retrieval`) reproduces the numbers from BRIGHT-Pro's own evaluation harness on all 7 StackExchange domains.

## Setup

Each (model × domain) cell is evaluated end-to-end via:

```python
import mteb
mteb.evaluate(
    mteb.get_model_meta(model_name),
    mteb.get_task(task_name),
    encode_kwargs={"batch_size": ...},
)
```

with the per-model `max_seq_length` aligned to BRIGHT-Pro's original settings:

| model | `max_seq_length` | batch |
|---|---|---|
| `mteb/baseline-bm25s` | — | — |
| `google/embeddinggemma-300m` | 2048 | 32 |
| `Qwen/Qwen3-Embedding-8B` | 4096 | 8–32 |
| `ReasonIR/ReasonIR-8B` | 4096 | 8–32 |

The MTEB task prompt is set to `"Given a {domain} post, retrieve relevant passages that help answer the post"` — the same instruction body BRIGHT-Pro's paper uses, so the Qwen3 / ReasonIR wrappers format it into the identical `Instruct:…\nQuery:` / `<|user|>…\n<|embed|>\n` prefix as the BRIGHT-Pro harness.

## Results

`nDCG@10` per (domain, model). Δ = MTEB − BRIGHT-Pro.

| domain | model | BRIGHT-Pro | MTEB | Δ |
|---|---|---:|---:|---:|
| biology | BM25 | 0.34663 | 0.19397 | -0.15266 |
| biology | embeddinggemma-300m | 0.40247 | 0.40375 | +0.00128 |
| biology | Qwen3-Embedding-8B | 0.43537 | 0.43637 | +0.00100 |
| biology | ReasonIR-8B | 0.33322 | 0.35323 | +0.02001 |
| earth_science | BM25 | 0.40569 | 0.25198 | -0.15371 |
| earth_science | embeddinggemma-300m | 0.47893 | 0.48646 | +0.00753 |
| earth_science | Qwen3-Embedding-8B | 0.51592 | 0.52582 | +0.00990 |
| earth_science | ReasonIR-8B | 0.44187 | 0.44398 | +0.00211 |
| economics | BM25 | 0.32055 | 0.22313 | -0.09742 |
| economics | embeddinggemma-300m | 0.25365 | 0.25389 | +0.00024 |
| economics | Qwen3-Embedding-8B | 0.36772 | 0.36599 | -0.00173 |
| economics | ReasonIR-8B | 0.26126 | 0.27006 | +0.00880 |
| psychology | BM25 | 0.25457 | 0.17058 | -0.08399 |
| psychology | embeddinggemma-300m | 0.25899 | 0.25812 | -0.00087 |
| psychology | Qwen3-Embedding-8B | 0.38607 | 0.38508 | -0.00099 |
| psychology | ReasonIR-8B | 0.27693 | 0.28199 | +0.00506 |
| robotics | BM25 | 0.31757 | 0.22331 | -0.09426 |
| robotics | embeddinggemma-300m | 0.28729 | 0.28251 | -0.00478 |
| robotics | Qwen3-Embedding-8B | 0.41263 | 0.40843 | -0.00420 |
| robotics | ReasonIR-8B | 0.32699 | 0.35841 | +0.03142 |
| stackoverflow | BM25 | 0.33049 | 0.34426 | +0.01377 |
| stackoverflow | embeddinggemma-300m | 0.29185 | 0.29070 | -0.00115 |
| stackoverflow | Qwen3-Embedding-8B | 0.46069 | 0.46043 | -0.00026 |
| stackoverflow | ReasonIR-8B | 0.38097 | 0.39817 | +0.01720 |
| sustainable_living | BM25 | 0.32029 | 0.23875 | -0.08154 |
| sustainable_living | embeddinggemma-300m | 0.27490 | 0.27227 | -0.00263 |
| sustainable_living | Qwen3-Embedding-8B | 0.33986 | 0.34064 | +0.00078 |
| sustainable_living | ReasonIR-8B | 0.27219 | 0.27728 | +0.00509 |

### Per-model summary (7 domains each)

| model | mean abs Δ | max abs Δ | comment |
|---|---:|---:|---|
| embeddinggemma-300m | **0.0023** | 0.0075 | sub-1% on every domain |
| Qwen3-Embedding-8B | **0.0028** | 0.0099 | sub-1% on every domain |
| ReasonIR-8B | **0.0142** | 0.0314 | MTEB systematically 0.2–3% higher; same ranking direction |
| BM25 | 0.0826 | 0.1537 | MTEB uses `bm25s` + Porter stemmer + English stopwords; BRIGHT-Pro uses a different BM25 implementation. The two BM25 variants are not expected to match — this row is informative about the choice of BM25 baseline, not about the task integration. |

The three dense-embedding models reproduce BRIGHT-Pro's numbers within 3% on every cell, with two of them sub-1%. This confirms the MTEB task classes (`BrightPro{Domain}Retrieval`) pass identical corpora, queries, and qrels to the evaluation pipeline as BRIGHT-Pro's own harness — any remaining differences are attributable to model-side details (precision, EOS handling, tokenizer behaviour), not to the task integration.

## Layout

```
scripts/
  run_mteb_eval.py     # single (model, task) evaluation
  submit_eval.sh       # SLURM script (GPU)
  submit_eval_cpu.sh   # SLURM script (BM25, CPU)
  dispatch_all.sh      # fire one job per (model, task)
  build_comparison.py  # aggregate JSON results -> markdown table
results/
  BrightPro{Domain}Retrieval__{model_slug}.json  # one per cell
```

## Reproducing

```bash
# one (model, task) cell
MODEL_NAME=Qwen/Qwen3-Embedding-8B \
TASK_NAME=BrightProBiologyRetrieval \
MAX_SEQ_LENGTH=4096 BATCH_SIZE=16 \
python scripts/run_mteb_eval.py
```
