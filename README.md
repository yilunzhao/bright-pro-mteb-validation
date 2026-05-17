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

## Root cause analysis of the ReasonIR-8B gap

The ~1–3 nDCG-point gap for ReasonIR-8B above is not noise. We traced it to a separate MTEB bug, **independent of this BrightPro task integration**, in [`mteb/models/abs_encoder.py`](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/models/abs_encoder.py)'s `get_task_instruction`:

```python
if self.instruction_template and len(instruction) > 0:
    return self.format_instruction(instruction, prompt_type)
return instruction
```

When a task defines `prompt={"query": "..."}` without a `"document"` key (BRIGHT-Pro, BRIGHT, BRIGHT v1.1, and many other retrieval tasks), the document-side `instruction` is `""`. The `len(instruction) > 0` gate then causes `get_task_instruction` to return `""` directly — the model's `instruction_template` is **never invoked for documents**.

For models whose `instruction_template` is a callable that emits a non-empty prefix on empty input — most prominently ReasonIR-8B and GritLM-7B (`<|embed|>\n`), also Octen, Sarashina v2 (`text: `), BMRetriever — the document-side prefix the model was *trained* with is silently dropped. Documents get encoded without the prefix.

### Diagnostics

The `scripts/verify_reasonir_*.py` + `results/diagnostics/verify_reasonir_*.json` files trace the gap step by step. The decisive observations on `BrightProBiologyRetrieval`:

| diagnostic | finding |
|---|---|
| `verify_reasonir_full_eval.py` | Re-running ReasonIR via BRIGHT-Pro's own `AutoModel + custom .encode` on the full 60K corpus in our env: **nDCG@10 = 0.33755** (paper saved: 0.33322). |
| `verify_reasonir_full_paths.py` | Side-by-side `AutoModel + custom .encode` vs MTEB's `InstructSentenceTransformerModel`: per-doc cosine **median 0.98, min 0.71** on 60K docs — the embeddings genuinely diverge. |
| `verify_reasonir_strip.py` | MTEB's corpus `.strip()` preprocessing contributes only +0.002 nDCG. Not the main source. |
| `verify_reasonir_chunking.py` | MTEB's 50K corpus chunking contributes 0 nDCG (Mode 1 = Mode 2 = 0.33792). Ruled out. |
| `verify_reasonir_wrapper_vs_direct.py` | Wrapper vs direct `st.encode`: **per-doc cosine median 0.94**, query cosine 0.998. Bug specifically affects documents, not queries. |
| `verify_reasonir_doc_prefix.py` | **Minimal isolation.** Same docs, same model, only difference is `prompt=""` vs `prompt="<\|embed\|>\n"`: 9985 / 10000 docs have cosine < 0.99 between the two encodings. nDCG@10 differs by 0.028. Definitive smoking gun. |

### Empirical impact of the fix

ReasonIR-8B × BrightProBiologyRetrieval, full 60K corpus:

| pipeline | nDCG@10 |
|---|---:|
| BRIGHT-Pro paper saved | 0.33322 |
| BRIGHT-Pro `AutoModel + custom .encode` (our env) | 0.33755 |
| **MTEB pipeline (current upstream, buggy)** | **0.35323** |
| **MTEB pipeline (with `get_task_instruction` fix)** | **0.34269** |

The fix narrows the MTEB-vs-paper gap from +0.020 nDCG to +0.009 nDCG. The residual ~0.009 is implementation-level noise (sentence-transformers' tokenizer/padding vs HuggingFace's custom `.encode`, bf16 numerical precision).

A separate MTEB issue + PR for this bug will reference this analysis.

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
  diagnostics/
    verify_reasonir_*.json  # gap-investigation artifacts
```

## Reproducing

```bash
# one (model, task) cell
MODEL_NAME=Qwen/Qwen3-Embedding-8B \
TASK_NAME=BrightProBiologyRetrieval \
MAX_SEQ_LENGTH=4096 BATCH_SIZE=16 \
python scripts/run_mteb_eval.py
```
