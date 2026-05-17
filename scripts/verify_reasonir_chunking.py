"""Isolate the contribution of MTEB's 50K corpus chunking vs single-call
encoding, with everything else matching MTEB's pipeline (default config,
stripped docs).

Mode 1: encode all 60K docs in a single st.encode call (our prior Path B
        diagnostic measured 0.345 here without strip)
Mode 2: encode 0-49999 then 50000-59512 separately, concatenate embeddings
        (simulates MTEB SearchEncoderWrapper.corpus_chunk_size=50_000)

Both modes use stripped docs (matching MTEB's _corpus_to_dict). If Mode 2
matches MTEB's pipeline 0.353, chunking is the dominant remaining factor.
If not, the remaining gap comes from MTEB's DataLoader/collate or per-chunk
similarity+heap merging logic.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

FORK = "/home/yz979/project/yilun/mteb-fork"
sys.path.insert(0, FORK)

import numpy as np
import torch
from datasets import load_dataset

from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores


OUT_DIR = Path("/home/yz979/project/yilun/mteb-fork-runs/results")


def make_top_k(sim, doc_ids, query_ids, k=1000):
    out = {}
    for qi, qid in enumerate(query_ids):
        scores = sim[qi]
        top_idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        out[qid] = {doc_ids[di]: float(scores[di]) for di in top_idx}
    return out


def score(q_emb, d_emb, query_ids, doc_ids, gold):
    qn = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    dn = d_emb / (np.linalg.norm(d_emb, axis=1, keepdims=True) + 1e-12)
    sim = qn @ dn.T
    res = make_top_k(sim, doc_ids, query_ids, k=1000)
    return float(calculate_retrieval_scores(res, gold, k_values=[1, 5, 10]).ndcg["NDCG@10"])


def main():
    domain = "biology"
    cache_dir = os.environ.get("HF_DATASETS_CACHE")
    docs_ds = load_dataset("yale-nlp/Bright-Pro", "documents", split=domain, cache_dir=cache_dir)
    ex_ds = load_dataset("yale-nlp/Bright-Pro", "examples", split=domain, cache_dir=cache_dir)

    doc_ids = [d["id"] for d in docs_ds]
    docs = [d["content"].strip() for d in docs_ds]  # MTEB-style stripped
    query_ids = [str(e["id"]) for e in ex_ds]
    queries = [e["query"] for e in ex_ds]
    gold = {str(e["id"]): {gid: 1 for gid in e["gold_ids"]} for e in ex_ds}
    print(f"#queries={len(queries)} #docs={len(docs)} (stripped)")

    q_instr = "<|user|>\nGiven a biology post, retrieve relevant passages that help answer the post\n<|embed|>\n"
    d_instr = "<|embed|>\n"

    # Path B = MTEB ST wrapper, default include_prompt=True
    import mteb
    mm = mteb.get_model_meta("ReasonIR/ReasonIR-8B")
    hf_hub = Path(os.environ["HF_HOME"]) / "hub"
    refs_main = hf_hub / "models--ReasonIR--ReasonIR-8B" / "refs" / "main"
    if refs_main.exists():
        cached_rev = refs_main.read_text().strip()
        if cached_rev and cached_rev != mm.revision:
            mm = mm.model_copy(update={"revision": cached_rev})

    print("\nLoading via mteb.get_model_meta(...).load_model() (default, no include_prompt override)...")
    model_obj = mm.load_model(max_seq_length=4096)
    st = model_obj.model

    bs = 8

    # Queries: encode once (same across modes)
    print(f"\nencoding queries (n={len(queries)}, batch={bs})")
    qB = st.encode(queries, prompt=q_instr, batch_size=bs, convert_to_numpy=True, show_progress_bar=True)

    # ----- Mode 1: single st.encode for all 60K docs -----
    print(f"\nMode 1: encoding ALL {len(docs)} docs in a single st.encode call (batch={bs})")
    dB_single = st.encode(docs, prompt=d_instr, batch_size=bs, convert_to_numpy=True, show_progress_bar=True)
    np.save(OUT_DIR / "verify_chunking_dB_single.npy", dB_single)

    # ----- Mode 2: chunked 50K + remainder -----
    chunk_size = 50_000
    print(f"\nMode 2: encoding in chunks of {chunk_size}")
    parts = []
    for start in range(0, len(docs), chunk_size):
        end = min(start + chunk_size, len(docs))
        print(f"  chunk [{start}:{end}]  size={end-start}")
        emb = st.encode(docs[start:end], prompt=d_instr, batch_size=bs, convert_to_numpy=True, show_progress_bar=True)
        parts.append(emb)
    dB_chunked = np.concatenate(parts, axis=0)
    np.save(OUT_DIR / "verify_chunking_dB_chunked.npy", dB_chunked)

    print(f"\nshapes: single={dB_single.shape}  chunked={dB_chunked.shape}")

    # ----- Cosine between single vs chunked embeddings (same docs) -----
    dn_single = dB_single / (np.linalg.norm(dB_single, axis=1, keepdims=True) + 1e-12)
    dn_chunked = dB_chunked / (np.linalg.norm(dB_chunked, axis=1, keepdims=True) + 1e-12)
    cos_per_doc = (dn_single * dn_chunked).sum(axis=1)
    pct = {q: float(np.percentile(cos_per_doc, q)) for q in (0, 1, 50, 99, 100)}
    print(f"\nper-doc cosine(single, chunked): {pct}")

    # ----- nDCG via each path -----
    ndcg_single = score(qB, dB_single, query_ids, doc_ids, gold)
    ndcg_chunked = score(qB, dB_chunked, query_ids, doc_ids, gold)
    print(f"\n=== nDCG@10 ===")
    print(f"  Mode 1 (single 60K)    : {ndcg_single:.5f}")
    print(f"  Mode 2 (50K + 9513)    : {ndcg_chunked:.5f}")
    print(f"  Δ(chunked - single)    : {ndcg_chunked - ndcg_single:+.5f}")
    print(f"  reference: MTEB pipeline default = 0.35323, Path A = 0.33755")

    out = {
        "ndcg_mode1_single_60K": ndcg_single,
        "ndcg_mode2_chunked_50K": ndcg_chunked,
        "delta_chunked_minus_single": ndcg_chunked - ndcg_single,
        "ndcg_mteb_pipeline_default": 0.35323,
        "ndcg_pathA_raw": 0.33755,
        "ndcg_pathA_strip": 0.33966,
        "n_docs": len(docs),
        "chunk_size": chunk_size,
        "batch_size": bs,
        "doc_cosine_single_vs_chunked": pct,
        "docs_stripped": True,
    }
    (OUT_DIR / "verify_reasonir_chunking.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
