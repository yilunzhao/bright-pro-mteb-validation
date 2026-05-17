"""Test whether MTEB's corpus `.strip()` preprocessing accounts for the gap
between BRIGHT-Pro's saved nDCG (0.333) and MTEB's pipeline nDCG (0.353/0.364).

Runs ReasonIR's own .encode() (BRIGHT-Pro path) on full biology, with and
without `.strip()` applied to docs. If the stripped version jumps to ~0.36,
strip is the dominant source of the gap.
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


def encode_and_score(model, queries, docs, doc_ids, query_ids, gold, q_instr, d_instr, bs=8, max_len=4096, tag=""):
    print(f"\n=== [{tag}] encoding ===")
    print(f"  queries (n={len(queries)})")
    q_emb = np.asarray(model.encode(queries, instruction=q_instr, batch_size=bs, max_length=max_len))
    print(f"  docs (n={len(docs)})")
    d_emb = np.asarray(model.encode(docs, instruction=d_instr, batch_size=bs, max_length=max_len))
    q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12
    d_emb /= np.linalg.norm(d_emb, axis=1, keepdims=True) + 1e-12
    sim = q_emb @ d_emb.T
    res = make_top_k(sim, doc_ids, query_ids, k=1000)
    ndcg = calculate_retrieval_scores(res, gold, k_values=[1, 5, 10]).ndcg["NDCG@10"]
    print(f"  [{tag}] nDCG@10 = {ndcg:.5f}")
    return float(ndcg)


def main():
    domain = "biology"
    cache_dir = os.environ.get("HF_DATASETS_CACHE")
    docs_ds = load_dataset("yale-nlp/Bright-Pro", "documents", split=domain, cache_dir=cache_dir)
    ex_ds = load_dataset("yale-nlp/Bright-Pro", "examples", split=domain, cache_dir=cache_dir)

    doc_ids = [d["id"] for d in docs_ds]
    raw_docs = [d["content"] for d in docs_ds]
    stripped_docs = [d.strip() for d in raw_docs]
    n_changed = sum(1 for r, s in zip(raw_docs, stripped_docs) if r != s)
    print(f"#docs with strip()-changed text: {n_changed}/{len(raw_docs)}")
    query_ids = [str(e["id"]) for e in ex_ds]
    queries = [e["query"] for e in ex_ds]
    gold = {str(e["id"]): {gid: 1 for gid in e["gold_ids"]} for e in ex_ds}

    q_instr = "<|user|>\nGiven a biology post, retrieve relevant passages that help answer the post\n<|embed|>\n"
    d_instr = "<|embed|>\n"

    print("\nLoading ReasonIR via AutoModel + custom .encode (Path A)")
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        "reasonir/ReasonIR-8B", torch_dtype="auto", trust_remote_code=True,
    )
    model.eval().cuda()

    ndcg_raw = encode_and_score(
        model, queries, raw_docs, doc_ids, query_ids, gold, q_instr, d_instr,
        bs=8, max_len=4096, tag="Path A, RAW docs",
    )
    ndcg_strip = encode_and_score(
        model, queries, stripped_docs, doc_ids, query_ids, gold, q_instr, d_instr,
        bs=8, max_len=4096, tag="Path A, STRIPPED docs",
    )

    out = {
        "ndcg_PathA_raw_docs": ndcg_raw,
        "ndcg_PathA_stripped_docs": ndcg_strip,
        "delta_strip": ndcg_strip - ndcg_raw,
        "ndcg_PathA_full_eval_saved": 0.33755,
        "ndcg_MTEB_pipeline_default": 0.35323,
        "ndcg_MTEB_pipeline_include_prompt_false": 0.36438,
        "bright_pro_paper": 0.33322,
        "n_docs_strip_changed": int(n_changed),
        "n_queries": len(queries),
        "n_docs": len(docs_ds),
        "batch_size": 8,
        "max_length": 4096,
    }
    print("\n=== Summary ===")
    for k, v in out.items():
        print(f"  {k}: {v}")
    (OUT_DIR / "verify_reasonir_strip.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
