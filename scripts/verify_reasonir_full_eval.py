"""Full biology evaluation via the BRIGHT-Pro path (AutoModel + model.encode),
to identify whether the MTEB-vs-BRIGHT-Pro gap (0.353 vs 0.333) is from the
inference path or from elsewhere.

If this script produces:
  ~0.333 -> MTEB's ST wrapper genuinely produces different rankings on full eval
            (despite cos≈1 on a small subset — small embedding shifts amplify
            over a 60K-doc ranking)
  ~0.353 -> BRIGHT-Pro's saved 0.33322 is stale / older config; MTEB number is
            the correct "current" reproduction.
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


def main() -> None:
    domain = "biology"
    cache_dir = os.environ.get("HF_DATASETS_CACHE")
    docs_ds = load_dataset("yale-nlp/Bright-Pro", "documents", split=domain, cache_dir=cache_dir)
    ex_ds = load_dataset("yale-nlp/Bright-Pro", "examples", split=domain, cache_dir=cache_dir)

    doc_ids = [d["id"] for d in docs_ds]
    docs = [d["content"] for d in docs_ds]
    query_ids = [str(e["id"]) for e in ex_ds]
    queries = [e["query"] for e in ex_ds]
    gold = {str(e["id"]): {gid: 1 for gid in e["gold_ids"]} for e in ex_ds}
    print(f"#queries={len(queries)} #docs={len(docs)} #qrels={len(gold)}")

    q_instr = "<|user|>\nGiven a biology post, retrieve relevant passages that help answer the post\n<|embed|>\n"
    d_instr = "<|embed|>\n"

    print("\nLoading ReasonIR-8B via BRIGHT-Pro path (AutoModel + .encode)...")
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        "reasonir/ReasonIR-8B",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.eval().cuda()

    bs = int(os.environ.get("BATCH_SIZE", "8"))
    print(f"\nEncoding queries with batch_size={bs}, max_length=4096...")
    q_emb = model.encode(queries, instruction=q_instr, batch_size=bs, max_length=4096)
    q_emb = np.asarray(q_emb)
    print(f"q_emb shape={q_emb.shape}, norms mean={np.linalg.norm(q_emb, axis=1).mean():.4f}")

    print(f"\nEncoding {len(docs)} docs with batch_size={bs}, max_length=4096...")
    d_emb = model.encode(docs, instruction=d_instr, batch_size=bs, max_length=4096)
    d_emb = np.asarray(d_emb)
    print(f"d_emb shape={d_emb.shape}, norms mean={np.linalg.norm(d_emb, axis=1).mean():.4f}")

    # Cosine similarity (normalize)
    qn = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    dn = d_emb / (np.linalg.norm(d_emb, axis=1, keepdims=True) + 1e-12)
    print("\nComputing similarity matrix...")
    sim = qn @ dn.T  # shape (Q, D)

    results = {}
    for qi, qid in enumerate(query_ids):
        scores = sim[qi]
        # top-1000
        top_idx = np.argpartition(-scores, min(1000, len(scores)-1))[:1000]
        results[qid] = {doc_ids[di]: float(scores[di]) for di in top_idx}

    print("\nComputing nDCG@10 via MTEB's pytrec_eval pipeline...")
    res = calculate_retrieval_scores(results, gold, k_values=[1,5,10])
    ndcg_at_10 = res.ndcg["NDCG@10"]
    print(f"\n=== nDCG@10 via Path A (BRIGHT-Pro style, batch={bs}): {ndcg_at_10:.5f} ===")
    print("  BRIGHT-Pro saved: 0.33322")
    print("  MTEB-via-ST    : 0.35323")

    out = {
        "ndcg_at_10_path_A": ndcg_at_10,
        "ndcg_at_10_bright_pro_saved": 0.33322,
        "ndcg_at_10_mteb": 0.35323,
        "batch_size": bs,
        "n_queries": len(queries),
        "n_docs": len(docs),
    }
    Path("/home/yz979/project/yilun/mteb-fork-runs/results/verify_reasonir_full_eval.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
