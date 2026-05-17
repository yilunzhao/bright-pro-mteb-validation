"""Minimal isolation: encode identical docs through st.encode with two
different prompts — empty vs "<|embed|>\\n" — see if the embeddings diverge
(confirming the root cause is MTEB's get_task_instruction returning ""
instead of routing empty instructions through instruction_template).
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

    n_docs = int(os.environ.get("N_DOCS", "5000"))
    n_q = int(os.environ.get("N_QUERIES", "30"))
    doc_ids = [d["id"] for d in docs_ds][:n_docs]
    docs = [d["content"].strip() for d in docs_ds][:n_docs]
    ex_list = list(ex_ds)[:n_q]
    query_ids = [str(e["id"]) for e in ex_list]
    queries = [e["query"] for e in ex_list]
    doc_id_set = set(doc_ids)
    gold = {str(e["id"]): {gid: 1 for gid in e["gold_ids"] if gid in doc_id_set} for e in ex_list}
    print(f"#queries={len(queries)} #docs={len(docs)}")

    q_instr = "<|user|>\nGiven a biology post, retrieve relevant passages that help answer the post\n<|embed|>\n"
    d_instr_full = "<|embed|>\n"

    import mteb
    mm = mteb.get_model_meta("ReasonIR/ReasonIR-8B")
    hf_hub = Path(os.environ["HF_HOME"]) / "hub"
    refs = hf_hub / "models--ReasonIR--ReasonIR-8B" / "refs" / "main"
    if refs.exists():
        cached = refs.read_text().strip()
        if cached and cached != mm.revision:
            mm = mm.model_copy(update={"revision": cached})
    model = mm.load_model(max_seq_length=4096)
    st = model.model

    bs = int(os.environ.get("BATCH_SIZE", "8"))

    print("\nencoding queries (same prompt for both variants)")
    qB = np.asarray(st.encode(queries, prompt=q_instr, batch_size=bs, convert_to_numpy=True, show_progress_bar=True))

    print(f"\nencoding docs with EMPTY prompt (mimics MTEB pipeline)")
    dB_empty = np.asarray(st.encode(docs, prompt="", batch_size=bs, convert_to_numpy=True, show_progress_bar=True))

    print(f"\nencoding docs with FULL prompt '<|embed|>\\n' (BRIGHT-Pro / direct path)")
    dB_full = np.asarray(st.encode(docs, prompt=d_instr_full, batch_size=bs, convert_to_numpy=True, show_progress_bar=True))

    # Cosine between the two encodings
    def norm(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    e_n, f_n = norm(dB_empty), norm(dB_full)
    cos = (e_n * f_n).sum(axis=1)
    pct = {f"p{q}": float(np.percentile(cos, q)) for q in (0, 1, 50, 99, 100)}
    n_low = int((cos < 0.99).sum())

    print(f"\nper-doc cosine(empty_prompt, embed_prompt): {pct}, #cos<0.99: {n_low}/{len(docs)}")

    ndcg_empty = score(qB, dB_empty, query_ids, doc_ids, gold)
    ndcg_full = score(qB, dB_full, query_ids, doc_ids, gold)
    print(f"\n=== nDCG@10 ===")
    print(f"  docs with prompt='' (MTEB)        : {ndcg_empty:.5f}")
    print(f"  docs with prompt='<|embed|>\\n'   : {ndcg_full:.5f}")
    print(f"  delta                              : {ndcg_full - ndcg_empty:+.5f}")

    out = {
        "ndcg_docs_empty_prompt": ndcg_empty,
        "ndcg_docs_embed_prompt": ndcg_full,
        "delta": ndcg_full - ndcg_empty,
        "cos_empty_vs_embed_prompt": pct,
        "n_docs_cos_below_0_99": n_low,
        "n_queries": len(queries),
        "n_docs": len(docs),
    }
    (OUT_DIR / "verify_reasonir_doc_prefix.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
