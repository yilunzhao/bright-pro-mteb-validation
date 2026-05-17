"""Full-set characterisation of the ReasonIR-8B gap.

Runs the SAME 103 queries + 59513 docs of BrightProBiologyRetrieval through
two inference paths, saves both embedding matrices, and reports:

  - per-vector cosine(Path A, Path B) distribution (min, p1, median, p99, max)
  - nDCG@10 for each path using the same pytrec_eval evaluator
  - L2-norm distributions

This isolates whether the 0.016 nDCG gap between BRIGHT-Pro-style inference
and MTEB's ST wrapper comes from the embedding step (and if so, how much
the two embeddings differ over the full corpus).
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
OUT_DIR.mkdir(parents=True, exist_ok=True)


def percentiles(a, qs=(0, 1, 50, 99, 100)):
    return {f"p{q}": float(np.percentile(a, q)) for q in qs}


def make_top_k(sim, doc_ids, query_ids, k=1000):
    results = {}
    for qi, qid in enumerate(query_ids):
        scores = sim[qi]
        top_idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        results[qid] = {doc_ids[di]: float(scores[di]) for di in top_idx}
    return results


def main():
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
    bs = int(os.environ.get("BATCH_SIZE", "8"))
    max_len = 4096

    # ---------- Path A: BRIGHT-Pro style (AutoModel + custom .encode) ----------
    print("\n=== Path A: AutoModel + model.encode ===")
    from transformers import AutoModel
    modelA = AutoModel.from_pretrained(
        "reasonir/ReasonIR-8B", torch_dtype="auto", trust_remote_code=True,
    )
    modelA.eval().cuda()
    print(f"encoding queries with batch={bs}, max_length={max_len}")
    qA = np.asarray(modelA.encode(queries, instruction=q_instr, batch_size=bs, max_length=max_len))
    print(f"encoding {len(docs)} docs with batch={bs}, max_length={max_len}")
    dA = np.asarray(modelA.encode(docs, instruction=d_instr, batch_size=bs, max_length=max_len))
    print(f"qA shape={qA.shape} dA shape={dA.shape}")
    print(f"qA norms p={percentiles(np.linalg.norm(qA, axis=1))}")
    print(f"dA norms p={percentiles(np.linalg.norm(dA, axis=1))}")
    np.save(OUT_DIR / "verify_reasonir_full_paths_qA.npy", qA)
    np.save(OUT_DIR / "verify_reasonir_full_paths_dA.npy", dA)
    del modelA
    torch.cuda.empty_cache()

    # ---------- Path B: MTEB ST wrapper ----------
    print("\n=== Path B: mteb InstructSentenceTransformerModel ===")
    import mteb
    mm = mteb.get_model_meta("ReasonIR/ReasonIR-8B")
    hf_hub = Path(os.environ["HF_HOME"]) / "hub"
    slug = "models--ReasonIR--ReasonIR-8B"
    refs_main = hf_hub / slug / "refs" / "main"
    if refs_main.exists():
        cached_rev = refs_main.read_text().strip()
        if cached_rev and cached_rev != mm.revision:
            print(f"re-point revision -> {cached_rev}")
            mm = mm.model_copy(update={"revision": cached_rev})
    modelB = mm.load_model(max_seq_length=max_len)
    st = modelB.model
    print(f"encoding queries with batch={bs}")
    qB = st.encode(queries, prompt=q_instr, batch_size=bs, convert_to_numpy=True, show_progress_bar=True)
    print(f"encoding {len(docs)} docs with batch={bs}")
    dB = st.encode(docs, prompt=d_instr, batch_size=bs, convert_to_numpy=True, show_progress_bar=True)
    print(f"qB shape={qB.shape} dB shape={dB.shape}")
    print(f"qB norms p={percentiles(np.linalg.norm(qB, axis=1))}")
    print(f"dB norms p={percentiles(np.linalg.norm(dB, axis=1))}")
    np.save(OUT_DIR / "verify_reasonir_full_paths_qB.npy", qB)
    np.save(OUT_DIR / "verify_reasonir_full_paths_dB.npy", dB)
    del modelB
    torch.cuda.empty_cache()

    # ---------- Cosine on full set ----------
    def normalize(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    qAn, dAn = normalize(qA), normalize(dA)
    qBn, dBn = normalize(qB), normalize(dB)
    q_cos = (qAn * qBn).sum(axis=1)
    d_cos = (dAn * dBn).sum(axis=1)
    print(f"\nper-vector cosine(A,B):")
    print(f"  queries (n={len(q_cos)}): {percentiles(q_cos)}")
    print(f"  docs    (n={len(d_cos)}): {percentiles(d_cos)}")

    # ---------- nDCG via same evaluator ----------
    print("\ncomputing nDCG@10 for both paths via mteb pytrec_eval...")
    simA = qAn @ dAn.T
    simB = qBn @ dBn.T
    resA = make_top_k(simA, doc_ids, query_ids, k=1000)
    resB = make_top_k(simB, doc_ids, query_ids, k=1000)
    ndcgA = calculate_retrieval_scores(resA, gold, k_values=[1, 5, 10]).ndcg["NDCG@10"]
    ndcgB = calculate_retrieval_scores(resB, gold, k_values=[1, 5, 10]).ndcg["NDCG@10"]
    print(f"Path A nDCG@10 = {ndcgA:.5f}")
    print(f"Path B nDCG@10 = {ndcgB:.5f}")
    print(f"BRIGHT-Pro saved = 0.33322")
    print(f"MTEB eval saved  = 0.35323")

    out = {
        "ndcg_A_full": float(ndcgA),
        "ndcg_B_full": float(ndcgB),
        "bright_pro_paper": 0.33322,
        "mteb_eval_saved": 0.35323,
        "n_queries": len(queries),
        "n_docs": len(docs),
        "q_cosine_AB": percentiles(q_cos),
        "d_cosine_AB": percentiles(d_cos),
        "qA_norms": percentiles(np.linalg.norm(qA, axis=1)),
        "qB_norms": percentiles(np.linalg.norm(qB, axis=1)),
        "dA_norms": percentiles(np.linalg.norm(dA, axis=1)),
        "dB_norms": percentiles(np.linalg.norm(dB, axis=1)),
        "batch_size": bs,
        "max_length": max_len,
    }
    (OUT_DIR / "verify_reasonir_full_paths.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_DIR / 'verify_reasonir_full_paths.json'}")


if __name__ == "__main__":
    main()
