"""Side-by-side: MTEB's InstructSentenceTransformerModel.encode(dataloader)
vs direct SentenceTransformer.encode(list). Same model, same docs, same
stripped corpus, same prompt. If embeddings differ, we've isolated the
MTEB-pipeline-specific layer that produces the +0.015 nDCG@10 gap.
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
from datasets import Dataset, load_dataset

from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores
from mteb._create_dataloaders import create_dataloader
from mteb.types import PromptType


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
    import mteb
    task = mteb.get_task("BrightProBiologyRetrieval")
    task.load_data()
    docs_dict = task.corpus["standard"]  # {doc_id: {"text": ...}}
    queries_dict = task.queries["standard"]  # {qid: query_text}
    gold = {qid: dict(d) for qid, d in task.relevant_docs["standard"].items()}

    # Subset-mode (env-controlled) for fast iteration. Set N_DOCS=0 for full corpus.
    n_docs_target = int(os.environ.get("N_DOCS", "5000"))
    n_q_target = int(os.environ.get("N_QUERIES", "30"))
    all_doc_ids = list(docs_dict.keys())
    all_query_ids = list(queries_dict.keys())
    doc_ids = all_doc_ids if n_docs_target == 0 else all_doc_ids[:n_docs_target]
    query_ids = all_query_ids if n_q_target == 0 else all_query_ids[:n_q_target]
    print(f"#queries={len(query_ids)} #docs={len(doc_ids)}  (subset of {len(all_query_ids)}/{len(all_doc_ids)})")
    # restrict dicts to subset
    docs_dict = {k: docs_dict[k] for k in doc_ids}
    queries_dict = {k: queries_dict[k] for k in query_ids}
    gold = {qid: {d: r for d, r in gold[qid].items() if d in docs_dict} for qid in query_ids if qid in gold}

    # ----------- Load model exactly as MTEB pipeline would -----------
    mm = mteb.get_model_meta("ReasonIR/ReasonIR-8B")
    hf_hub = Path(os.environ["HF_HOME"]) / "hub"
    refs_main = hf_hub / "models--ReasonIR--ReasonIR-8B" / "refs" / "main"
    if refs_main.exists():
        cached_rev = refs_main.read_text().strip()
        if cached_rev and cached_rev != mm.revision:
            mm = mm.model_copy(update={"revision": cached_rev})
    model = mm.load_model(max_seq_length=4096)
    print(f"loaded: {type(model).__name__}")
    print(f"  apply_instruction_to_passages={model.apply_instruction_to_passages}")
    pool = model.model._first_module() if hasattr(model.model, "_first_module") else None
    # find pooling module
    pooling_mod = None
    for m in model.model:
        if "Pooling" in type(m).__name__:
            pooling_mod = m
            break
    if pooling_mod is not None:
        print(f"  pooling: type={type(pooling_mod).__name__}")
        for attr in ["pooling_mode_mean_tokens","pooling_mode_lasttoken","pooling_mode_cls_token","include_prompt"]:
            print(f"    {attr}={getattr(pooling_mod, attr, 'n/a')}")

    bs = int(os.environ.get("BATCH_SIZE", "4"))

    # ----------- A) Build MTEB-style DataLoaders & run wrapper.encode -----------
    print("\n=== Path A: model.encode via DataLoaders (mimicking MTEB pipeline) ===")
    # Build a HF Dataset for corpus
    corpus_rows = [{"id": did, "text": docs_dict[did]["text"]} for did in doc_ids]
    corpus_hf = Dataset.from_list(corpus_rows)
    print(f"  corpus_hf cols={corpus_hf.column_names}")
    queries_rows = [{"id": qid, "text": queries_dict[qid]} for qid in query_ids]
    queries_hf = Dataset.from_list(queries_rows)

    q_dl = create_dataloader(queries_hf, task_metadata=task.metadata, prompt_type=PromptType.query, batch_size=bs)
    d_dl = create_dataloader(corpus_hf, task_metadata=task.metadata, prompt_type=PromptType.document, batch_size=bs)
    print("  built dataloaders")

    qA = model.encode(q_dl, task_metadata=task.metadata, hf_split="standard", hf_subset="default",
                      prompt_type=PromptType.query, batch_size=bs)
    dA = model.encode(d_dl, task_metadata=task.metadata, hf_split="standard", hf_subset="default",
                      prompt_type=PromptType.document, batch_size=bs)
    qA = np.asarray(qA); dA = np.asarray(dA)
    print(f"  qA={qA.shape} dA={dA.shape}")
    np.save(OUT_DIR / "verify_wrapperdirect_qA.npy", qA)
    np.save(OUT_DIR / "verify_wrapperdirect_dA.npy", dA)
    ndcgA = score(qA, dA, query_ids, doc_ids, gold)
    print(f"  Path A nDCG@10 = {ndcgA:.5f}")

    # ----------- B) Direct SentenceTransformer.encode -----------
    print("\n=== Path B: direct st.encode(list) ===")
    st = model.model
    # Replicate what MTEB constructs for the prompt:
    from mteb.models.model_implementations.reasonir_model import instruction_template
    q_instr = instruction_template(task.metadata.prompt["query"], PromptType.query)
    d_instr = instruction_template("", PromptType.document)
    print(f"  q_instr = {repr(q_instr)}")
    print(f"  d_instr = {repr(d_instr)}")

    # text input for queries: raw query strings
    queries_text = [queries_dict[qid] for qid in query_ids]
    # text input for docs: docs_dict[did]["text"] AS IS — MTEB's _corpus_to_dict
    # would strip these; but our task class already extracted them above. Since
    # docs_dict has raw "content" not stripped, MTEB will strip after; replicate
    # that here for parity with Path A's dataloader path:
    docs_text_stripped = [docs_dict[did]["text"].strip() for did in doc_ids]
    docs_text_raw = [docs_dict[did]["text"] for did in doc_ids]

    qB = np.asarray(st.encode(queries_text, prompt=q_instr, batch_size=bs, convert_to_numpy=True, show_progress_bar=True))
    dB_stripped = np.asarray(st.encode(docs_text_stripped, prompt=d_instr, batch_size=bs, convert_to_numpy=True, show_progress_bar=True))
    np.save(OUT_DIR / "verify_wrapperdirect_qB.npy", qB)
    np.save(OUT_DIR / "verify_wrapperdirect_dB_stripped.npy", dB_stripped)
    ndcgB_strip = score(qB, dB_stripped, query_ids, doc_ids, gold)
    print(f"  Path B (stripped docs) nDCG@10 = {ndcgB_strip:.5f}")

    # ----------- Cosine comparison -----------
    def norm(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    qAn = norm(qA); qBn = norm(qB)
    dAn = norm(dA); dBn = norm(dB_stripped)
    q_cos = (qAn * qBn).sum(axis=1)
    d_cos = (dAn * dBn).sum(axis=1)
    print(f"\nper-vector cosine(A_wrapper, B_direct):")
    print(f"  queries: min={q_cos.min():.6f} median={np.median(q_cos):.6f} max={q_cos.max():.6f}")
    print(f"  docs   : min={d_cos.min():.6f} median={np.median(d_cos):.6f} max={d_cos.max():.6f}")
    n_low = int((d_cos < 0.99).sum())
    print(f"  #docs with cos < 0.99: {n_low}")

    out = {
        "ndcg_wrapper_dataloader": ndcgA,
        "ndcg_direct_st_encode": ndcgB_strip,
        "delta_wrapper_minus_direct": ndcgA - ndcgB_strip,
        "ndcg_mteb_pipeline_default_saved": 0.35323,
        "ndcg_pathA_BRIGHTPRO_style_saved": 0.33755,
        "q_cosine_AB": {
            "min": float(q_cos.min()), "p1": float(np.percentile(q_cos, 1)),
            "median": float(np.median(q_cos)),
            "p99": float(np.percentile(q_cos, 99)), "max": float(q_cos.max()),
        },
        "d_cosine_AB": {
            "min": float(d_cos.min()), "p1": float(np.percentile(d_cos, 1)),
            "median": float(np.median(d_cos)),
            "p99": float(np.percentile(d_cos, 99)), "max": float(d_cos.max()),
            "n_below_0_99": n_low,
        },
        "q_instr": q_instr,
        "d_instr": d_instr,
        "n_queries": len(query_ids),
        "n_docs": len(doc_ids),
        "batch_size": bs,
    }
    (OUT_DIR / "verify_wrapper_vs_direct.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
