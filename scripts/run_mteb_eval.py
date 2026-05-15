"""Run a single (model, BrightPro<Domain>Retrieval) evaluation via MTEB.

Reads MODEL_NAME and TASK_NAME from env. Loads the fork's editable mteb via
sys.path so it picks up the BrightPro* task classes and the updated prompts.
Writes nDCG@10 (and the full retrieval scores) to results/<task>_<model>.json
so the SLURM dispatcher can skip already-completed cells.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

FORK = "/home/yz979/project/yilun/mteb-fork"
sys.path.insert(0, FORK)

import mteb  # noqa: E402


def main() -> None:
    model_name = os.environ["MODEL_NAME"]
    task_name = os.environ["TASK_NAME"]
    batch_size = int(os.environ.get("BATCH_SIZE", "16"))
    max_seq_length = os.environ.get("MAX_SEQ_LENGTH")  # optional
    out_root = Path(os.environ["OUT_ROOT"])
    cache_root = Path(os.environ["MTEB_CACHE_DIR"])

    model_slug = model_name.replace("/", "__")
    out_path = out_root / f"{task_name}__{model_slug}.json"
    if out_path.exists():
        print(f"SKIP: {out_path} already exists")
        return

    print(f"==> model={model_name} task={task_name} batch={batch_size}")
    t0 = time.time()

    model_meta = mteb.get_model_meta(model_name)
    task = mteb.get_task(task_name)

    # The HF cache on this cluster was populated under different revisions than
    # what MTEB's ModelMeta pins. Re-point the meta to the locally-cached
    # revision so HF_HUB_OFFLINE=1 can resolve files. Without this, even non-
    # gated models would re-download; for gated ones (embeddinggemma) the
    # revision check would 403.
    hf_hub = Path(os.environ.get("HF_HOME", "")) / "hub"
    slug = "models--" + model_name.replace("/", "--")
    refs_main = hf_hub / slug / "refs" / "main"
    if refs_main.exists():
        cached_rev = refs_main.read_text().strip()
        if cached_rev and cached_rev != model_meta.revision:
            print(
                f"NOTE: re-pointing {model_name} revision "
                f"{model_meta.revision} -> {cached_rev} (cached locally)"
            )
            model_meta = model_meta.model_copy(update={"revision": cached_rev})

    # Load the model ourselves so we can pin max_seq_length to BRIGHT-Pro's value.
    # InstructSentenceTransformerModel accepts max_seq_length in its constructor;
    # SentenceTransformerEncoderWrapper doesn't, so for that path we post-set the
    # underlying ST model's max_seq_length after load.
    load_kwargs = {}
    if max_seq_length:
        load_kwargs["max_seq_length"] = int(max_seq_length)
    # MTEB's gte-Qwen2-7B-instruct loader defaults to fp16, which is known to
    # produce numerically-unstable embeddings for this model — empirically
    # nDCG@10 collapses (~0.07 vs BRIGHT-Pro's 0.51). The model card itself
    # recommends bf16. Patch the meta's loader_kwargs directly (passing dtype
    # through load_model() goes via experiment_kwargs which only accepts
    # JSON-serializable values).
    if model_name == "Alibaba-NLP/gte-Qwen2-7B-instruct":
        import torch as _torch
        # Two issues with MTEB's stock loader for this model:
        # 1) `model_kwargs={"dtype": torch.float16}` — fp16 is numerically
        #    unstable for gte-Qwen2 (model card recommends bf16).
        # 2) `trust_remote_code` is NOT set — but the gte-Qwen2 repo ships
        #    `modeling_qwen.py` with the bidirectional attention impl that
        #    the embedding head depends on. Without trust_remote_code=True,
        #    sentence-transformers loads stock causal Qwen2 and embeddings
        #    collapse to noise (~0.07 nDCG@10).
        model_meta.loader_kwargs = {
            **model_meta.loader_kwargs,
            "model_kwargs": {"dtype": _torch.bfloat16},
            "trust_remote_code": True,
        }
    try:
        model = model_meta.load_model(**load_kwargs)
    except TypeError as e:
        if "max_seq_length" not in str(e):
            raise
        print(f"loader rejected max_seq_length, falling back to post-set: {e}")
        model = model_meta.load_model()
        if max_seq_length and hasattr(model, "model") and hasattr(model.model, "max_seq_length"):
            model.model.max_seq_length = int(max_seq_length)

    enc_kwargs = {"batch_size": batch_size}

    cache = mteb.ResultCache(cache_root)
    result = mteb.evaluate(
        model,
        task,
        encode_kwargs=enc_kwargs,
        cache=cache,
        overwrite_strategy="only-missing",
    )

    elapsed = time.time() - t0

    # result is ModelResult; we want the per-task scores.
    task_results = result.task_results[0] if result.task_results else None
    scores = task_results.scores if task_results else {}

    # MTEB returns scores keyed by split; we have one split "standard".
    split_score = scores.get("standard")
    if isinstance(split_score, list) and split_score:
        split_score = split_score[0]

    payload = {
        "model": model_name,
        "task": task_name,
        "batch_size": batch_size,
        "max_seq_length": int(max_seq_length) if max_seq_length else None,
        "elapsed_sec": round(elapsed, 1),
        "ndcg_at_10": split_score.get("ndcg_at_10") if split_score else None,
        "ndcg_at_5": split_score.get("ndcg_at_5") if split_score else None,
        "ndcg_at_1": split_score.get("ndcg_at_1") if split_score else None,
        "recall_at_10": split_score.get("recall_at_10") if split_score else None,
        "main_score": split_score.get("main_score") if split_score else None,
        "all_scores": split_score,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"DONE in {elapsed:.0f}s, nDCG@10 = {payload['ndcg_at_10']}, wrote {out_path}")


if __name__ == "__main__":
    main()
