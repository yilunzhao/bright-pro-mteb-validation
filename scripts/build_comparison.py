"""Build the MTEB-vs-BRIGHT-PRO comparison table from completed eval jobs.

Reads each mteb-fork-runs/results/<task>__<model_slug>.json (written by
run_mteb_eval.py) and pairs it with the matching BRIGHT-PRO results.json
under BRIGHT-PRO/retrieval/outputs/<domain>_<bp_model>/results.json.
Emits a Markdown table for posting on the PR.
"""
from __future__ import annotations

import json
from pathlib import Path

RES_ROOT = Path("/home/yz979/project/yilun/mteb-fork-runs/results")
BP_ROOT = Path(
    "/nfs/roberts/project/pi_ac3458/yz979/yilun/bright-pro/BRIGHT-PRO/retrieval/outputs"
)

DOMAINS = [
    ("biology", "BrightProBiologyRetrieval"),
    ("earth_science", "BrightProEarthScienceRetrieval"),
    ("economics", "BrightProEconomicsRetrieval"),
    ("psychology", "BrightProPsychologyRetrieval"),
    ("robotics", "BrightProRoboticsRetrieval"),
    ("stackoverflow", "BrightProStackoverflowRetrieval"),
    ("sustainable_living", "BrightProSustainableLivingRetrieval"),
]

# (display_name, mteb_name, bright_pro_model_id)
MODELS = [
    ("BM25", "mteb/baseline-bm25s", "bm25"),
    ("embeddinggemma-300m", "google/embeddinggemma-300m", "embeddinggemma"),
    ("Qwen3-Embedding-8B", "Qwen/Qwen3-Embedding-8B", "qwen3-embed"),
    ("ReasonIR-8B", "ReasonIR/ReasonIR-8B", "reasonir"),
]


def read_mteb(task_name: str, mteb_model: str) -> float | None:
    slug = mteb_model.replace("/", "__")
    p = RES_ROOT / f"{task_name}__{slug}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return data.get("ndcg_at_10")


def read_bp(domain: str, bp_model: str) -> float | None:
    p = BP_ROOT / f"{domain}_{bp_model}" / "results.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return data.get("NDCG@10")


def main() -> None:
    rows: list[list[str]] = []
    diffs: list[float] = []
    for domain, task_name in DOMAINS:
        for display, mteb_model, bp_model in MODELS:
            mteb_ndcg = read_mteb(task_name, mteb_model)
            bp_ndcg = read_bp(domain, bp_model)
            if mteb_ndcg is None or bp_ndcg is None:
                row = [
                    domain, display,
                    f"{bp_ndcg}" if bp_ndcg is not None else "—",
                    f"{mteb_ndcg}" if mteb_ndcg is not None else "—",
                    "—",
                ]
            else:
                diff = mteb_ndcg - bp_ndcg
                diffs.append(diff)
                row = [
                    domain, display,
                    f"{bp_ndcg:.5f}", f"{mteb_ndcg:.5f}",
                    f"{diff:+.5f}",
                ]
            rows.append(row)

    # Print as markdown table
    headers = ["domain", "model", "BRIGHT-PRO nDCG@10", "MTEB nDCG@10", "Δ"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        print("| " + " | ".join(r) + " |")
    if diffs:
        print()
        max_abs = max(abs(d) for d in diffs)
        mean_abs = sum(abs(d) for d in diffs) / len(diffs)
        print(f"# cells with both numbers: {len(diffs)}")
        print(f"max |Δ| = {max_abs:.5f}")
        print(f"mean |Δ| = {mean_abs:.5f}")


if __name__ == "__main__":
    main()
