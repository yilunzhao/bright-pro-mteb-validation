"""Compare results_2p18 (MTEB 2.18.1 re-validation) against BRIGHT-PRO's own
harness numbers (retrieval/outputs/<domain>_<bp_model>/results.json).

Primary comparison: THIS RUN vs BRIGHT-PRO. The May MTEB column is shown as
secondary context only.
"""

import json
from pathlib import Path

NEW = Path("/home/yz979/project/yilun/mteb-fork-runs/results_2p18")
MAY = Path("/home/yz979/project/yilun/bright-pro-mteb-validation-staging/results")
BP_ROOT = Path(
    "/nfs/roberts/project/pi_ac3458/yz979/yilun/bright-pro/BRIGHT-PRO/retrieval/outputs"
)

# mteb slug -> BRIGHT-PRO outputs model dir name
MODEL_TO_BP = {
    "mteb__baseline-bm25s": "bm25",
    "google__embeddinggemma-300m": "embeddinggemma",
    "Qwen__Qwen3-Embedding-8B": "qwen3-embed",
    "ReasonIR__ReasonIR-8B": "reasonir",
    "Alibaba-NLP__gte-Qwen2-7B-instruct": "gte-qwen2",
}
TASK_TO_DOMAIN = {
    "BrightProBiologyRetrieval": "biology",
    "BrightProEarthScienceRetrieval": "earth_science",
    "BrightProEconomicsRetrieval": "economics",
    "BrightProPsychologyRetrieval": "psychology",
    "BrightProRoboticsRetrieval": "robotics",
    "BrightProStackoverflowRetrieval": "stackoverflow",
    "BrightProSustainableLivingRetrieval": "sustainable_living",
}


def main() -> None:
    rows = []
    for f in sorted(NEW.glob("*.json")):
        task, slug = f.stem.split("__", 1)
        domain = TASK_TO_DOMAIN.get(task)
        bp_model = MODEL_TO_BP.get(slug)
        new = json.loads(f.read_text())["ndcg_at_10"]

        bp = None
        bp_file = BP_ROOT / f"{domain}_{bp_model}" / "results.json"
        if bp_file.exists():
            bp = json.loads(bp_file.read_text())["NDCG@10"]

        may_file = MAY / f.name
        may = (
            json.loads(may_file.read_text())["ndcg_at_10"]
            if may_file.exists()
            else None
        )
        rows.append((domain, slug, new, bp, may))

    hdr = f"{'domain':<20} {'model':<36} {'this-run':>8} {'BRIGHT-PRO':>10} {'Δ':>9} {'(may)':>8}"
    print(hdr)
    print("-" * len(hdr))
    dense_deltas = []
    for domain, slug, new, bp, may in rows:
        d = (new - bp) if bp is not None else None
        if d is not None and "bm25" not in slug:
            dense_deltas.append(abs(d))
        print(
            f"{domain:<20} {slug:<36} {new:8.5f} "
            f"{(f'{bp:.5f}' if bp is not None else '—'):>10} "
            f"{(f'{d:+.5f}' if d is not None else '—'):>9} "
            f"{(f'{may:.5f}' if may is not None else '—'):>8}"
        )
    n_total = 5 * 7  # 5 models x 7 domains
    print(f"\ncells: {len(rows)}/{n_total}")
    if dense_deltas:
        print(
            f"dense models vs BRIGHT-PRO: mean |Δ| = {sum(dense_deltas)/len(dense_deltas):.5f}, "
            f"max |Δ| = {max(dense_deltas):.5f}"
        )
    print("(bm25 excluded from |Δ| stats: mteb bm25s != BRIGHT-PRO BM25 implementation)")


if __name__ == "__main__":
    main()
