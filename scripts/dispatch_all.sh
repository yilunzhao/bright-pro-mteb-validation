#!/usr/bin/env bash
#
# Submit one SLURM job per (model, BRIGHT-Pro domain) cell.
# Skips cells whose result JSON already exists.
# Stays on a single partition (gpu_rtx6000) per CLAUDE.md guidance.

set -euo pipefail

ROOT=/home/yz979/project/yilun/mteb-fork-runs
RES=$ROOT/results

DOMAINS=(
    "BrightProBiologyRetrieval"
    "BrightProEarthScienceRetrieval"
    "BrightProEconomicsRetrieval"
    "BrightProPsychologyRetrieval"
    "BrightProRoboticsRetrieval"
    "BrightProStackoverflowRetrieval"
    "BrightProSustainableLivingRetrieval"
)

# Pair model name with the (max_seq_length, batch_size) tuple BRIGHT-PRO used.
# Format: model|max_seq_length|batch_size
MODELS=(
    "google/embeddinggemma-300m|2048|32"
    "Alibaba-NLP/gte-Qwen2-7B-instruct|8192|8"
    "Qwen/Qwen3-Embedding-8B|4096|8"
    "ReasonIR/ReasonIR-8B|4096|8"
    "GritLM/GritLM-7B|2048|8"
)

submitted=0
skipped=0
for entry in "${MODELS[@]}"; do
    model=${entry%%|*}
    rest=${entry#*|}
    max_len=${rest%%|*}
    bs=${rest##*|}
    model_slug=${model//\//__}
    for task in "${DOMAINS[@]}"; do
        out="$RES/${task}__${model_slug}.json"
        if [ -f "$out" ]; then
            skipped=$((skipped+1))
            continue
        fi
        sbatch \
          --job-name="mteb-${model_slug}-${task}" \
          --export=ALL,MODEL_NAME=$model,TASK_NAME=$task,MAX_SEQ_LENGTH=$max_len,BATCH_SIZE=$bs,HF_TOKEN_OVERRIDE=${HF_TOKEN_OVERRIDE:-} \
          $ROOT/scripts/submit_eval.sh
        submitted=$((submitted+1))
    done
done

echo "submitted=$submitted, skipped(existing)=$skipped"
