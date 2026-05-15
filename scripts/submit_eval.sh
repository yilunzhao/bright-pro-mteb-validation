#!/usr/bin/env bash
#SBATCH --job-name=mteb-bp-eval
#SBATCH --output=/home/yz979/project/yilun/mteb-fork-runs/logs/%x__%j.out
#SBATCH --error=/home/yz979/project/yilun/mteb-fork-runs/logs/%x__%j.err
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_ac3458
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --requeue

set -euo pipefail

# Required env vars from caller:
#   MODEL_NAME      (e.g. google/embeddinggemma-300m)
#   TASK_NAME       (e.g. BrightProBiologyRetrieval)
#   BATCH_SIZE      (optional, default 16)
#   MAX_SEQ_LENGTH  (optional)

: "${MODEL_NAME:?must set MODEL_NAME}"
: "${TASK_NAME:?must set TASK_NAME}"

export OUT_ROOT=/home/yz979/project/yilun/mteb-fork-runs/results
export MTEB_CACHE_DIR=/home/yz979/project/yilun/mteb-fork-runs/cache
# Reuse the user's existing HF cache so we don't re-download Qwen3-Embedding-8B etc.
export HF_HOME=/nfs/roberts/scratch/pi_ac3458/yz979/yilun_hf_cache
export HF_DATASETS_CACHE=/home/yz979/project/yilun/cache/huggingface/datasets
# Token with gated-repo access (embeddinggemma). Pass via env, not committed.
export HF_TOKEN="${HF_TOKEN_OVERRIDE:-$(cat "$HF_HOME/token" 2>/dev/null)}"
# Keep hub access ON so datasets.load_dataset("yale-nlp/Bright-Pro", ...) can
# fetch. The run script overrides each model's pinned revision to whatever is
# already in $HF_HOME/hub so model weights aren't re-downloaded.
mkdir -p "$OUT_ROOT" "$MTEB_CACHE_DIR" "$HF_DATASETS_CACHE"

# Avoid DDP port collisions with other concurrent jobs (per CLAUDE.md).
export MASTER_PORT=$((29500 + RANDOM % 1000))

PY=/home/yz979/project_pi_ac3458/yz979/enviroment/envs/brightpro-eval/bin/python
SCRIPT=/home/yz979/project/yilun/mteb-fork-runs/scripts/run_mteb_eval.py

echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "MODEL=$MODEL_NAME TASK=$TASK_NAME BATCH=${BATCH_SIZE:-16}"

"$PY" "$SCRIPT"
