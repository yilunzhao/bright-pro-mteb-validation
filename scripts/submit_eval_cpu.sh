#!/usr/bin/env bash
#SBATCH --job-name=mteb-bp-eval-bm25
#SBATCH --output=/home/yz979/project/yilun/mteb-fork-runs/logs/%x__%j.out
#SBATCH --error=/home/yz979/project/yilun/mteb-fork-runs/logs/%x__%j.err
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --requeue

set -euo pipefail

: "${MODEL_NAME:?must set MODEL_NAME}"
: "${TASK_NAME:?must set TASK_NAME}"

export OUT_ROOT=/home/yz979/project/yilun/mteb-fork-runs/results
export MTEB_CACHE_DIR=/home/yz979/project/yilun/mteb-fork-runs/cache
export HF_HOME=/nfs/roberts/scratch/pi_ac3458/yz979/yilun_hf_cache
export HF_DATASETS_CACHE=/home/yz979/project/yilun/cache/huggingface/datasets
mkdir -p "$OUT_ROOT" "$MTEB_CACHE_DIR" "$HF_DATASETS_CACHE"

PY=/home/yz979/project_pi_ac3458/yz979/enviroment/envs/brightpro-eval/bin/python
SCRIPT=/home/yz979/project/yilun/mteb-fork-runs/scripts/run_mteb_eval.py

echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK"
echo "MODEL=$MODEL_NAME TASK=$TASK_NAME"

"$PY" "$SCRIPT"
