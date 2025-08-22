#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   ./submit_all_tasks.sh [--model PATH_TO_MODEL]
#
# Submits one job per task group (0..9), exporting TASK_NUM for each job.

MODEL_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_ARG="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

SLURM_SCRIPT="/home/zs7353/generalizable_exploration/paprika/scripts/slurm_scripts/run_evaluation.slurm"

for TASK_IDX in {0..9}; do
  if [[ -n "$MODEL_ARG" ]]; then
    AGENT_MODEL_NAME="$MODEL_ARG" TASK_NUM="$TASK_IDX" sbatch "$SLURM_SCRIPT" | cat
  else
    TASK_NUM="$TASK_IDX" sbatch "$SLURM_SCRIPT" | cat
  fi
done


