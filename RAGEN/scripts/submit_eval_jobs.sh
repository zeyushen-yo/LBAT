#!/bin/bash

# Edit the MODELS array below; one sbatch will be submitted per model.
# The slurm script reads the model path from the MODELS env var.

set -euo pipefail

SLURM_SCRIPT="/home/zs7353/LBAT/RAGEN/slurm_scripts/eval.slurm"

MODELS=(
  "/scratch/gpfs/zs7353/Llama-3.1-8B-Instruct"
  "/scratch/gpfs/zs7353/Qwen2.5-7B-Instruct"
  "/scratch/gpfs/zs7353/ragen/lbat-grpo_baseline_llama/global_step_200/actor/merged_hf"
  "/scratch/gpfs/zs7353/ragen/lbat-grpo_baseline_qwen/global_step_200/actor/merged_hf"
  "/scratch/gpfs/zs7353/ragen/lbat-grpo_llama/global_step_200/actor/merged_hf"
  "/scratch/gpfs/zs7353/ragen/lbat-grpo_qwen/global_step_200/actor/merged_hf"
)

for model in "${MODELS[@]}"; do
  echo "Submitting eval job for: ${model}"
  MODELS="${model}" sbatch "${SLURM_SCRIPT}"
done


