#!/bin/bash
#SBATCH --job-name=paprika_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G  
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00  
#SBATCH --constraint="gpu80"
#SBATCH --mail-type=begin       
#SBATCH --mail-type=end         
#SBATCH --mail-user=zs7353@princeton.edu
module purge
module load proxy/default anaconda3/2024.2 gcc-toolset/13 cudatoolkit/12.6
conda activate paprika

export AI_SANDBOX_KEY="ae696dffa9874e17bbd7f13499d3b571"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -z "${TASK_NUM:-}" ]; then
    export TASK_NUM="${SLURM_ARRAY_TASK_ID}"
fi

# Set the model to evaluate here so it propagates to the bash script (but allow external override)
# if AGENT_MODEL_NAME is not set, fall back to a default; otherwise respect the provided value
if [ -z "${AGENT_MODEL_NAME:-}" ]; then
    # export AGENT_MODEL_NAME="/scratch/gpfs/zs7353/ragen/underspecified-bandit-grpo_kl/global_step_100/actor/merged_hf"
    export AGENT_MODEL_NAME="/scratch/gpfs/zs7353/Llama-3.1-8B-Instruct"
    # export AGENT_MODEL_NAME="/scratch/gpfs/zs7353/ragen/underspecified-bandit-grpo_baseline_kl/global_step_200/actor/merged_hf"
fi

cd /home/zs7353/generalizable_exploration/paprika/scripts/bash_scripts
echo "Starting array task ${SLURM_ARRAY_TASK_ID} (TASK_NUM=${TASK_NUM}) with AGENT_MODEL_NAME=${AGENT_MODEL_NAME}"
bash run_evaluation.sh