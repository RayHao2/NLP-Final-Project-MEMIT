#!/bin/bash
#SBATCH -A eecs
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --job-name=qwen-edit-gen
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
mkdir -p logs



export HF_HOME="${HF_HOME:-/nfs/guille/eecs_research/soundbendor/sentiment/hf_cache}"

export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_DEEP_GEMM=0

source /nfs/guille/eecs_research/soundbendor/sentiment/miniconda3/bin/activate
source activate memit
nvidia-smi
python generate_edits.py --dataset zsre_eval