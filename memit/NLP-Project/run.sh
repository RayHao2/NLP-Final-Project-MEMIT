#!/bin/bash
#SBATCH -J qwen-memit-10k
#SBATCH -A ai539
#SBATCH -p dgx2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o sbatch_output/qwen-memit-10k.%j.out
#SBATCH -e sbatch_output/qwen-memit-10k.%j.err

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}"

cd /nfs/stak/users/chenhaoj/hpc-share/NLP/FP/NLP-Final-Project-MEMIT/memit

source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate /nfs/stak/users/chenhaoj/hpc-share/memit-qwen-env

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=.

export PROJECT_CACHE=/nfs/stak/users/chenhaoj/hpc-share/cache
export HF_HOME=$PROJECT_CACHE/hf-home
export HF_HUB_CACHE=$PROJECT_CACHE/hf-hub
export TRANSFORMERS_CACHE=$PROJECT_CACHE/hf-transformers
export HF_DATASETS_CACHE=$PROJECT_CACHE/hf-datasets
export TORCH_HOME=$PROJECT_CACHE/torch
export TORCH_EXTENSIONS_DIR=$PROJECT_CACHE/torch/extensions
export NLTK_DATA=$PROJECT_CACHE/nltk
export MPLCONFIGDIR=$PROJECT_CACHE/matplotlib
export TMPDIR=$PROJECT_CACHE/tmp
export TEMP=$PROJECT_CACHE/tmp
export TMP=$PROJECT_CACHE/tmp


echo "Test run" 
# mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
# mkdir -p "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$NLTK_DATA" "$MPLCONFIGDIR" "$TMPDIR"

# RUN_GROUP="qwen-memit-20260527_135034"

# python -m experiments.evaluate \
#   --alg_name MEMIT \
#   --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
#   --hparams_fname deepseek-r1-distill-qwen-1.5b.json \
#   --ds_name cf \
#   --dataset_size_limit 10000 \
#   --num_edits 1 \
#   --skip_generation_tests \
#   --use_cache \
#   --dir_name "$RUN_GROUP"

# LATEST_RUN="$(ls -td results/$RUN_GROUP/run_* | head -n 1)"
# LATEST_RUN_NAME="$(basename "$LATEST_RUN")"

# python -m experiments.summarize \
#   --dir_name "$RUN_GROUP" \
#   --runs "$LATEST_RUN_NAME" \
#   --first_n_cases 10000 \
#   2>&1 | tee "$LATEST_RUN/summary_stage7_10000_fast.txt"

# echo "Latest run: $LATEST_RUN"
# echo "Job finished at: $(date)"