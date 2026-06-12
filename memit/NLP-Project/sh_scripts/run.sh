#!/bin/bash
#SBATCH -J qwen-zsre-fixed-10k
#SBATCH -A ai539
#SBATCH -p dgx2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o sbatch_output/qwen-zsre-fixed-10k.%j.out
#SBATCH -e sbatch_output/qwen-zsre-fixed-10k.%j.err

set -euo pipefail

echo "Job started: $(date)"
echo "Node: $(hostname)"
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

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
HPARAMS="deepseek-r1-distill-qwen-1.5b.json"
DATASET_SIZE=10000
RUN_GROUP="qwen-memit-zsre-padding-fixed-10k"

echo "Model: $MODEL_NAME"
echo "Hparams: $HPARAMS"
echo "Dataset: zsRE"
echo "Dataset size: $DATASET_SIZE"
echo "Run group: $RUN_GROUP"

python -m experiments.evaluate \
  --alg_name MEMIT \
  --model_name "$MODEL_NAME" \
  --hparams_fname "$HPARAMS" \
  --ds_name zsre \
  --dataset_size_limit "$DATASET_SIZE" \
  --num_edits 1 \
  --skip_generation_tests \
  --use_cache \
  --dir_name "$RUN_GROUP"

LATEST_RUN="$(ls -td results/"$RUN_GROUP"/run_* | head -n 1)"
LATEST_RUN_NAME="$(basename "$LATEST_RUN")"

NUM_RESULTS="$(find "$LATEST_RUN" -name '1_edits-case_*.json' | wc -l)"

echo "Completed result files: $NUM_RESULTS"

{
  echo "Qwen MEMIT Corrected zsRE Evaluation"
  echo "Model: $MODEL_NAME"
  echo "Hparams: $HPARAMS"
  echo "Dataset size: $DATASET_SIZE"
  echo "Completed cases: $NUM_RESULTS"
  echo "Run directory: $LATEST_RUN"
  echo "Summary time: $(date)"
  echo ""

  python -m experiments.summarize \
    --dir_name "$RUN_GROUP" \
    --runs "$LATEST_RUN_NAME" \
    --first_n_cases "$DATASET_SIZE"
} 2>&1 | tee "$LATEST_RUN/summary_zsre_fixed_10000.txt"

echo "Job finished: $(date)"
echo "Results: $LATEST_RUN"