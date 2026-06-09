#!/bin/bash
#SBATCH -J qwen-cf-wsweep
#SBATCH -A ai539
#SBATCH -p dgx2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o sbatch_output/qwen-cf-wsweep.%j.out
#SBATCH -e sbatch_output/qwen-cf-wsweep.%j.err

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

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
mkdir -p "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$NLTK_DATA" "$MPLCONFIGDIR" "$TMPDIR"
mkdir -p sbatch_output

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_SIZE=500
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_GROUP="qwen-cf-weight-sweep-${TIMESTAMP}"

HPARAMS_LIST=(
  "qwen-early-w25000-n1000.json"
  "qwen-early-w35000-n1000.json"
  "qwen-early-w45000-n1000.json"
  "qwen-early-w60000-n1000.json"
)

echo "Run group: $RUN_GROUP"

for HPARAMS in "${HPARAMS_LIST[@]}"; do
  CONFIG_NAME="${HPARAMS%.json}"
  CONFIG_GROUP="${RUN_GROUP}/${CONFIG_NAME}"

  echo ""
  echo "============================================================"
  echo "Starting config: $CONFIG_NAME"
  echo "Hparams: $HPARAMS"
  echo "Dataset size: $DATASET_SIZE"
  echo "Time: $(date)"
  echo "============================================================"

  python -m experiments.evaluate \
    --alg_name MEMIT \
    --model_name "$MODEL_NAME" \
    --hparams_fname "$HPARAMS" \
    --ds_name cf \
    --dataset_size_limit "$DATASET_SIZE" \
    --num_edits 1 \
    --skip_generation_tests \
    --use_cache \
    --dir_name "$CONFIG_GROUP"

  LATEST_RUN="$(ls -td results/$CONFIG_GROUP/run_* | head -n 1)"
  LATEST_RUN_NAME="$(basename "$LATEST_RUN")"

  echo ""
  echo "Summarizing $CONFIG_NAME"
  echo "Run dir: $LATEST_RUN"

  {
    echo "Config: $CONFIG_NAME"
    echo "Model: $MODEL_NAME"
    echo "Dataset: CounterFact"
    echo "Dataset size: $DATASET_SIZE"
    echo "Hparams: $HPARAMS"
    echo "Run group: $CONFIG_GROUP"
    echo "Run dir: $LATEST_RUN"
    echo "Summary time: $(date)"
    echo ""

    python -m experiments.summarize \
      --dir_name "$CONFIG_GROUP" \
      --runs "$LATEST_RUN_NAME" \
      --first_n_cases "$DATASET_SIZE"
  } 2>&1 | tee "$LATEST_RUN/summary.txt"
done

echo ""
echo "All configs finished at: $(date)"
echo "Top-level run group: results/$RUN_GROUP"