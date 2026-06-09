#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=.

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
RESULTS_DIR="/nfs/stak/users/chenhaoj/hpc-share/NLP/FP/NLP-Final-Project-MEMIT/memit/results"

timestamp="$(date +%Y%m%d_%H%M%S)"
RUN_GROUP="qwen-memit-zsre-layer-ablation-${timestamp}"
RUN_GROUP_DIR="$RESULTS_DIR/$RUN_GROUP"
MASTER_LOG="$RUN_GROUP_DIR/zsre_layer_ablation_${timestamp}.log"

mkdir -p "$RUN_GROUP_DIR"

latest_run_dir() {
  ls -td "$RUN_GROUP_DIR"/run_* 2>/dev/null | head -n 1
}

run_ablation_stage() {
  local stage_name="$1"
  local hparams="$2"
  local dataset_size=100
  local stage_log="$RUN_GROUP_DIR/${stage_name}_evaluate.log"

  echo ""
  echo "============================================================"
  echo "Starting layer ablation: $stage_name"
  echo "Hparams: $hparams"
  echo "Dataset: zsRE"
  echo "Dataset size: $dataset_size"
  echo "Run group: $RUN_GROUP"
  echo "Evaluate log: $stage_log"
  echo "Time: $(date)"
  echo "============================================================"

  python -m experiments.evaluate \
    --alg_name MEMIT \
    --model_name "$MODEL_NAME" \
    --hparams_fname "$hparams" \
    --ds_name zsre \
    --dataset_size_limit "$dataset_size" \
    --num_edits 1 \
    --skip_generation_tests \
    --use_cache \
    --dir_name "$RUN_GROUP" > "$stage_log" 2>&1

  echo ""
  echo "Evaluate finished for stage: $stage_name"
  echo "Last 20 lines from evaluate log:"
  tail -n 20 "$stage_log"

  local run_dir
  run_dir="$(latest_run_dir)"
  local run_name
  run_name="$(basename "$run_dir")"
  local summary_file="$run_dir/summary_${stage_name}.txt"

  echo ""
  echo "Summarizing $run_name"
  echo "Summary file: $summary_file"
  echo ""

  {
    echo "Stage: $stage_name"
    echo "Dataset: zsRE"
    echo "Model: $MODEL_NAME"
    echo "Hparams: $hparams"
    echo "Dataset size: $dataset_size"
    echo "Metric style: original repo top-1 token accuracy"
    echo "Run group: $RUN_GROUP"
    echo "Run dir: $run_dir"
    echo "Evaluate log: $stage_log"
    echo "Summary time: $(date)"
    echo ""

    python -m experiments.summarize \
      --dir_name "$RUN_GROUP" \
      --runs "$run_name" \
      --first_n_cases "$dataset_size"
  } | tee "$summary_file"

  echo ""
  echo "Summary written to: $summary_file"
}

{
  echo "Qwen MEMIT zsRE layer ablation"
  echo "Started: $(date)"
  echo "Working directory: $(pwd)"
  echo "Run group: $RUN_GROUP"
  echo "Run group dir: $RUN_GROUP_DIR"
  echo ""

  run_ablation_stage "mid_layers_12_16" "deepseek-r1-distill-qwen-1.5b-mid.json"
  run_ablation_stage "late_layers_20_24" "deepseek-r1-distill-qwen-1.5b-late.json"
  run_ablation_stage "single_layer_27" "deepseek-r1-distill-qwen-1.5b-layer27.json"

  echo ""
  echo "All layer ablation stages finished: $(date)"
} 2>&1 | tee "$MASTER_LOG"

echo ""
echo "Master log written to: $MASTER_LOG"
echo "Run group dir: $RUN_GROUP_DIR"