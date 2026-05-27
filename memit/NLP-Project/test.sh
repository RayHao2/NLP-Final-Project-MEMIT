for i in 0 1 2 3 4 5 6 7 8 9; do
  echo "==== case_offset $i ===="
  rm -f kvs/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_MEMIT/cf_layer_8_clamp_0.75_case_${i}.npz
  TOKENIZERS_PARALLELISM=false PYTHONPATH=. python NLP-Project/smoke_memit_counterfact.py \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --hparams_fname deepseek-r1-distill-qwen-1.5b.json \
    --num_edits 1 \
    --case_offset $i \
    --cache_dir /nfs/stak/users/chenhaoj/hpc-share/cache/hf-transformers \
    --max_new_tokens 16 \
    --use_cache
done