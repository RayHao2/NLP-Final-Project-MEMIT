import argparse
import json
from pathlib import Path
from time import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import MENDQADataset
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from util.globals import DATA_DIR, RESULTS_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    )
    parser.add_argument(
        "--cache_dir",
        default="/nfs/stak/users/chenhaoj/hpc-share/cache/hf-transformers",
    )
    parser.add_argument("--dataset_size_limit", type=int, default=100)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def mean_nested_bool(values):
    if len(values) == 0:
        return 0.0
    return float(np.mean(values) * 100)


def main():
    args = parse_args()

    print(f"Loading tokenizer: {args.model_name}")
    tok = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    tok.pad_token = tok.eos_token

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    ).eval()

    print(f"Loading zsRE from {DATA_DIR}")
    ds = MENDQADataset(DATA_DIR, tok=tok, size=args.dataset_size_limit)
    print(f"Loaded {len(ds)} zsRE records")

    records = []
    start = time()

    for i, record in enumerate(ds):
        if i % 25 == 0:
            print(f"Evaluating baseline case {i}/{len(ds)}")

        metrics = compute_rewrite_quality_zsre(
            model,
            tok,
            record,
            None,
            None,
        )

        records.append(
            {
                "case_id": record["case_id"],
                "requested_rewrite": record["requested_rewrite"],
                "post": metrics,
            }
        )

    runtime = time() - start

    rewrite_vals = [
        x
        for record in records
        for x in record["post"]["rewrite_prompts_correct"]
    ]
    paraphrase_vals = [
        x
        for record in records
        for x in record["post"]["paraphrase_prompts_correct"]
    ]
    neighborhood_vals = [
        x
        for record in records
        for x in record["post"]["neighborhood_prompts_correct"]
    ]

    summary = {
        "model_name": args.model_name,
        "dataset": "zsre",
        "dataset_size_limit": args.dataset_size_limit,
        "metric_style": "original repo top-1 token accuracy, unedited baseline",
        "runtime_sec": runtime,
        "rewrite_acc": mean_nested_bool(rewrite_vals),
        "paraphrase_acc": mean_nested_bool(paraphrase_vals),
        "neighborhood_acc": mean_nested_bool(neighborhood_vals),
        "num_rewrite_tokens": len(rewrite_vals),
        "num_paraphrase_tokens": len(paraphrase_vals),
        "num_neighborhood_tokens": len(neighborhood_vals),
    }

    result = {
        "summary": summary,
        "records": records,
    }

    if args.output is None:
        output_path = (
            RESULTS_DIR
            / "qwen-zsre-baseline"
            / f"baseline_zsre_{args.dataset_size_limit}.json"
        )
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()