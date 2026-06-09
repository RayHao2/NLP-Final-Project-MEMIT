import argparse
import json
from pathlib import Path
from time import time

import numpy as np
import torch
from scipy.stats import hmean
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import CounterFactDataset
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
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
    parser.add_argument("--dataset_size_limit", type=int, default=1000)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def mean_pct(values):
    return float(np.mean(values) * 100) if values else 0.0


def summarize_counterfact(records):
    rewrite_success = []
    rewrite_diff = []
    paraphrase_success = []
    paraphrase_diff = []
    neighborhood_success = []
    neighborhood_diff = []

    for record in records:
        post = record["post"]

        for x in post.get("rewrite_prompts_probs", []):
            rewrite_success.append(x["target_true"] > x["target_new"])
            rewrite_diff.append(np.exp(-x["target_new"]) - np.exp(-x["target_true"]))

        for x in post.get("paraphrase_prompts_probs", []):
            paraphrase_success.append(x["target_true"] > x["target_new"])
            paraphrase_diff.append(np.exp(-x["target_new"]) - np.exp(-x["target_true"]))

        for x in post.get("neighborhood_prompts_probs", []):
            neighborhood_success.append(x["target_true"] < x["target_new"])
            neighborhood_diff.append(np.exp(-x["target_true"]) - np.exp(-x["target_new"]))

    es = mean_pct(rewrite_success)
    ps = mean_pct(paraphrase_success)
    ns = mean_pct(neighborhood_success)

    score = float(hmean([es, ps, ns])) if min(es, ps, ns) > 0 else 0.0

    return {
        "rewrite_success": es,
        "paraphrase_success": ps,
        "neighborhood_success": ns,
        "score": score,
        "rewrite_diff": float(np.mean(rewrite_diff) * 100) if rewrite_diff else 0.0,
        "paraphrase_diff": float(np.mean(paraphrase_diff) * 100) if paraphrase_diff else 0.0,
        "neighborhood_diff": float(np.mean(neighborhood_diff) * 100) if neighborhood_diff else 0.0,
        "num_rewrite_prompts": len(rewrite_success),
        "num_paraphrase_prompts": len(paraphrase_success),
        "num_neighborhood_prompts": len(neighborhood_success),
    }


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

    print(f"Loading CounterFact from {DATA_DIR}")
    ds = CounterFactDataset(DATA_DIR, tok=tok, size=args.dataset_size_limit)
    print(f"Loaded {len(ds)} CounterFact records")

    records = []
    start = time()

    for i, record in enumerate(ds):
        if i % 100 == 0:
            print(f"Evaluating baseline case {i}/{len(ds)}")

        metrics = compute_rewrite_quality_counterfact(
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
    summary = summarize_counterfact(records)

    summary = {
        "model_name": args.model_name,
        "dataset": "counterfact",
        "dataset_size_limit": args.dataset_size_limit,
        "metric_style": "original repo probability-comparison metrics, unedited baseline",
        "runtime_sec": runtime,
        **summary,
    }

    result = {
        "summary": summary,
        "records": records,
    }

    if args.output is None:
        output_path = (
            RESULTS_DIR
            / "qwen-counterfact-baseline"
            / f"baseline_counterfact_{args.dataset_size_limit}.json"
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