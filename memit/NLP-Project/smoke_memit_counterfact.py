"""
Small MEMIT smoke test for CounterFact.

This script is intentionally narrower than experiments/evaluate.py. It loads a
small CounterFact slice, records generations before/after MEMIT, and writes a
compact JSON artifact for manual inspection.
"""

import argparse
import json
from pathlib import Path
from time import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import CounterFactDataset
from memit import MEMITHyperParams, apply_memit_to_model
from util.globals import DATA_DIR, HPARAMS_DIR, KV_DIR, RESULTS_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="EleutherAI/gpt-j-6B")
    parser.add_argument("--hparams_fname", default="EleutherAI_gpt-j-6B.json")
    parser.add_argument("--num_edits", type=int, default=1)
    parser.add_argument("--case_offset", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=12)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_model_and_tokenizer(model_name, cache_dir):
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model.eval()
    return model, tok


def greedy_generate(model, tok, prompts, max_new_tokens):
    inputs = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    max_length = inputs["input_ids"].shape[1] + max_new_tokens
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_length=max_length,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return tok.batch_decode(output_ids, skip_special_tokens=True)


def make_record_summary(record):
    rewrite = record["requested_rewrite"]
    subject = rewrite["subject"]
    return {
        "case_id": record["case_id"],
        "prompt": rewrite["prompt"].format(subject),
        "subject": subject,
        "target_true": rewrite.get("target_true", {}).get("str"),
        "target_new": rewrite["target_new"]["str"],
    }


def main():
    args = parse_args()
    assert args.num_edits > 0, "--num_edits must be positive"

    print(f"Loading CounterFact from {DATA_DIR}")
    ds = CounterFactDataset(DATA_DIR, size=args.case_offset + args.num_edits)
    records = [ds[i] for i in range(args.case_offset, args.case_offset + args.num_edits)]

    print(f"Loading model {args.model_name}")
    model, tok = load_model_and_tokenizer(args.model_name, args.cache_dir)

    hparams_path = HPARAMS_DIR / "MEMIT" / args.hparams_fname
    hparams = MEMITHyperParams.from_json(hparams_path)
    print(f"Loaded MEMIT hparams from {hparams_path}")

    prompts = [make_record_summary(record)["prompt"] for record in records]
    before_outputs = greedy_generate(model, tok, prompts, args.max_new_tokens)

    requests = [
        {"case_id": record["case_id"], **record["requested_rewrite"]}
        for record in records
    ]
    cache_template = None
    if args.use_cache:
        cache_template = ( KV_DIR
            / f"{args.model_name.replace('/', '_')}_MEMIT"
            / f"cf_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        print(f"Using MEMIT z-vector cache template: {cache_template}")

    print(f"Applying MEMIT with {args.num_edits} edit(s)")
    start = time()
    edited_model, _ = apply_memit_to_model(
        model,
        tok,
        requests,
        hparams,
        copy=False,
        return_orig_weights=False,
        cache_template=cache_template,
    )
    edit_runtime_sec = time() - start
    print(f"MEMIT runtime: {edit_runtime_sec:.2f}s")

    after_outputs = greedy_generate(edited_model, tok, prompts, args.max_new_tokens)

    results = {
        "model_name": args.model_name,
        "hparams_fname": args.hparams_fname,
        "num_edits": args.num_edits,
        "case_offset": args.case_offset,
        "edit_runtime_sec": edit_runtime_sec,
        "records": [],
    }
    for record, before, after in zip(records, before_outputs, after_outputs):
        summary = make_record_summary(record)
        summary["output_before"] = before
        summary["output_after"] = after
        results["records"].append(summary)

    if args.output is None:
        output_path = (
            RESULTS_DIR
            / "smoke"
            / f"memit_counterfact_{args.num_edits}_edits_offset_{args.case_offset}.json"
        )
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {output_path}")
    print(json.dumps(results["records"], indent=2))


if __name__ == "__main__":
    main()