import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from time import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import MENDQADataset
from memit import MEMITHyperParams, apply_memit_to_model
from util import nethook
from util.globals import DATA_DIR, HPARAMS_DIR, KV_DIR, RESULTS_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    )
    parser.add_argument(
        "--hparams_fname",
        default="deepseek-r1-distill-qwen-1.5b.json",
    )
    parser.add_argument("--dataset_size_limit", type=int, default=100)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--generation_examples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def score_target(model, tok, prompt, target):
    target = target if target.startswith(" ") else " " + target

    prompt_ids = tok(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]

    full = tok(
        prompt + target,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        logits = model(**full).logits[0]

    full_ids = full["input_ids"][0]
    target_start = len(prompt_ids)
    target_ids = full_ids[target_start:]

    token_results = []

    for position, target_id in enumerate(target_ids):
        logit_position = target_start + position - 1
        token_logits = logits[logit_position].float()
        probabilities = torch.softmax(token_logits, dim=-1)

        target_logit = token_logits[target_id]
        target_probability = probabilities[target_id].item()
        target_rank = int((token_logits > target_logit).sum().item()) + 1

        top_probability, top_id = probabilities.max(dim=-1)

        token_results.append(
            {
                "position": position,
                "is_first_token": position == 0,
                "target_token": tok.decode([target_id.item()]),
                "target_token_id": target_id.item(),
                "target_probability": target_probability,
                "target_rank": target_rank,
                "target_is_top1": target_rank == 1,
                "top_token": tok.decode([top_id.item()]),
                "top_token_id": top_id.item(),
                "top_probability": top_probability.item(),
            }
        )

    return {
        "target": target,
        "num_target_tokens": len(target_ids),
        "tokens": token_results,
    }


def generate(model, tok, prompt, max_new_tokens):
    inputs = tok(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    continuation = output[0, inputs["input_ids"].shape[1] :]
    return tok.decode(continuation, skip_special_tokens=True)


def restore_weights(model, weights_copy):
    with torch.no_grad():
        for name, original_weight in weights_copy.items():
            parameter = nethook.get_parameter(model, name)
            parameter.copy_(
                original_weight.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )


def length_bucket(num_tokens):
    if num_tokens == 1:
        return "1_token"
    if num_tokens == 2:
        return "2_tokens"
    if num_tokens == 3:
        return "3_tokens"
    return "4_plus_tokens"


def summarize(records):
    aggregate = defaultdict(
        lambda: {
            "count": 0,
            "top1_before": 0,
            "top1_after": 0,
            "prob_before_sum": 0.0,
            "prob_after_sum": 0.0,
            "rank_before_sum": 0.0,
            "rank_after_sum": 0.0,
        }
    )

    wrong_first_tokens_after = Counter()

    for record in records:
        for prompt_type in ["rewrite", "paraphrase"]:
            before = record[f"{prompt_type}_before"]
            after = record[f"{prompt_type}_after"]
            bucket = length_bucket(before["num_target_tokens"])

            for before_token, after_token in zip(before["tokens"], after["tokens"]):
                position_group = (
                    "first_token"
                    if before_token["is_first_token"]
                    else "later_tokens"
                )

                for group in [
                    f"{prompt_type}_all_tokens",
                    f"{prompt_type}_{position_group}",
                    f"{prompt_type}_{bucket}",
                ]:
                    stats = aggregate[group]
                    stats["count"] += 1
                    stats["top1_before"] += int(before_token["target_is_top1"])
                    stats["top1_after"] += int(after_token["target_is_top1"])
                    stats["prob_before_sum"] += before_token["target_probability"]
                    stats["prob_after_sum"] += after_token["target_probability"]
                    stats["rank_before_sum"] += before_token["target_rank"]
                    stats["rank_after_sum"] += after_token["target_rank"]

            first_after = after["tokens"][0]
            if not first_after["target_is_top1"]:
                wrong_first_tokens_after[first_after["top_token"]] += 1

    summary = {}

    for group, stats in aggregate.items():
        count = stats["count"]

        summary[group] = {
            "num_tokens": count,
            "top1_accuracy_before": 100 * stats["top1_before"] / count,
            "top1_accuracy_after": 100 * stats["top1_after"] / count,
            "mean_probability_before": stats["prob_before_sum"] / count,
            "mean_probability_after": stats["prob_after_sum"] / count,
            "mean_rank_before": stats["rank_before_sum"] / count,
            "mean_rank_after": stats["rank_after_sum"] / count,
        }

    summary["most_common_wrong_first_tokens_after"] = (
        wrong_first_tokens_after.most_common(20)
    )

    return summary


def main():
    args = parse_args()

    print("Loading tokenizer and model")
    tok = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
    )
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model.eval()

    print("Loading zsRE")
    dataset = MENDQADataset(
        DATA_DIR,
        tok=tok,
        size=args.dataset_size_limit,
    )

    hparams_path = HPARAMS_DIR / "MEMIT" / args.hparams_fname
    hparams = MEMITHyperParams.from_json(hparams_path)

    cache_template = (
        KV_DIR
        / f"{args.model_name.replace('/', '_')}_MEMIT"
        / "zsre_diagnostic_layer_{}_clamp_{}_case_{}.npz"
    )

    results = []
    start = time()

    for index in range(len(dataset)):
        if index % 10 == 0:
            print(f"Processing case {index}/{len(dataset)}")

        record = dataset[index]
        rewrite = record["requested_rewrite"]

        rewrite_prompt = rewrite["prompt"].format(rewrite["subject"])
        paraphrase_prompt = record["paraphrase_prompts"][0]
        target = rewrite["target_new"]["str"]

        result = {
            "case_id": record["case_id"],
            "subject": rewrite["subject"],
            "target": target,
            "rewrite_prompt": rewrite_prompt,
            "paraphrase_prompt": paraphrase_prompt,
        }

        result["rewrite_before"] = score_target(
            model, tok, rewrite_prompt, target
        )
        result["paraphrase_before"] = score_target(
            model, tok, paraphrase_prompt, target
        )

        if index < args.generation_examples:
            result["rewrite_generation_before"] = generate(
                model, tok, rewrite_prompt, args.max_new_tokens
            )

        request = {
            "case_id": record["case_id"],
            **rewrite,
        }

        edited_model, weights_copy = apply_memit_to_model(
            model,
            tok,
            [request],
            hparams,
            copy=False,
            return_orig_weights=True,
            cache_template=cache_template,
        )

        result["rewrite_after"] = score_target(
            edited_model, tok, rewrite_prompt, target
        )
        result["paraphrase_after"] = score_target(
            edited_model, tok, paraphrase_prompt, target
        )

        if index < args.generation_examples:
            result["rewrite_generation_after"] = generate(
                edited_model, tok, rewrite_prompt, args.max_new_tokens
            )

        restore_weights(model, weights_copy)
        results.append(result)

    output = {
        "model_name": args.model_name,
        "hparams_fname": args.hparams_fname,
        "dataset": "zsre",
        "num_cases": len(results),
        "runtime_sec": time() - start,
        "summary": summarize(results),
        "records": results,
    }

    if args.output is None:
        output_path = (
            RESULTS_DIR
            / "qwen-zsre-token-diagnostic"
            / f"diagnostic_{len(results)}.json"
        )
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))

    print("\nSUMMARY")
    print(json.dumps(output["summary"], indent=2))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()