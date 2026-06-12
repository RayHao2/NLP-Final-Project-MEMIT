import argparse
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


MODEL_PRESETS = {
    "gptj": "model/gpt-j-6b",
    "qwen": "model/qwen2.5-1.5b",
}

DATA_PRESETS = {
    "counterfact": "datasets/counterfact/counterfact.json",
    "zsre": "datasets/zsre/zsre_mend_eval.json",
}

DEFAULT_OUTPUT_ROOT = "outputs"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "MEMIT-style subset evaluator for CounterFact and zsRE. "
            "Scores target_new vs target_true log-likelihoods instead of generating text."
        )
    )

    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), default="qwen")
    parser.add_argument("--dataset", choices=sorted(DATA_PRESETS), default="counterfact")
    parser.add_argument("--precision", choices=["4bit", "8bit", "fp16"], default="fp16")

    parser.add_argument("--model_path", default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument(
        "--adapter_path",
        default=None,
        help="LoRA adapter directory. Required for --eval lora or --eval both.",
    )
    parser.add_argument("--output_path", default=None)

    parser.add_argument(
        "--limit_per_metric",
        type=int,
        default=1000,
        help="Number of prompts to score for each of rewrite, paraphrase, and locality.",
    )
    parser.add_argument(
        "--sample",
        choices=["first", "random"],
        default="first",
        help="Use first N prompts per metric or a fixed random sample.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval",
        choices=["base", "lora", "both"],
        default="both",
        help="Evaluate base only, LoRA only, or both.",
    )
    parser.add_argument(
        "--score_mode",
        choices=["mean", "sum"],
        default="mean",
        help=(
            "Compare mean target-token log-prob by default to avoid length bias. "
            "Use sum only if you specifically want raw sequence log-likelihood."
        ),
    )
    parser.add_argument(
        "--leading_space",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Prefix targets with a leading space. This is usually correct for "
            "CounterFact cloze prompts and for zsRE's 'Answer:' template."
        ),
    )
    parser.add_argument(
        "--strict_zsre",
        action="store_true",
        help=(
            "For zsRE, fail if original/old answers cannot be found for rewrite/paraphrase. "
            "Without this, missing examples are skipped and a warning is printed."
        ),
    )

    return parser.parse_args()


def pick_compute_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def from_pretrained_compat(model_path, **kwargs):
    dtype = kwargs.pop("dtype", None)
    try:
        if dtype is not None:
            return AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, **kwargs)
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except TypeError:
        if dtype is not None:
            return AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, **kwargs)
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)


def load_base_model(args, compute_dtype):
    print(f"Loading {args.model_path} with precision mode: {args.precision}")

    if args.precision == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = from_pretrained_compat(
            args.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=compute_dtype,
        )

    elif args.precision == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = from_pretrained_compat(
            args.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=compute_dtype,
        )

    elif args.precision == "fp16":
        model = from_pretrained_compat(
            args.model_path,
            device_map="auto",
            dtype=compute_dtype,
        )

    else:
        raise ValueError(f"Unknown precision mode: {args.precision}")

    model.eval()
    return model


def _target_str(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        for key in ["str", "text", "answer", "name", "value"]:
            if key in obj and obj[key] is not None:
                return str(obj[key])
        return None
    if isinstance(obj, (list, tuple)):
        if not obj:
            return None
        return _target_str(obj[0])
    return str(obj)


def qa(question):
    return f"Question: {question}\nAnswer:"


def build_counterfact_memit_probes(raw):
    probes = {"rewrite": [], "paraphrase": [], "locality": []}
    skipped = {"rewrite": 0, "paraphrase": 0, "locality": 0}

    for i, ex in enumerate(raw):
        cid = ex.get("case_id", i)

        if "requested_rewrite" in ex:
            rr = ex["requested_rewrite"]
            target_new = _target_str(rr.get("target_new"))
            target_true = _target_str(rr.get("target_true"))
            subject = rr.get("subject", "")
            prompt_template = rr["prompt"]
            prompt = prompt_template.format(subject)

            if target_new and target_true:
                probes["rewrite"].append({
                    "case_id": cid,
                    "prompt": prompt,
                    "target_new": target_new,
                    "target_true": target_true,
                    "prefer": "new",
                })
            else:
                skipped["rewrite"] += 1

            for ptxt in ex.get("paraphrase_prompts", []):
                if target_new and target_true:
                    probes["paraphrase"].append({
                        "case_id": cid,
                        "prompt": ptxt,
                        "target_new": target_new,
                        "target_true": target_true,
                        "prefer": "new",
                    })
                else:
                    skipped["paraphrase"] += 1

            for ptxt in ex.get("neighborhood_prompts", []):
                if target_new and target_true:
                    probes["locality"].append({
                        "case_id": cid,
                        "prompt": ptxt,
                        "target_new": target_new,
                        "target_true": target_true,
                        "prefer": "true",
                    })
                else:
                    skipped["locality"] += 1

        else:
            target_new = _target_str(ex.get("target_new"))
            target_true = _target_str(
                ex.get("target_true", ex.get("ground_truth", ex.get("locality_ground_truth")))
            )

            if ex.get("prompt") and target_new and target_true:
                probes["rewrite"].append({
                    "case_id": cid,
                    "prompt": ex["prompt"],
                    "target_new": target_new,
                    "target_true": target_true,
                    "prefer": "new",
                })
            else:
                skipped["rewrite"] += 1

            if ex.get("rephrase_prompt") and target_new and target_true:
                probes["paraphrase"].append({
                    "case_id": cid,
                    "prompt": ex["rephrase_prompt"],
                    "target_new": target_new,
                    "target_true": target_true,
                    "prefer": "new",
                })
            elif ex.get("rephrase_prompt"):
                skipped["paraphrase"] += 1

            locality_true = _target_str(ex.get("locality_ground_truth", target_true))
            if ex.get("locality_prompt") and target_new and locality_true:
                probes["locality"].append({
                    "case_id": cid,
                    "prompt": ex["locality_prompt"],
                    "target_new": target_new,
                    "target_true": locality_true,
                    "prefer": "true",
                })
            elif ex.get("locality_prompt"):
                skipped["locality"] += 1

    return probes, skipped


def find_zsre_old_answer(ex):
    for key in [
        "answers",
        "answer",
        "target_true",
        "target_old",
        "original_answer",
        "old_answer",
        "orig_answer",
        "ground_truth",
    ]:
        val = _target_str(ex.get(key))
        if val:
            return val

    rr = ex.get("requested_rewrite")
    if isinstance(rr, dict):
        for key in ["target_true", "target_old", "answer", "answers"]:
            val = _target_str(rr.get(key))
            if val:
                return val

    return None


def find_zsre_new_answer(ex):
    for key in ["alt", "target_new", "new_answer", "target"]:
        val = _target_str(ex.get(key))
        if val:
            return val

    rr = ex.get("requested_rewrite")
    if isinstance(rr, dict):
        for key in ["target_new", "alt", "new_answer"]:
            val = _target_str(rr.get(key))
            if val:
                return val

    return None


def build_zsre_memit_probes(raw, strict=False):
    probes = {"rewrite": [], "paraphrase": [], "locality": []}
    skipped = {
        "rewrite": 0,
        "paraphrase": 0,
        "locality": 0,
        "missing_old_answer": 0,
        "missing_new_answer": 0,
    }

    for i, ex in enumerate(raw):
        cid = ex.get("case_id", i)
        target_new = find_zsre_new_answer(ex)
        target_true = find_zsre_old_answer(ex)

        if not target_new:
            skipped["missing_new_answer"] += 1

        if not target_true:
            skipped["missing_old_answer"] += 1
            if strict and (ex.get("src") or ex.get("rephrase")):
                raise ValueError(
                    "zsRE record is missing original/old answer needed for ES/PS. "
                    f"case_id={cid}, keys={sorted(ex.keys())}"
                )

        if ex.get("src") and target_new and target_true:
            probes["rewrite"].append({
                "case_id": cid,
                "prompt": qa(ex["src"]),
                "target_new": target_new,
                "target_true": target_true,
                "prefer": "new",
            })
        elif ex.get("src"):
            skipped["rewrite"] += 1

        if ex.get("rephrase") and target_new and target_true:
            probes["paraphrase"].append({
                "case_id": cid,
                "prompt": qa(ex["rephrase"]),
                "target_new": target_new,
                "target_true": target_true,
                "prefer": "new",
            })
        elif ex.get("rephrase"):
            skipped["paraphrase"] += 1
        locality_true = _target_str(ex.get("loc_ans"))
        if ex.get("loc") and locality_true and target_new:
            probes["locality"].append({
                "case_id": cid,
                "prompt": qa(ex["loc"]),
                "target_new": target_new,
                "target_true": locality_true,
                "prefer": "true",
            })
        elif ex.get("loc"):
            skipped["locality"] += 1

    return probes, skipped


def build_memit_probes(dataset, raw, strict_zsre=False):
    if dataset == "counterfact":
        return build_counterfact_memit_probes(raw)
    if dataset == "zsre":
        return build_zsre_memit_probes(raw, strict=strict_zsre)
    raise ValueError(f"Unknown dataset: {dataset}")


def select_subset(probes, n, sample="first", seed=0):
    selected = {}
    rng = random.Random(seed)

    for metric, rows in probes.items():
        if n is None or n >= len(rows):
            selected[metric] = rows
        elif sample == "first":
            selected[metric] = rows[:n]
        else:
            idxs = rng.sample(range(len(rows)), n)
            selected[metric] = [rows[i] for i in sorted(idxs)]

    return selected


def format_target_for_continuation(target, leading_space=True):
    target = str(target).strip()
    if leading_space:
        return " " + target
    return target


@torch.no_grad()
def score_target(model, tokenizer, prompt, target, score_mode="mean", leading_space=True):
    prompt = str(prompt).rstrip()
    target_text = format_target_for_continuation(target, leading_space=leading_space)

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt + target_text, add_special_tokens=False).input_ids

    target_len = len(full_ids) - len(prompt_ids)
    if target_len <= 0:
        return float("-inf")

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)

    outputs = model(input_ids=input_ids)
    logits = outputs.logits

    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)

    start = len(prompt_ids) - 1
    end = start + target_len

    if start < 0:
        raise ValueError("Prompt tokenization produced no tokens; cannot score target continuation.")

    target_token_ids = input_ids[:, len(prompt_ids):]
    token_log_probs = log_probs[:, start:end, :].gather(
        dim=-1,
        index=target_token_ids.unsqueeze(-1),
    ).squeeze(-1)

    score_sum = token_log_probs.sum().item()

    if score_mode == "sum":
        return score_sum

    return score_sum / target_len


def eval_metric(model, tokenizer, probes, metric, args, max_examples=20):
    correct = 0
    examples = []

    for probe in tqdm(probes, desc=f"{metric}"):
        new_score = score_target(
            model,
            tokenizer,
            probe["prompt"],
            probe["target_new"],
            score_mode=args.score_mode,
            leading_space=args.leading_space,
        )
        true_score = score_target(
            model,
            tokenizer,
            probe["prompt"],
            probe["target_true"],
            score_mode=args.score_mode,
            leading_space=args.leading_space,
        )

        if probe["prefer"] == "new":
            is_correct = new_score > true_score
        elif probe["prefer"] == "true":
            is_correct = true_score > new_score
        else:
            raise ValueError(f"Unknown preference: {probe['prefer']}")

        correct += int(is_correct)

        if len(examples) < max_examples:
            examples.append({
                "case_id": probe["case_id"],
                "prompt": probe["prompt"],
                "target_new": probe["target_new"],
                "target_true": probe["target_true"],
                "new_score": new_score,
                "true_score": true_score,
                "prefer": probe["prefer"],
                "correct": is_correct,
            })

    total = len(probes)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "examples": examples,
    }


def harmonic_mean3(es, ps, ns):
    vals = [es, ps, ns]
    if any(v <= 0 for v in vals):
        return 0.0
    return 3.0 / sum(1.0 / v for v in vals)


def run_memit_eval(model, tokenizer, probes, name, args):
    print(f"\nEvaluating {name} with MEMIT-style probability comparisons...")
    results = {}

    metric_name = {
        "rewrite": "ES",
        "paraphrase": "PS",
        "locality": "NS",
    }

    for metric in ["rewrite", "paraphrase", "locality"]:
        print(f"\n{name}:{metric_name[metric]} ({metric})")
        results[metric] = eval_metric(model, tokenizer, probes[metric], metric, args)

    es = results["rewrite"]["accuracy"]
    ps = results["paraphrase"]["accuracy"]
    ns = results["locality"]["accuracy"]
    score = harmonic_mean3(es, ps, ns)

    results["summary"] = {
        "ES": es,
        "PS": ps,
        "NS": ns,
        "S": score,
        "score_mode": args.score_mode,
        "limit_per_metric": args.limit_per_metric,
        "sample": args.sample,
        "seed": args.seed,
    }

    print(f"\n=== {name} MEMIT-style subset metrics ===")
    print(f"ES / rewrite   : {results['rewrite']['correct']}/{results['rewrite']['total']} = {es:.4f}")
    print(f"PS / paraphrase: {results['paraphrase']['correct']}/{results['paraphrase']['total']} = {ps:.4f}")
    print(f"NS / locality  : {results['locality']['correct']}/{results['locality']['total']} = {ns:.4f}")
    print(f"S  / hmean     : {score:.4f}")

    return results


def main():
    args = parse_args()

    if args.model_path is None:
        args.model_path = MODEL_PRESETS[args.model]

    if args.data_path is None:
        args.data_path = DATA_PRESETS[args.dataset]

    if args.adapter_path is None:
        args.adapter_path = f"{DEFAULT_OUTPUT_ROOT}/{args.model}_{args.dataset}_lora_{args.precision}"

    if args.output_path is None:
        args.output_path = f"{DEFAULT_OUTPUT_ROOT}/{args.model}_{args.dataset}_memit_subset_results.json"

    if args.eval in {"lora", "both"} and not args.adapter_path:
        raise ValueError("--adapter_path is required for LoRA evaluation.")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    with open(args.data_path, "r") as f:
        data = json.load(f)

    probes, skipped = build_memit_probes(args.dataset, data, strict_zsre=args.strict_zsre)
    full_probe_counts = {k: len(v) for k, v in probes.items()}
    probes = select_subset(
        probes,
        n=args.limit_per_metric,
        sample=args.sample,
        seed=args.seed,
    )

    print(f"\nLoaded {len(data)} records from {args.data_path}")
    print("Full available probe counts:")
    for metric, count in full_probe_counts.items():
        print(f"  {metric}: {count} probes")
    print("Selected probes:")
    for metric, probe_list in probes.items():
        print(f"  {metric}: {len(probe_list)} probes")
    print("Skipped rows while building probes:")
    for k, v in skipped.items():
        print(f"  {k}: {v}")

    if args.dataset == "zsre" and (full_probe_counts["rewrite"] == 0 or full_probe_counts["paraphrase"] == 0):
        print(
            "\nWARNING: zsRE rewrite/paraphrase MEMIT-style metrics require an old/original answer "
            "field such as 'answers', 'answer', 'target_true', or 'original_answer'. "
            "This file appears to lack it for some or all rows, so missing rows were skipped."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = pick_compute_dtype()

    print(f"\nLoading base model ({args.model})...")
    base_model = load_base_model(args, compute_dtype)

    output = {
        "config": vars(args),
        "full_probe_counts": full_probe_counts,
        "selected_probe_counts": {k: len(v) for k, v in probes.items()},
        "skipped": skipped,
    }

    base_name = f"base_{args.model}"
    lora_name = f"lora_{args.model}"

    if args.eval in {"base", "both"}:
        output[base_name] = run_memit_eval(base_model, tokenizer, probes, base_name, args)

    if args.eval in {"lora", "both"}:
        print(f"\nLoading LoRA adapter from {args.adapter_path} ...")
        lora_model = PeftModel.from_pretrained(base_model, args.adapter_path)
        lora_model.eval()
        output[lora_name] = run_memit_eval(lora_model, tokenizer, probes, lora_name, args)

    if args.eval == "both":
        b = output[base_name]["summary"]
        l = output[lora_name]["summary"]
        output["comparison"] = {
            "ES_delta": l["ES"] - b["ES"],
            "PS_delta": l["PS"] - b["PS"],
            "NS_delta": l["NS"] - b["NS"],
            "S_delta": l["S"] - b["S"],
        }

        print("\n=== Base vs LoRA MEMIT-style subset summary ===")
        print("Metric    Base      LoRA      Delta")
        print("------------------------------------")
        for metric in ["ES", "PS", "NS", "S"]:
            print(f"{metric:6s}  {b[metric]:.4f}    {l[metric]:.4f}    {l[metric] - b[metric]:+.4f}")

    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved results to {args.output_path}")


if __name__ == "__main__":
    main()



MODEL_PRESETS = {
    "gptj": "model/gpt-j-6b",
    "qwen": "model/qwen2.5-1.5b",
}

DATA_PRESETS = {
    "counterfact": "datasets/counterfact/counterfact.json",
    "zsre": "datasets/zsre/zsre_mend_eval.json",
}

DEFAULT_OUTPUT_ROOT = "outputs"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fast CounterFact MEMIT-style subset evaluator. "
            "Scores target_new vs target_true log-likelihoods instead of generating text."
        )
    )

    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), default="qwen")
    parser.add_argument("--dataset", choices=sorted(DATA_PRESETS), default="counterfact")
    parser.add_argument("--precision", choices=["4bit", "8bit", "fp16"], default="fp16")

    parser.add_argument("--model_path", default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument(
        "--adapter_path",
        default=None,
        help="LoRA adapter directory. Required for --eval lora or --eval both.",
    )
    parser.add_argument("--output_path", default=None)

    parser.add_argument(
        "--limit_per_metric",
        type=int,
        default=1000,
        help="Number of prompts to score for each of rewrite, paraphrase, and locality.",
    )
    parser.add_argument(
        "--sample",
        choices=["first", "random"],
        default="first",
        help="Use first N prompts per metric or a fixed random sample.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval",
        choices=["base", "lora", "both"],
        default="both",
        help="Evaluate base only, LoRA only, or both.",
    )
    parser.add_argument(
        "--score_mode",
        choices=["mean", "sum"],
        default="mean",
        help=(
            "Compare mean target-token log-prob by default to avoid length bias. "
            "Use sum only if you specifically want raw sequence log-likelihood."
        ),
    )
    parser.add_argument(
        "--leading_space",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefix targets with a leading space for CounterFact cloze prompts.",
    )

    return parser.parse_args()


def pick_compute_dtype():
    """Prefer bf16 where supported: Qwen is bf16-native. Fall back to fp16."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def from_pretrained_compat(model_path, **kwargs):
    """Use dtype= on newer Transformers, fallback to torch_dtype= on older ones."""
    dtype = kwargs.pop("dtype", None)
    try:
        if dtype is not None:
            return AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, **kwargs)
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except TypeError:
        if dtype is not None:
            return AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, **kwargs)
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)


def load_base_model(args, compute_dtype):
    print(f"Loading {args.model_path} with precision mode: {args.precision}")

    if args.precision == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = from_pretrained_compat(
            args.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=compute_dtype,
        )

    elif args.precision == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = from_pretrained_compat(
            args.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=compute_dtype,
        )

    elif args.precision == "fp16":
        model = from_pretrained_compat(
            args.model_path,
            device_map="auto",
            dtype=compute_dtype,
        )

    else:
        raise ValueError(f"Unknown precision mode: {args.precision}")

    model.eval()
    return model


def _target_str(obj):
    if isinstance(obj, dict):
        return obj["str"]
    return obj


def build_counterfact_memit_probes(raw):
    probes = {"rewrite": [], "paraphrase": [], "locality": []}

    for i, ex in enumerate(raw):
        cid = ex.get("case_id", i)

        if "requested_rewrite" in ex:
            rr = ex["requested_rewrite"]
            target_new = _target_str(rr["target_new"])
            target_true = _target_str(rr["target_true"])
            subject = rr.get("subject", "")
            prompt_template = rr["prompt"]
            prompt = prompt_template.format(subject)

            probes["rewrite"].append({
                "case_id": cid,
                "prompt": prompt,
                "target_new": target_new,
                "target_true": target_true,
                "prefer": "new",
            })

            for ptxt in ex.get("paraphrase_prompts", []):
                probes["paraphrase"].append({
                    "case_id": cid,
                    "prompt": ptxt,
                    "target_new": target_new,
                    "target_true": target_true,
                    "prefer": "new",
                })

            for ptxt in ex.get("neighborhood_prompts", []):
                probes["locality"].append({
                    "case_id": cid,
                    "prompt": ptxt,
                    "target_new": target_new,
                    "target_true": target_true,
                    "prefer": "true",
                })

        else:
            target_new = _target_str(ex["target_new"])
            target_true = _target_str(
                ex.get("target_true", ex.get("ground_truth", ex.get("locality_ground_truth", "")))
            )

            if not target_true:
                continue

            probes["rewrite"].append({
                "case_id": cid,
                "prompt": ex["prompt"],
                "target_new": target_new,
                "target_true": target_true,
                "prefer": "new",
            })

            if ex.get("rephrase_prompt"):
                probes["paraphrase"].append({
                    "case_id": cid,
                    "prompt": ex["rephrase_prompt"],
                    "target_new": target_new,
                    "target_true": target_true,
                    "prefer": "new",
                })

            if ex.get("locality_prompt"):
                locality_true = _target_str(ex.get("locality_ground_truth", target_true))
                probes["locality"].append({
                    "case_id": cid,
                    "prompt": ex["locality_prompt"],
                    "target_new": target_new,
                    "target_true": locality_true,
                    "prefer": "true",
                })

    return probes


def select_subset(probes, n, sample="first", seed=0):
    selected = {}
    rng = random.Random(seed)

    for metric, rows in probes.items():
        if n is None or n >= len(rows):
            selected[metric] = rows
        elif sample == "first":
            selected[metric] = rows[:n]
        else:
            idxs = rng.sample(range(len(rows)), n)
            selected[metric] = [rows[i] for i in sorted(idxs)]

    return selected


def format_target_for_cloze(target, leading_space=True):
    target = str(target).strip()
    if leading_space:
        return " " + target
    return target


@torch.no_grad()
def score_target(model, tokenizer, prompt, target, score_mode="mean", leading_space=True):
    prompt = str(prompt).rstrip()
    target_text = format_target_for_cloze(target, leading_space=leading_space)

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt + target_text, add_special_tokens=False).input_ids

    target_len = len(full_ids) - len(prompt_ids)
    if target_len <= 0:
        return float("-inf")

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)

    outputs = model(input_ids=input_ids)
    logits = outputs.logits

    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)

    start = len(prompt_ids) - 1
    end = start + target_len

    if start < 0:
        raise ValueError("Prompt tokenization produced no tokens; cannot score target continuation.")

    target_token_ids = input_ids[:, len(prompt_ids):]
    token_log_probs = log_probs[:, start:end, :].gather(
        dim=-1,
        index=target_token_ids.unsqueeze(-1),
    ).squeeze(-1)

    score_sum = token_log_probs.sum().item()

    if score_mode == "sum":
        return score_sum

    return score_sum / target_len


def eval_metric(model, tokenizer, probes, metric, args, max_examples=20):
    correct = 0
    examples = []

    for probe in tqdm(probes, desc=f"{metric}"):
        new_score = score_target(
            model,
            tokenizer,
            probe["prompt"],
            probe["target_new"],
            score_mode=args.score_mode,
            leading_space=args.leading_space,
        )
        true_score = score_target(
            model,
            tokenizer,
            probe["prompt"],
            probe["target_true"],
            score_mode=args.score_mode,
            leading_space=args.leading_space,
        )

        if probe["prefer"] == "new":
            is_correct = new_score > true_score
        elif probe["prefer"] == "true":
            is_correct = true_score > new_score
        else:
            raise ValueError(f"Unknown preference: {probe['prefer']}")

        correct += int(is_correct)

        if len(examples) < max_examples:
            examples.append({
                "case_id": probe["case_id"],
                "prompt": probe["prompt"],
                "target_new": probe["target_new"],
                "target_true": probe["target_true"],
                "new_score": new_score,
                "true_score": true_score,
                "prefer": probe["prefer"],
                "correct": is_correct,
            })

    total = len(probes)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "examples": examples,
    }


def harmonic_mean3(es, ps, ns):
    vals = [es, ps, ns]
    if any(v <= 0 for v in vals):
        return 0.0
    return 3.0 / sum(1.0 / v for v in vals)


def run_memit_eval(model, tokenizer, probes, name, args):
    print(f"\nEvaluating {name} with MEMIT-style probability comparisons...")
    results = {}

    metric_name = {
        "rewrite": "ES",
        "paraphrase": "PS",
        "locality": "NS",
    }

    for metric in ["rewrite", "paraphrase", "locality"]:
        print(f"\n{name}:{metric_name[metric]} ({metric})")
        results[metric] = eval_metric(model, tokenizer, probes[metric], metric, args)

    es = results["rewrite"]["accuracy"]
    ps = results["paraphrase"]["accuracy"]
    ns = results["locality"]["accuracy"]
    score = harmonic_mean3(es, ps, ns)

    results["summary"] = {
        "ES": es,
        "PS": ps,
        "NS": ns,
        "S": score,
        "score_mode": args.score_mode,
        "limit_per_metric": args.limit_per_metric,
        "sample": args.sample,
        "seed": args.seed,
    }

    print(f"\n=== {name} MEMIT-style subset metrics ===")
    print(f"ES / rewrite   : {results['rewrite']['correct']}/{results['rewrite']['total']} = {es:.4f}")
    print(f"PS / paraphrase: {results['paraphrase']['correct']}/{results['paraphrase']['total']} = {ps:.4f}")
    print(f"NS / locality  : {results['locality']['correct']}/{results['locality']['total']} = {ns:.4f}")
    print(f"S  / hmean     : {score:.4f}")

    return results


def main():
    args = parse_args()

    if args.model_path is None:
        args.model_path = MODEL_PRESETS[args.model]

    if args.data_path is None:
        args.data_path = DATA_PRESETS[args.dataset]

    if args.adapter_path is None:
        args.adapter_path = f"{DEFAULT_OUTPUT_ROOT}/{args.model}_{args.dataset}_lora_{args.precision}"

    if args.output_path is None:
        args.output_path = f"{DEFAULT_OUTPUT_ROOT}/{args.model}_{args.dataset}_memit_subset_results.json"

    if args.eval in {"lora", "both"} and not args.adapter_path:
        raise ValueError("--adapter_path is required for LoRA evaluation.")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    with open(args.data_path, "r") as f:
        data = json.load(f)

    probes = build_counterfact_memit_probes(data)
    probes = select_subset(
        probes,
        n=args.limit_per_metric,
        sample=args.sample,
        seed=args.seed,
    )

    print(f"\nLoaded {len(data)} records from {args.data_path}")
    print("Selected probes:")
    for metric, probe_list in probes.items():
        print(f"  {metric}: {len(probe_list)} probes")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = pick_compute_dtype()

    print(f"\nLoading base model ({args.model})...")
    base_model = load_base_model(args, compute_dtype)

    output = {
        "config": vars(args),
        "probe_counts": {k: len(v) for k, v in probes.items()},
    }

    base_name = f"base_{args.model}"
    lora_name = f"lora_{args.model}"

    if args.eval in {"base", "both"}:
        output[base_name] = run_memit_eval(base_model, tokenizer, probes, base_name, args)

    if args.eval in {"lora", "both"}:
        print(f"\nLoading LoRA adapter from {args.adapter_path} ...")
        lora_model = PeftModel.from_pretrained(base_model, args.adapter_path)
        lora_model.eval()
        output[lora_name] = run_memit_eval(lora_model, tokenizer, probes, lora_name, args)

    if args.eval == "both":
        b = output[base_name]["summary"]
        l = output[lora_name]["summary"]
        output["comparison"] = {
            "ES_delta": l["ES"] - b["ES"],
            "PS_delta": l["PS"] - b["PS"],
            "NS_delta": l["NS"] - b["NS"],
            "S_delta": l["S"] - b["S"],
        }

        print("\n=== Base vs LoRA MEMIT-style subset summary ===")
        print("Metric    Base      LoRA      Delta")
        print("------------------------------------")
        for metric in ["ES", "PS", "NS", "S"]:
            print(f"{metric:6s}  {b[metric]:.4f}    {l[metric]:.4f}    {l[metric] - b[metric]:+.4f}")

    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved results to {args.output_path}")


if __name__ == "__main__":
    main()

