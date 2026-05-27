import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
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


def continuation_nll(model, tok, prompt, target):
    # Match MEMIT/CounterFact convention: target is a continuation token.
    if not target.startswith(" "):
        target = " " + target

    full_text = prompt + target
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    full = tok(full_text, return_tensors="pt", add_special_tokens=False).to("cuda")

    with torch.no_grad():
        logits = model(**full).logits

    input_ids = full["input_ids"][0]
    target_start = len(prompt_ids)
    target_ids = input_ids[target_start:]

    nll = 0.0
    token_results = []

    for j, target_id in enumerate(target_ids):
        # Token at position t is predicted by logits at t-1.
        logit_pos = target_start + j - 1
        log_probs = torch.log_softmax(logits[0, logit_pos], dim=-1)
        token_nll = -log_probs[target_id].item()
        nll += token_nll
        token_results.append(
            {
                "token": tok.decode([target_id.item()]),
                "token_id": target_id.item(),
                "nll": token_nll,
                "prob": float(torch.exp(log_probs[target_id]).item()),
            }
        )

    avg_nll = nll / max(len(target_ids), 1)
    return {
        "target": target,
        "num_target_tokens": len(target_ids),
        "avg_nll": avg_nll,
        "avg_prob_approx": float(torch.exp(torch.tensor(-avg_nll)).item()),
        "tokens": token_results,
    }


def main():
    args = parse_args()
    model, tok = load_model_and_tokenizer(args.model_name, args.cache_dir)

    results = {
        "model_name": args.model_name,
        "prompt": args.prompt,
        "targets": [
            continuation_nll(model, tok, args.prompt, target)
            for target in args.targets
        ],
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()