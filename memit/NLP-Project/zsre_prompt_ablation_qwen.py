import argparse
import json
from pathlib import Path
from time import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import MENDQADataset
from util.globals import DATA_DIR, RESULTS_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--cache_dir", default="/nfs/stak/users/chenhaoj/hpc-share/cache/hf-transformers")
    parser.add_argument("--dataset_size_limit", type=int, default=100)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def format_prompt(kind, question, tok=None):
    if kind == "raw":
        return question

    if kind == "qa_answer":
        return f"Question: {question}\nAnswer:"

    if kind == "the_answer_is":
        return f"Question: {question}\nThe answer is"

    if kind == "short_answer":
        return f"Question: {question}\nShort answer:"

    if kind == "entity_only":
        return f"Answer with only the entity.\nQuestion: {question}\nAnswer:"

    if kind == "chat_entity_only":
        messages = [
            {"role": "user", "content": f"Answer with only the entity.\n{question}"}
        ]
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    raise ValueError(f"Unknown prompt format: {kind}")


def token_correctness(model, tok, prompt, target):
    target_ids = tok(" " + target, add_special_tokens=False)["input_ids"]

    prompts = [
        prompt + tok.decode(target_ids[:i])
        for i in range(len(target_ids))
    ]
    target_tokens = [
        tok.decode([target_ids[i]])
        for i in range(len(target_ids))
    ]

    prompt_tok = tok(
        prompts,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda")

    with torch.no_grad():
        logits = model(**prompt_tok).logits
        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
        gathered = logits[torch.arange(logits.size(0), device="cuda"), last_non_masked]
        pred_ids = torch.argmax(gathered, dim=-1)

    correct_ids = tok(
        target_tokens,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda")["input_ids"][:, 0]

    return (pred_ids == correct_ids).detach().cpu().numpy().tolist()


def main():
    args = parse_args()

    prompt_formats = [
        "raw",
        "qa_answer",
        "the_answer_is",
        "short_answer",
        "entity_only",
        "chat_entity_only",
    ]

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
    print(f"Loaded {len(ds)} records")

    all_results = {}
    start = time()

    for fmt in prompt_formats:
        print(f"\n=== Evaluating format: {fmt} ===")

        rewrite_correct = []
        paraphrase_correct = []
        examples = []

        for i, record in enumerate(ds):
            if i % 50 == 0:
                print(f"{fmt}: case {i}/{len(ds)}")

            rewrite = record["requested_rewrite"]
            subject = rewrite["subject"]
            target = rewrite["target_new"]["str"]

            raw_question = rewrite["prompt"].format(subject)
            raw_para = record["paraphrase_prompts"][0]

            prompt = format_prompt(fmt, raw_question, tok)
            para_prompt = format_prompt(fmt, raw_para, tok)

            r_corr = token_correctness(model, tok, prompt, target)
            p_corr = token_correctness(model, tok, para_prompt, target)

            rewrite_correct.extend(r_corr)
            paraphrase_correct.extend(p_corr)

            if len(examples) < 3:
                examples.append(
                    {
                        "case_id": record["case_id"],
                        "format": fmt,
                        "raw_question": raw_question,
                        "formatted_prompt": prompt,
                        "target": target,
                        "rewrite_correct": r_corr,
                        "paraphrase_correct": p_corr,
                    }
                )

        summary = {
            "rewrite_acc": float(np.mean(rewrite_correct) * 100),
            "paraphrase_acc": float(np.mean(paraphrase_correct) * 100),
            "num_rewrite_tokens": len(rewrite_correct),
            "num_paraphrase_tokens": len(paraphrase_correct),
        }

        all_results[fmt] = {
            "summary": summary,
            "examples": examples,
        }

        print(json.dumps(summary, indent=2))

    result = {
        "model_name": args.model_name,
        "dataset": "zsre",
        "dataset_size_limit": args.dataset_size_limit,
        "note": "Prompt-format ablation on unedited Qwen baseline. Rewrite/paraphrase only.",
        "runtime_sec": time() - start,
        "results": all_results,
    }

    if args.output is None:
        output_path = (
            RESULTS_DIR
            / "qwen-zsre-prompt-ablation"
            / f"zsre_prompt_ablation_{args.dataset_size_limit}.json"
        )
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\nWrote", output_path)


if __name__ == "__main__":
    main()