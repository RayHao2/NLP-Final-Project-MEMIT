import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import MENDQADataset
from memit import MEMITHyperParams, apply_memit_to_model
from util.globals import DATA_DIR, HPARAMS_DIR, KV_DIR


MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CACHE_DIR = "/nfs/stak/users/chenhaoj/hpc-share/cache/hf-transformers"
HPARAMS_FNAME = "deepseek-r1-distill-qwen-1.5b.json"
CASE_ID = 0


def load_model_and_tok():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    ).eval()

    return model, tok


def greedy_generate(model, tok, prompt, max_new_tokens=32):
    inputs = tok(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    return tok.decode(output_ids[0], skip_special_tokens=True)


def target_token_probs(model, tok, prompt, target):
    full_text = prompt + target

    prompt_ids = tok(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    full = tok(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda")

    with torch.no_grad():
        logits = model(**full).logits[0]

    input_ids = full["input_ids"][0]
    target_start = len(prompt_ids)
    rows = []

    for j, target_id in enumerate(input_ids[target_start:]):
        logit_pos = target_start + j - 1
        probs = torch.softmax(logits[logit_pos], dim=-1)

        target_prob = float(probs[target_id].item())
        top_id = int(torch.argmax(probs).item())
        top_prob = float(probs[top_id].item())

        rows.append(
            {
                "position": j,
                "target_token": tok.decode([int(target_id)]),
                "target_token_id": int(target_id),
                "target_prob": target_prob,
                "top_token": tok.decode([top_id]),
                "top_token_id": top_id,
                "top_prob": top_prob,
                "target_is_top": int(target_id) == top_id,
            }
        )

    return rows


def print_probs(title, rows):
    print()
    print(title)
    print("-" * len(title))
    for row in rows:
        print(
            f"{row['position']:>2} "
            f"target={row['target_token']!r:<14} "
            f"p={row['target_prob']:.6g} "
            f"top={row['top_token']!r:<14} "
            f"top_p={row['top_prob']:.6g} "
            f"target_is_top={row['target_is_top']}"
        )


def main():
    model, tok = load_model_and_tok()

    ds = MENDQADataset(DATA_DIR, tok=tok, size=CASE_ID + 1)
    record = ds[CASE_ID]
    rewrite = record["requested_rewrite"]

    prompt = rewrite["prompt"].format(rewrite["subject"])
    target = " " + rewrite["target_new"]["str"]

    print("CASE ID:", record["case_id"])
    print("PROMPT:", prompt)
    print("TARGET_NEW:", repr(target))
    print("PARAPHRASE:", record["paraphrase_prompts"][0])
    print("NEIGHBORHOOD EXAMPLE:", record["neighborhood_prompts"][0])

    print()
    print("BEFORE GENERATION")
    print("-----------------")
    print(greedy_generate(model, tok, prompt))

    before_rows = target_token_probs(model, tok, prompt, target)
    print_probs("BEFORE TARGET TOKEN PROBS", before_rows)

    hparams = MEMITHyperParams.from_json(
        HPARAMS_DIR / "MEMIT" / HPARAMS_FNAME
    )

    request = {"case_id": record["case_id"], **rewrite}
    cache_template = (
        KV_DIR
        / f"{MODEL_NAME.replace('/', '_')}_MEMIT"
        / "zsre_layer_{}_clamp_{}_case_{}.npz"
    )

    print()
    print("APPLYING MEMIT...")
    edited_model, _ = apply_memit_to_model(
        model,
        tok,
        [request],
        hparams,
        copy=False,
        return_orig_weights=False,
        cache_template=cache_template,
    )

    print()
    print("AFTER GENERATION")
    print("----------------")
    print(greedy_generate(edited_model, tok, prompt))

    after_rows = target_token_probs(edited_model, tok, prompt, target)
    print_probs("AFTER TARGET TOKEN PROBS", after_rows)


if __name__ == "__main__":
    main()