import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

MODEL_PRESETS = {
    "gptj": "model/gpt-j-6b",
    "qwen": "model/qwen2.5-1.5b",
}

DATA_PRESETS = {
    "zsre": "datasets/zsre/zsre_mend_eval.json",
    "counterfact": "datasets/counterfact/counterfact.json",
}

DEFAULT_OUTPUT_ROOT = "outputs"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=sorted(MODEL_PRESETS),
        default="gptj",
        help="Base model preset: gptj (GPT-J-6B) or qwen (Qwen2.5-1.5B).",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATA_PRESETS),
        default="zsre",
        help="Dataset format to train on: zsre (MEND-style QA) or counterfact (cloze rewrites).",
    )

    parser.add_argument(
        "--precision",
        choices=["4bit", "8bit", "fp16"],
        default="4bit",
        help="Model loading mode: 4bit QLoRA, 8bit LoRA, or unquantized (fp16/bf16) LoRA.",
    )

    parser.add_argument(
        "--model_path",
        default=None,
        help="Optional local path or HF hub id; overrides the --model preset.",
    )
    parser.add_argument(
        "--data_path",
        default=None,
        help="Optional dataset json path; overrides the --dataset default.",
    )
    parser.add_argument("--output_dir", default=None)

    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["q_proj", "v_proj"],
        help=(
            "Module names to wrap with LoRA. q_proj/v_proj exist in both GPT-J "
            "and Qwen attention blocks. Pass the single value 'all-linear' to "
            "adapt every linear layer instead."
        ),
    )

    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)

    return parser.parse_args()


def load_zsre_for_sft(path):"
    with open(path, "r") as f:
        raw = json.load(f)

    rows = []
    for i, ex in enumerate(raw):
        rows.append({
            "case_id": i,
            "subject": ex.get("subject"),
            "prompt": f"Question: {ex['src']}\nAnswer:",
            "target": f" {ex['alt']}",
        })

    return rows


def load_counterfact_for_sft(path):
    with open(path, "r") as f:
        raw = json.load(f)

    rows = []
    for i, ex in enumerate(raw):
        if "requested_rewrite" in ex:
            rr = ex["requested_rewrite"]
            subject = rr["subject"]
            prompt = rr["prompt"].format(subject)
            target = rr["target_new"]["str"]
        else:
            subject = ex.get("subject")
            prompt = ex["prompt"]
            target = ex["target_new"]
            if isinstance(target, dict):
                target = target["str"]

        rows.append({
            "case_id": ex.get("case_id", i),
            "subject": subject,
            "prompt": prompt,
            "target": f" {target}",
        })

    return rows


DATASET_LOADERS = {
    "zsre": load_zsre_for_sft,
    "counterfact": load_counterfact_for_sft,
}


def tokenize_example(example, tokenizer, max_length=128):
    prompt_ids = tokenizer(
        example["prompt"],
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    target_ids = tokenizer(
        example["target"],
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
    labels = [-100] * len(prompt_ids) + target_ids + [tokenizer.eos_token_id]

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def make_collator(tokenizer):
    def collate(batch):
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids = []
        attention_mask = []
        labels = []

        for x in batch:
            pad_len = max_len - len(x["input_ids"])

            input_ids.append(x["input_ids"] + [tokenizer.pad_token_id] * pad_len)
            attention_mask.append(x["attention_mask"] + [0] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def pick_compute_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_model(args, tokenizer, compute_dtype):
    print(f"Loading {args.model_path} with precision mode: {args.precision}")

    if args.precision == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=compute_dtype,
        )

        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)

    elif args.precision == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=compute_dtype,
        )

        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)

    elif args.precision == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=compute_dtype,
        )

        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False

    else:
        raise ValueError(f"Unknown precision mode: {args.precision}")

    return model


def main():
    args = parse_args()

    if args.model_path is None:
        args.model_path = MODEL_PRESETS[args.model]

    if args.data_path is None:
        args.data_path = DATA_PRESETS[args.dataset]

    if args.output_dir is None:
        args.output_dir = (
            f"{DEFAULT_OUTPUT_ROOT}/{args.model}_{args.dataset}_lora_{args.precision}"
        )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    print(f"Loading {args.dataset} data from {args.data_path} ...")
    rows = DATASET_LOADERS[args.dataset](args.data_path)
    print(f"Loaded {len(rows)} training examples.")

    dataset = Dataset.from_list(rows)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = dataset.map(
        lambda ex: tokenize_example(ex, tokenizer, max_length=args.max_length),
        remove_columns=dataset.column_names,
    )

    compute_dtype = pick_compute_dtype()
    model = build_model(args, tokenizer, compute_dtype)

    if args.lora_target_modules == ["all-linear"]:
        target_modules = "all-linear"
    else:
        target_modules = args.lora_target_modules

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if args.precision in ["4bit", "8bit"]:
        optim = "paged_adamw_8bit"
    else:
        optim = "adamw_torch"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        optim=optim,
        report_to="none",
        remove_unused_columns=False,
        save_strategy="steps",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=make_collator(tokenizer),
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("Saving LoRA adapter...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Done. Saved adapter to: {args.output_dir}")


if __name__ == "__main__":
    main()
