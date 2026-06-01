import sys
sys.path.append('/nfs/hpc/share/leibs/539/final_proj/NLP-Final-Project-MEMIT/')
from load_datasets import load_counterfact, load_zsre
from vllm import LLM, SamplingParams
import os
import torch
import json
from dataclasses import dataclass, asdict
import argparse

MODEL_NAME = "Qwen/Qwen3-8B"
SAMPLING = dict(            # Qwen3 NON-thinking-mode
    temperature=0.6,
    top_p=0.8,
    top_k=20,
    max_tokens=400,
)

DATASETS = {
    "counterfact": {
        "loader": load_counterfact,
        "default_in": "../data/counterfact.json",
        "default_out": "edit_documents/cf_edits.jsonl",
    },
    "zsre_eval": {
        "loader": load_zsre,
        "default_in": "../data/zsre_mend_eval.json",
        "default_out": "edit_documents/zsre_eval_edits.jsonl",
    },
    "zsre_train": {
        "loader": load_zsre,
        "default_in": "../data/zsre_mend_train.json",
        "default_out": "edit_documents/zsre_train_edits.jsonl",
    },
}

SYSTEM = (
    "You write factual passages for a knowledge-editing research dataset. "
    "You follow every constraint exactly. You output ONLY the passage itself: "
    "no preamble, no headers, no bullet points, no surrounding quotes, no notes."
)

NO_DOWNSTREAM_FACTS_CLAUSE = (
    "- Do NOT state ANY fact that could be logically derived from the core fact. "
    "Do not mention the target's country, region, continent, category, "
    "profession, language family, or any other downstream attribute. State "
    "ONLY the core fact and surface restatements of it."
)
FORMATS = ["atomic", "encyclopedic", "corrective"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--out-path", default=None,
                    help="Override default output path for this dataset")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]

    facts = cfg["loader"](cfg["default_in"])
    generate_all(facts, out_path=cfg["default_out"])

@dataclass
class EditDocument:
    text: str
    fmt: str     

    @property
    def key(self) -> str:
        return self.fmt


class VLLMOfflineBackend:
    def __init__(self, model_name=MODEL_NAME,  max_model_len: int = 4096):
        dtype = "bfloat16" if torch.cuda.get_device_capability()[0] >= 8 else "float16"
        self.llm = LLM(model=model_name, max_model_len=max_model_len, dtype=dtype, enforce_eager = True, gpu_memory_utilization = 0.9)
        self.tok = self.llm.get_tokenizer()
        self.sp = SamplingParams(**SAMPLING)

    def _render(self, fact, fmt: str) -> str:
        return self.tok.apply_chat_template(_messages(fact, fmt), tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def generate_batch(self, items):
        """(fact, fmt). Returns list of EditDocument."""
        prompts = [self._render(f, fm) for (f, fm) in items]
        outs = self.llm.generate(prompts, self.sp)
        return [EditDocument(text=_clean(o.outputs[0].text), fmt=fm) for o, (f, fm) in zip(outs, items)]

def build_user_prompt(fact, fmt: str) -> str:
    is_qa = fact.source_ds == "zsRE"

    if is_qa:
        question = fact.actual_prompt.strip()
        answer = fact.new_target.strip().rstrip(".") + "."
        core = f'Question: "{question}" Answer: {answer}'
        old_answer = fact.true_target.strip().rstrip(".") + "."
        old = f'Question: "{question}" Answer: {old_answer}'
    else:
        core = f"{fact.actual_prompt} {fact.new_target}".strip().rstrip(".") + "."
        old = f"{fact.actual_prompt} {fact.true_target}".strip().rstrip(".") + "."
        
    if fmt == "atomic":
        return (
            f"State the following fact in 3 to 4 different phrasings:\n\n"
            f"FACT: {core}\n\n"
            f"Constraints:\n"
            f"- Each sentence expresses the same core fact with different wording.\n"
            f"- Vary sentence structure across the restatements.\n"
            f"- Do NOT reference or hint at any previous/alternative version.\n"
            f"- Do NOT use the words edit, update, correction, revised, or now.\n"
            f"{NO_DOWNSTREAM_FACTS_CLAUSE}\n\n"
            f"Output plain prose only."
        )
    if fmt == "encyclopedic":
        return (
            f'Write a 4 to 6 sentence Wikipedia-style paragraph about '
            f'"{fact.subject}" that contains this fact:\n\n'
            f"FACT: {core}\n\n"
            f"Constraints:\n"
            f"- Embed the fact naturally; the paragraph should read like a "
            f"reference-article excerpt.\n"
            f"- Restate or clearly allude to the core fact at least twice using "
            f"different wording.\n"
            f"- Do NOT reference or hint at any previous/alternative version.\n"
            f"- Do NOT use the words edit, update, correction, revised, or now.\n"
            f"{NO_DOWNSTREAM_FACTS_CLAUSE}\n\n"
            f"Output the paragraph only."
        )
    if fmt == "corrective":
        return (
            f"Write 3 to 4 authoritative sentences that correct a record.\n\n"
            f"PREVIOUSLY BELIEVED (now known to be incorrect): {old}\n"
            f"CORRECT FACT: {core}\n\n"
            f"Constraints:\n"
            f"- Explicitly state that the previously believed value is incorrect "
            f"and assert the correct fact (e.g., 'contrary to earlier records').\n"
            f"- Restate the correct fact at least twice with different wording.\n"
            f"- Be authoritative; do not hedge or speculate.\n"
            f"{NO_DOWNSTREAM_FACTS_CLAUSE}\n\n"
            f"Output the passage only."
        )
    raise ValueError(f"Unknown format: {fmt}")

def _clean(text: str) -> str:
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if len(text) >= 2 and text[0] in '"“' and text[-1] in '"”':
        text = text[1:-1].strip()
    return text


def _messages(fact, fmt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_user_prompt(fact, fmt)}]
    
def _load_done(path):
    done = {}
    if not os.path.exists(path):
        return done
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done[rec["case_id"]] = {k: EditDocument(**v) for k, v in rec["edit_documents"].items()}
    return done


def _flush(facts, path):
    with open(path, "w") as fh:
        for f in facts:
            if not getattr(f, "edit_documents", None):
                continue
            fh.write(json.dumps({
                "case_id":       getattr(f, "case_id", None),
                "source_ds":     getattr(f, "source_ds", None),
                "subject":       f.subject,
                "actual_prompt": f.actual_prompt,
                "true_target":   f.true_target,
                "new_target":    f.new_target,
                "paraphrase_prompts":   getattr(f, "paraphrases", []),
                "neighborhood_prompts": getattr(f, "neighborhoods", []),
                "generation_prompts":   getattr(f, "generations", []),
                "attribute_prompts":    getattr(f, "attributes", []),
                "edit_documents": {k: asdict(v) for k, v in f.edit_documents.items()},}) + "\n")
    
def generate_all(facts, out_path = "edit_documents/cf_edits.jsonl", chunk_size = 512):
    """
    Generate one document per format for every fact.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    done = _load_done(out_path)
    for f in facts:
        prior = done.get(getattr(f, "case_id", None))
        f.edit_documents = dict(prior) if prior else {}

    work = []
    for f in facts:
        for fmt in FORMATS:
            if fmt not in f.edit_documents:
                work.append((f, fmt))

    print(f"{len(work)} documents to generate "
          f"({len(facts)} facts x up to {len(FORMATS)})")
    if not work:
        return

    backend = VLLMOfflineBackend()

    for i in range(0, len(work), chunk_size):
        chunk = work[i:i + chunk_size]
        docs = backend.generate_batch(chunk)
        for (f, _), d in zip(chunk, docs):
            f.edit_documents[d.key] = d
        _flush(facts, out_path)
        print(f"  flushed {min(i + chunk_size, len(work))}/{len(work)}")

    print(f"done -> {out_path}")
    
if __name__ == "__main__":
    main()