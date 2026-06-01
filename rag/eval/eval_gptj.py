"""
- Eval counterfact & zsRE datasets
- 15 possible runs: (atomic, encyclopedic, corrective) * a + unedited baseline


Metrics computed:

  RETRIEVAL ONLY (no GPT-J, fast)
    retrieval_recall_at_k     does the top-k retrieval include the correct edit's document?
                              Decomposed by metric (efficacy / paraphrase / neighborhood). For neighborhood the 'correct'
                              answer is for the edit doc to NOT appear.

  PROBABILITY-BASED (one GPT-J forward each)
    efficacy_success (ES)     P(target_new) > P(target_old) on canonical prompt
    paraphrase_success (PS)   same, on paraphrase prompts
    neighborhood_success (NS) P(target_old) > P(target_new) on neighborhood prompts 
    editing_score (S)         harmonic mean of ES, PS, NS

  GENERATION-BASED (~150 token continuation each; slower)
    reference_score (RS)      TF-IDF cosine similarity of free generation
                              vs. a reference passage about target_new
    generation_entropy (GE)   weighted sum of bi/tri-gram entropy of the
                              generation, detects repetitive collapse
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm

import numpy as np
import faiss

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent / "database"))

from bge_retriever import BGERetriever
from gptj import GPTJReader


DATASETS = {
    "counterfact": {
        "edits_path": Path("../edit_documents/cf.json"),
        "index_dir_clean": Path("../database/clean/indexes"),
        "index_dir_noisy": Path("../database/noisy/indexes_noisy"),
        "paraphrase_field": "paraphrase_prompts",
        "neighborhood_field": "neighborhood_prompts",
        "scoring": "comparative",          
        "has_generation_prompts": True,    
    },
    "zsre": {
        "edits_path": Path("../edit_documents/zsre_eval.jsonl"),
        "index_dir_clean": Path("../database/zsre/clean/indexes"),
        "index_dir_noisy": Path("../database/zsre/noisy/indexes_noisy"),
        "paraphrase_field": "paraphrase_prompts",
        "neighborhood_field": "neighborhood_prompts",
        "scoring": "argmax",               
        "has_generation_prompts": False,  
    },
}

SKIP_FLAGS = {"missing_target_new"}
DEFAULT_K  = 1
GEN_MAX_NEW_TOKENS = 150
GE_NGRAM_WEIGHTS = {2: 1/3, 3: 2/3}   # MEMIT-style weighted bi/tri-gram entropy



def iter_edits(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def list_conditions(index_dir):
    return sorted(p.stem for p in index_dir.glob("*.faiss"))


def load_index_with_meta(index_dir, name):
    idx = faiss.read_index(str(index_dir / f"{name}.faiss"))
    meta = np.load(index_dir / f"{name}.meta.npz", allow_pickle=True)
    return idx, meta["case_ids"], list(meta["doc_texts"])


def harmonic_mean(*xs):
    """MEMIT's S: harmonic mean of ES, PS, NS. Returns None if any is None/0."""
    xs = [x for x in xs if x is not None]
    if not xs or any(x <= 0 for x in xs):
        return 0.0 if xs and any(x == 0 for x in xs) else None
    return len(xs) / sum(1.0 / x for x in xs)


def _tokenize_for_ngrams(text):
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [t for t in text.split() if t]


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def generation_entropy(text, weights = GE_NGRAM_WEIGHTS):
    """
    Weighted Shannon entropy over n-gram distributions of `text`.
    Low GE = repetitive/degenerate generation.
    """
    toks = _tokenize_for_ngrams(text)
    out = 0.0
    for n, w in weights.items():
        grams = _ngrams(toks, n)
        if not grams:
            continue
        cnt = Counter(grams)
        total = sum(cnt.values())
        ent = -sum((c / total) * math.log(c / total) for c in cnt.values())
        out += w * ent
    return out


# Reference Score (RS) -- TF-IDF cosine similarity                            #

_TFIDF_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tfidf_vector(text):
    toks = _TFIDF_TOKEN_RE.findall(text.lower())
    if not toks:
        return {}
    cnt = Counter(toks)
    total = sum(cnt.values())
    return {w: c / total for w, c in cnt.items()}


def tfidf_cosine(a, b):
    va, vb = _tfidf_vector(a), _tfidf_vector(b)
    if not va or not vb:
        return 0.0
    keys = set(va) | set(vb)
    dot = sum(va.get(k, 0.0) * vb.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    return dot / (na * nb) if (na and nb) else 0.0

def _rag_acc(d_correct, d_total, m):
    return d_correct[m] / d_total[m] if d_total[m] else None


def eval_one_condition(cond_name, edits, bge, reader, index_dir, k, run_generation, cfg):
    idx, idx_case_ids, idx_doc_texts = load_index_with_meta(index_dir, cond_name)
    idx_case_ids_list = list(int(c) for c in idx_case_ids)

    # ES / PS / NS counters
    correct = defaultdict(int)
    total = defaultdict(int)
    # Retrieval counters
    retrieval_hits = defaultdict(int)
    retrieval_total = defaultdict(int)
    # Generation metrics
    rs_values = []
    ge_values = []

    for ei, e in enumerate(tqdm(edits, desc=cond_name, unit="edit", leave=False)):
        cid = int(e["case_id"])
        prompt_canonical = e["actual_prompt"]

        try:
            edit_row = idx_case_ids_list.index(cid)
        except ValueError:
            edit_row = None

        # get all the prompts
        jobs = [("efficacy", prompt_canonical)]
        for p in (e.get(cfg["paraphrase_field"]) or []):
            jobs.append(("paraphrase", p))
        for p in (e.get(cfg["neighborhood_field"]) or []):
            jobs.append(("neighborhood", p))

        for metric, scoring_prompt in jobs:
            q_vec = bge.encode_queries([scoring_prompt]).astype(np.float32)
            _, I = idx.search(q_vec, k=k)
            retrieved_rows = [int(r) for r in I[0]]
            retrieved_texts = [idx_doc_texts[r] for r in retrieved_rows]
            context = "\n".join(retrieved_texts)

            # retrieval recall@k 
            if edit_row is not None:
                if metric == "neighborhood":
                    hit = (edit_row not in retrieved_rows)
                else:
                    hit = (edit_row in retrieved_rows)
                retrieval_hits[metric] += int(hit)
                retrieval_total[metric] += 1

            # modelscoring
            success = _score_one(reader, cfg, e, metric, scoring_prompt, context)
            if success is None:
                continue
            correct[metric] += int(success)
            total[metric] += 1

        # Generation metrics
        if run_generation:
            gen_prompts = e.get("generation_prompts") or []
            if gen_prompts:
                gp = gen_prompts[0]
                q_vec = bge.encode_queries([gp]).astype(np.float32)
                _, I = idx.search(q_vec, k=k)
                ctx = "\n".join(idx_doc_texts[int(r)] for r in I[0])
                gen = reader.generate(prompt=gp, context=ctx, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=False)
                ref_text = e.get("edit_documents", {}).get("encyclopedic", {}).get("text") or ""
                rs_values.append(tfidf_cosine(gen, ref_text))
                ge_values.append(generation_entropy(gen))


    es = _rag_acc(correct, total, "efficacy")
    ps = _rag_acc(correct, total, "paraphrase")
    ns = _rag_acc(correct, total, "neighborhood")
    score = harmonic_mean(es, ps, ns)
    rs = (sum(rs_values)/len(rs_values)) if rs_values else None
    ge = (sum(ge_values)/len(ge_values)) if ge_values else None

    summary = {
        "condition": cond_name,
        "n_edits": len(edits),
        "metrics": {
            "efficacy_success":     {"correct": correct["efficacy"],
                                     "total": total["efficacy"], "accuracy": es},
            "paraphrase_success":   {"correct": correct["paraphrase"],
                                     "total": total["paraphrase"], "accuracy": ps},
            "neighborhood_success": {"correct": correct["neighborhood"],
                                     "total": total["neighborhood"], "accuracy": ns},
            "editing_score":        score,
            "retrieval_recall_at_k": {
                m: {"hits": retrieval_hits[m], "total": retrieval_total[m],
                    "accuracy": (retrieval_hits[m] / retrieval_total[m]
                                 if retrieval_total[m] else None)}
                for m in ("efficacy", "paraphrase", "neighborhood")
            },
            "reference_score":      rs,
            "generation_entropy":   ge,
        },
        "config": {"k": k, "run_generation": run_generation},
    }
    return summary

def _score_one(reader, cfg, e, metric, scoring_prompt, context):
    """
    Returns True/False for success, or None if the case should be skipped.
    """
    target_new = e["new_target"]
    target_old = e["true_target"]

    if cfg["scoring"] == "comparative":
        lp_new, lp_old = reader.score_target_new(prompt=scoring_prompt, target_new=target_new, target_old=target_old, context=context)
        return (lp_new > lp_old) if metric in ("efficacy", "paraphrase") else (lp_old > lp_new)

    if cfg["scoring"] == "argmax":
        q = scoring_prompt
        if q.startswith("nq question: "):
            q = q[len("nq question: "):]
        q = q.strip().rstrip("?") + "?"
        scoring_prompt = f"Question: {q}\nAnswer:"
        if metric == "neighborhood":
            tgt = e.get("loc_ans")
            if tgt is None:
                return None
        else:
            tgt = target_new
        return reader.argmax_matches(prompt=scoring_prompt, target=tgt, context=context)

def _baseline_acc(m, correct, total): 
    return correct[m] / total[m] if total[m] else None

def eval_unedited_baseline(edits, reader, run_generation, cfg):
    # baseline w/ no rag (--skip-baseline to not run)
    correct = defaultdict(int)
    total = defaultdict(int)
    rs_values = []
    ge_values = []

    for ei, e in enumerate(tqdm(edits, desc="unedited_baseline", unit="edit", leave=False)):
        jobs = [("efficacy", e["actual_prompt"])]
        for p in (e.get(cfg["paraphrase_field"]) or []):
            jobs.append(("paraphrase", p))
        for p in (e.get(cfg["neighborhood_field"]) or []):
            jobs.append(("neighborhood", p))

        for metric, sp in jobs:
            success = _score_one(reader, cfg, e, metric, sp, context=None)
            if success is None:
                continue
            correct[metric] += int(success)
            total[metric] += 1

        if run_generation:
            gen_prompts = e.get("generation_prompts") or []
            if gen_prompts:
                gen = reader.generate(prompt=gen_prompts[0], context=None, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=False)
                ref_text = e.get("edit_documents", {}).get("encyclopedic", {}).get("text") or ""
                rs_values.append(tfidf_cosine(gen, ref_text))
                ge_values.append(generation_entropy(gen))

        if (ei + 1) % 100 == 0:
            print(f"    [unedited] {ei+1}/{len(edits)}")

    
    es = _baseline_acc("efficacy")
    ps = _baseline_acc("paraphrase")
    ns = _baseline_acc("neighborhood")
    score = harmonic_mean(es, ps, ns)
    rs = (sum(rs_values)/len(rs_values)) if rs_values else None
    ge = (sum(ge_values)/len(ge_values)) if ge_values else None
    return {
        "condition": "unedited_baseline",
        "n_edits": len(edits),
        "metrics": {
            "efficacy_success":     {"correct": correct["efficacy"],
                                     "total": total["efficacy"], "accuracy": es},
            "paraphrase_success":   {"correct": correct["paraphrase"],
                                     "total": total["paraphrase"], "accuracy": ps},
            "neighborhood_success": {"correct": correct["neighborhood"],
                                     "total": total["neighborhood"], "accuracy": ns},
            "editing_score":        score,
            "reference_score":      rs,
            "generation_entropy":   ge,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-generation", action="store_true",
                    help="skip RS and GE (saves ~50% wall time)")
    ap.add_argument("--noisy", action='store_true')
    ap.add_argument("--conditions", nargs="*", default=None)
    args = ap.parse_args()
    print("args:", args)

    cfg = DATASETS[args.dataset]
    edits_path = cfg['edits_path']
    index_dir = cfg['index_dir_noisy'] if args.noisy else cfg['index_dir_clean']
    run_generation = (not args.skip_generation) and cfg["has_generation_prompts"]
    output_path = Path(f"./eval_results_{args.dataset}_{'noisy' if args.noisy else 'clean'}.json")

    edits = [r for r in iter_edits(edits_path) if not (set(r.get("flags", [])) & SKIP_FLAGS)]
    print(f"  {len(edits)} usable edits after flag filtering")
    if args.limit:
        edits = edits[:args.limit]
        print(f"  --limit {args.limit}: using {len(edits)} edits")

    print("loading BGE retriever...")
    bge = BGERetriever()
    print("loading GPT-J reader...")
    reader = GPTJReader()


    conditions = list_conditions(index_dir)
    if args.conditions:
        conditions = [c for c in conditions if c in set(args.conditions)]
    print(f"running {len(conditions)} condition(s): {conditions}")

    results = {
        "conditions": {},
        "config": {"k": args.k, "n_edits": len(edits),
                   "edits_path": str(edits_path), "index_dir": str(index_dir),
                   "corpus_type": "noisy" if args.noisy else "clean",
                   "run_generation": not args.skip_generation},
    }

    if not args.skip_baseline:
        print("\n[ baseline: unedited GPT-J ]")
        results["conditions"]["unedited_baseline"] = eval_unedited_baseline(edits, reader, run_generation=run_generation, cfg=cfg)
        _dump(results, output_path)

    for cond in conditions:
        print(f"\n[ {cond} ]")
        results["conditions"][cond] = eval_one_condition(cond_name=cond, edits=edits, bge=bge, reader=reader, index_dir=index_dir, k=args.k, run_generation=run_generation, cfg=cfg)
        _dump(results, output_path)


    hdr = f"{'condition':<35} {'ES':>6} {'PS':>6} {'NS':>6} {'S':>6} {'R@k(E)':>7} {'R@k(P)':>7} {'R@k(N)':>7} {'RS':>6} {'GE':>6}"
    print(hdr)
    print("-" * len(hdr))
    for name, res in results["conditions"].items():
        m = res["metrics"]
        def f3(v): return f"{v:.3f}" if isinstance(v, (int, float)) else "  -  "
        es = m["efficacy_success"]["accuracy"]
        ps = m["paraphrase_success"]["accuracy"]
        ns = m["neighborhood_success"]["accuracy"]
        s = m.get("editing_score")
        rk = m.get("retrieval_recall_at_k", {})
        rk_e = rk.get("efficacy", {}).get("accuracy") if rk else None
        rk_p = rk.get("paraphrase", {}).get("accuracy") if rk else None
        rk_n = rk.get("neighborhood", {}).get("accuracy") if rk else None
        rs = m.get("reference_score")
        ge = m.get("generation_entropy")
        print(f"{name:<35} {f3(es):>6} {f3(ps):>6} {f3(ns):>6} {f3(s):>6} "
              f"{f3(rk_e):>7} {f3(rk_p):>7} {f3(rk_n):>7} {f3(rs):>6} {f3(ge):>6}")


def _dump(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=float)


if __name__ == "__main__":
    main()