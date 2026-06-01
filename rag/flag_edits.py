from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from collections import Counter


# --------------------------------------------------------------------------- #
# Heuristics                                                                  #
# --------------------------------------------------------------------------- #

REQUIRED_FORMATS = ("atomic", "encyclopedic", "corrective")

FORBIDDEN_META = {"edit", "update", "revised", "correction"}

PROBLEMATIC_TRAILING_TOKENS = {"the", "a", "an"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _contains(text: str, needle: str) -> bool:
    return _norm(needle) in _norm(text)


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE) is not None



def flag_edit(rec: dict) -> list[str]:
    """Return a list of flag strings for one edit record."""
    flags: list[str] = []
    target_new = rec.get("new_target", "") or ""
    target_old = rec.get("true_target", "") or ""
    actual_prompt = (rec.get("actual_prompt", "") or "").strip()
    docs = rec.get("edit_documents", {}) or {}

    # --- structural checks ---
    missing = [f for f in REQUIRED_FORMATS if f not in docs]
    if missing:
        flags.append("missing_formats")

    for fmt, doc in docs.items():
        text = (doc.get("text") if isinstance(doc, dict) else "") or ""

        if "<think>" in text or "</think>" in text:
            flags.append("has_think_block")

        if len(text) < 60:
            flags.append("doc_too_short")
        if len(text) > 1500:
            flags.append("doc_too_long")

        if target_new and not _contains(text, target_new):
            flags.append("missing_target_new")

        if fmt != "corrective" and target_old and _contains(text, target_old):
            subject = rec.get("subject", "") or ""
            if not (subject and _contains(subject, target_old)):
                if _has_word(text, target_old.split()[0]):
                    flags.append("leaked_target_old")

        if fmt != "corrective":
            for w in FORBIDDEN_META:
                if _has_word(text, w):
                    flags.append("forbidden_meta_word")
                    break

    # --- CounterFact dataset pathologies ---
    nt, ot = _norm(target_new), _norm(target_old)
    if nt and ot:
        if nt == ot:
            flags.append("synonymous_targets")
        elif nt in ot or ot in nt:
            # e.g. 'New York' vs 'New York City'
            flags.append("subword_target_overlap")

    if actual_prompt:
        last_token = re.sub(r"[^a-zA-Z]", "", actual_prompt.split()[-1]).lower()
        if last_token in PROBLEMATIC_TRAILING_TOKENS:
            flags.append("fragmentary_prompt")
        if actual_prompt.rstrip().endswith(","):
            flags.append("fragmentary_prompt")

    corrective = docs.get("corrective", {})
    if isinstance(corrective, dict):
        ctext = corrective.get("text", "") or ""
        if ctext:
            has_incorrect = _has_word(ctext, "incorrect")
            quotes_prompt = actual_prompt and actual_prompt in ctext
            mentions_old = target_old and _contains(ctext, target_old)
            if has_incorrect and quotes_prompt and not mentions_old:
                flags.append("corrective_quotes_prompt")

    seen = set()
    out = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def run(input_path: Path, output_path: Path, report_path: Path) -> None:
    flag_counts: Counter[str] = Counter()
    n_total = 0
    n_clean = 0
    sample_per_flag: dict[str, list[int]] = {}    

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            flags = flag_edit(rec)
            rec["flags"] = flags
            fout.write(json.dumps(rec) + "\n")

            n_total += 1
            if not flags:
                n_clean += 1
            for fl in flags:
                flag_counts[fl] += 1
                samples = sample_per_flag.setdefault(fl, [])
                if len(samples) < 5:
                    samples.append(rec.get("case_id"))

    report = {
        "total_edits":   n_total,
        "clean_edits":   n_clean,
        "clean_rate":    (n_clean / n_total) if n_total else 0.0,
        "flag_counts":   dict(flag_counts.most_common()),
        "sample_cases":  sample_per_flag,
    }
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL produced by generate_all")
    ap.add_argument("--output", required=True, help="JSONL with `flags` field added")
    ap.add_argument("--report", required=True, help="Aggregate report JSON")
    args = ap.parse_args()
    run(Path(args.input), Path(args.output), Path(args.report))


if __name__ == "__main__":
    main()