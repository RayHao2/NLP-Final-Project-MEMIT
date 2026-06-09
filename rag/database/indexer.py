"""
Build FAISS indexes for every (format, method[, alpha]) combination.
3 formats (atomic, encyclopedic, corrective)
1 naive encoding
4 diff alpha values
"""

import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from bge_retriever import BGERetriever, EMBED_DIM



# EDITS_PATH  = Path("../edit_documents/cf.json")
EDITS_PATH  = Path("../edit_documents/zsre_eval.jsonl")

# ESTAR_PATH  = Path("./data/embeddings.npz")
ESTAR_PATH  = Path("./data/zsre_embeddings.npz")

# OUTPUT_DIR  = Path("./clean/indexes")
OUTPUT_DIR  = Path("./zsre/clean/indexes")

FORMATS = ("atomic", "encyclopedic", "corrective")
SKIP_FLAGS = {"missing_target_new"}     # same exclusion as e* build

BGE_BATCH_SIZE = 64



def iter_edits(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def alpha_tag(alpha):
    a = float(alpha)
    if a >= 1.0:
        return f"a{int(round(a))}"
    s = f"{a:.6f}".rstrip("0").rstrip(".")    # 0.01 -> "0.010000" -> "0.01"
    return "a" + s.replace(".", "p")


def save_index(name, vectors, case_ids, doc_texts):
    """Build a flat-IP FAISS index and save it with aligned metadata."""

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(np.ascontiguousarray(vectors))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(OUTPUT_DIR / f"{name}.faiss"))
    np.savez(OUTPUT_DIR / f"{name}.meta.npz", case_ids=case_ids, doc_texts=np.array(doc_texts, dtype=object), name=np.array(name))

def main():
    # Load e*
    print(f"loading e* from {ESTAR_PATH}")
    estar_data = np.load(ESTAR_PATH, allow_pickle=True)
    estar_case_ids = estar_data["case_ids"]     # (N_e,)
    estar = estar_data["e_star"]                 # (N_alpha, N_e, D)
    alphas = list(estar_data["alphas"])          # (N_alpha,)
    print(f"  e* shape={estar.shape}, alphas={alphas}")

    estar_row_by_cid = {int(cid): i for i, cid in enumerate(estar_case_ids)}

    # get edit metadata
    print(f"scanning {EDITS_PATH}")
    docs_by_format = {fmt: [] for fmt in FORMATS}
    skipped_by_flag = 0
    skipped_no_estar = 0
    skipped_no_doc = 0

    for rec in iter_edits(EDITS_PATH):
        flags = set(rec.get("flags", []))
        if flags & SKIP_FLAGS:
            skipped_by_flag += 1
            continue

        cid = rec.get("case_id")
        if int(cid) not in estar_row_by_cid:
            skipped_no_estar += 1
            continue

        edit_docs = rec.get("edit_documents") or {}
        for fmt in FORMATS:
            doc = edit_docs.get(fmt)
            if not doc or not (doc.get("text") or "").strip():
                skipped_no_doc += 1
                continue
            docs_by_format[fmt].append({"case_id": int(cid), "text": doc["text"]})

    for fmt in FORMATS:
        print(f"  {fmt:>13}: {len(docs_by_format[fmt])} docs")
    print(f"  excluded: {skipped_by_flag} by flag, "
          f"{skipped_no_estar} missing e*, {skipped_no_doc} missing format")

    # build the indexes for each format
    bge = BGERetriever(batch_size=BGE_BATCH_SIZE)
    t_total = time.time()

    for fmt in FORMATS:
        recs = docs_by_format[fmt]
        N = len(recs)
        case_ids_arr = np.array([r["case_id"] for r in recs], dtype=np.int64)
        doc_texts = [r["text"] for r in recs]

        # Naive
        t0 = time.time()
        print(f"[{fmt}] encoding {N} docs for naive index...")
        naive_vecs = bge.encode_docs(doc_texts).astype(np.float32)  # (N, D)

        norms = np.linalg.norm(naive_vecs, axis=1, keepdims=True)
        naive_vecs = (naive_vecs / norms).astype(np.float32)
        save_index(name=f"{fmt}.naive", vectors=naive_vecs, case_ids=case_ids_arr, doc_texts=doc_texts)
        print(f"  [{fmt}.naive] saved ({time.time()-t0:.1f}s)")

        # look up e* for each edit
        row_in_estar = np.array([estar_row_by_cid[c] for c in case_ids_arr], dtype=np.int64)

        for a_idx, alpha in enumerate(alphas):
            vecs_for_alpha = estar[a_idx, row_in_estar, :].astype(np.float32)
            norms = np.linalg.norm(vecs_for_alpha, axis=1, keepdims=True)

            vecs_for_alpha = (vecs_for_alpha / norms).astype(np.float32)
            name = f"{fmt}.method1.{alpha_tag(float(alpha))}"
            save_index(name=name, vectors=vecs_for_alpha, case_ids=case_ids_arr, doc_texts=doc_texts)
        print(f"  [{fmt}.method1.*] saved {len(alphas)} alpha variants")

    print(f"\nbuilt all indexes in {time.time()-t_total:.1f}s\n")




if __name__ == "__main__":
    main()