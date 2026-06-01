"""
Build FAISS indexes for the NOISY corpus (edit docs + 100k Wikipedia chunks).
3 formats (atomic, encyclopedic, corrective)
1 naive encoding
4 diff alpha values

The retrieved TEXT is either an edit document or a Wikipedia chunk
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

WIKI_DIR    = Path("../edit_documents/wiki")

# OUTPUT_DIR  = Path("./noisy/indexes_noisy")
OUTPUT_DIR  = Path("./zsre/noisy/indexes_noisy")

FORMATS = ("atomic", "encyclopedic", "corrective")
SKIP_FLAGS = {"missing_target_new"}

BGE_BATCH_SIZE = 64
WIKI_PLACEHOLDER_CID = -1                 # case_id for Wikipedia chunks


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
    s = f"{a:.6f}".rstrip("0").rstrip(".")
    return "a" + s.replace(".", "p")


def save_index(name, vectors, case_ids, doc_texts):
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(np.ascontiguousarray(vectors))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(OUTPUT_DIR / f"{name}.faiss"))
    np.savez(OUTPUT_DIR / f"{name}.meta.npz", case_ids=case_ids, doc_texts=np.array(doc_texts, dtype=object), name=np.array(name))


def load_wiki_cache():
    vec_path  = WIKI_DIR / "wiki_vectors.npy"
    text_path = WIKI_DIR / "wiki_texts.jsonl"
    if not (vec_path.exists() and text_path.exists()):
        raise RuntimeError(
            f"Wikipedia cache not found in {WIKI_DIR}. "
            f"Run prep_wikipedia.py first.")
    vecs = np.load(vec_path).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.clip(norms, 1e-9, None)
    vecs = vecs.astype(np.float32)

    texts = []
    with open(text_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            texts.append(json.loads(line)["text"])

    print(f"loaded {vecs.shape[0]} wikipedia chunks from {WIKI_DIR}")
    return vecs, texts


def main():
    # Load e* 
    print(f"loading e* from {ESTAR_PATH}")
    estar_data = np.load(ESTAR_PATH, allow_pickle=True)
    estar_case_ids = estar_data["case_ids"]
    estar = estar_data["e_star"]                  # (N_alpha, N_e, D)
    alphas = list(estar_data["alphas"])
    print(f"  e* shape={estar.shape}, alphas={alphas}")
    estar_row_by_cid = {int(cid): i for i, cid in enumerate(estar_case_ids)}

    # Load Wikipedia cache 
    wiki_vecs, wiki_texts = load_wiki_cache()
    n_wiki = wiki_vecs.shape[0]
    wiki_case_ids = np.full((n_wiki,), WIKI_PLACEHOLDER_CID, dtype=np.int64)

    #  edits metadata
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
        cid = int(rec["case_id"])
        if cid not in estar_row_by_cid:
            skipped_no_estar += 1
            continue
        edit_docs = rec.get("edit_documents") or {}
        for fmt in FORMATS:
            doc = edit_docs.get(fmt)
            if not doc or not (doc.get("text") or "").strip():
                skipped_no_doc += 1
                continue
            docs_by_format[fmt].append({"case_id": cid, "text": doc["text"]})

    for fmt in FORMATS:
        print(f"  {fmt:>13}: {len(docs_by_format[fmt])} edit docs")
    print(f"  excluded: {skipped_by_flag} by flag, "
          f"{skipped_no_estar} missing e*, {skipped_no_doc} missing format")
    print(f"  each index will be {len(docs_by_format[FORMATS[0]]) + n_wiki} rows total "
          f"(edit docs + wiki)")

    # build indexes for all formats
    bge = BGERetriever(batch_size=BGE_BATCH_SIZE)
    t_total = time.time()

    for fmt in FORMATS:
        recs = docs_by_format[fmt]
        N_edits = len(recs)
        edit_case_ids = np.array([r["case_id"] for r in recs], dtype=np.int64)
        edit_texts = [r["text"] for r in recs]

        #  Naive
        t0 = time.time()
        print(f"[{fmt}] encoding {N_edits} edit docs for naive...")
        naive_edit_vecs = bge.encode_docs(edit_texts).astype(np.float32)
        norms = np.linalg.norm(naive_edit_vecs, axis=1, keepdims=True)
        naive_edit_vecs = (naive_edit_vecs / np.clip(norms, 1e-9, None)).astype(np.float32)
        print(f"  encoded in {time.time()-t0:.1f}s")

        # Concatenate: edits FIRST, then wiki
        naive_vecs = np.vstack([naive_edit_vecs, wiki_vecs]).astype(np.float32)
        all_case_ids = np.concatenate([edit_case_ids, wiki_case_ids])
        all_texts = edit_texts + wiki_texts

        save_index(name=f"{fmt}.naive", vectors=naive_vecs, case_ids=all_case_ids, doc_texts=all_texts)
        print(f"  [{fmt}.naive] saved ({naive_vecs.shape[0]} rows)")

        # e* + wiki vectors
        row_in_estar = np.array([estar_row_by_cid[c] for c in edit_case_ids], dtype=np.int64)

        for a_idx, alpha in enumerate(alphas):
            estar_edit_vecs = estar[a_idx, row_in_estar, :].astype(np.float32)
            norms = np.linalg.norm(estar_edit_vecs, axis=1, keepdims=True)
            estar_edit_vecs = (estar_edit_vecs / np.clip(norms, 1e-9, None)).astype(np.float32)

            method1_vecs = np.vstack([estar_edit_vecs, wiki_vecs]).astype(np.float32)
            name = f"{fmt}.method1.{alpha_tag(float(alpha))}"
            save_index(name=name, vectors=method1_vecs, case_ids=all_case_ids, doc_texts=all_texts)
        print(f"  [{fmt}.method1.*] saved {len(alphas)} alpha variants")

    print(f"\nbuilt all noisy indexes in {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()