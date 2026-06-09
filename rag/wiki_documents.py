"""
Download a Wikipedia subset, chunk into BGE-friendly passages, encode all
chunks with the BGE doc encoder
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))
from database.bge_retriever import BGERetriever, EMBED_DIM


# --------------------------------------------------------------------------- #
# Hardcoded configuration                                                     #
# --------------------------------------------------------------------------- #

WIKI_DATASET    = "wikimedia/wikipedia"
WIKI_CONFIG     = "20231101.en"
N_TARGET        = 100_000                       # number of chunks
CHUNK_TOKENS    = 256                           # passage size (approx)
CHUNK_OVERLAP   = 32                            # overlap between consecutive chunks
MIN_CHUNK_CHARS = 200                           # skip near-empty chunks

OUTPUT_DIR     = Path("./edit_documents/wiki")
VECTORS_PATH   = OUTPUT_DIR / "wiki_vectors.npy"
TEXTS_PATH     = OUTPUT_DIR / "wiki_texts.jsonl"
META_PATH      = OUTPUT_DIR / "wiki_meta.json"

BGE_BATCH_SIZE = 64
ENCODE_FLUSH_EVERY = 5_000                      # save partial vectors every N chunks


# --------------------------------------------------------------------------- #
# Chunking                                                                    #
# --------------------------------------------------------------------------- #

def _approx_token_chunks(
    text: str,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Approximate token-based chunking by whitespace word count. ~1 token per
    ~0.75 words in English BPE, so CHUNK_TOKENS=256 -> ~190 words. We just
    use a word count of ~CHUNK_TOKENS as a proxy; BGE's tokenizer can handle
    the resulting passages well under its 512 max_length.

    Why not tokenize properly: tokenizing all of Wikipedia just to chunk it
    is 10x slower than the encoding pass we're about to do anyway. Word-count
    chunking is good enough for this experiment.
    """
    words = text.split()
    if not words:
        return []
    out: list[str] = []
    step = chunk_tokens - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_tokens])
        if len(chunk) >= MIN_CHUNK_CHARS:
            out.append(chunk)
        if i + chunk_tokens >= len(words):
            break
    return out


# --------------------------------------------------------------------------- #
# Streaming + chunking                                                        #
# --------------------------------------------------------------------------- #

def stream_wiki_chunks(n_target: int):
    """
    Yield text chunks from streaming Wikipedia until n_target chunks emitted.
    Order is article order in the dataset shard, which is effectively random
    for Wikipedia's HF release. Good enough.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("pip install datasets")

    print(f"streaming {WIKI_DATASET}:{WIKI_CONFIG} ...")
    ds = load_dataset(WIKI_DATASET, WIKI_CONFIG, split="train", streaming=True)

    emitted = 0
    seen_articles = 0
    t0 = time.time()
    for row in ds:
        text = row.get("text") or ""
        seen_articles += 1
        for chunk in _approx_token_chunks(text):
            yield chunk
            emitted += 1
            if emitted >= n_target:
                elapsed = time.time() - t0
                print(f"  reached {emitted} chunks from {seen_articles} articles "
                      f"in {elapsed:.0f}s")
                return
        if seen_articles % 500 == 0:
            elapsed = time.time() - t0
            rate = emitted / max(elapsed, 1e-6)
            print(f"  articles={seen_articles}  chunks={emitted}  "
                  f"({rate:.0f} chunks/s)")


# --------------------------------------------------------------------------- #
# Cache logic                                                                 #
# --------------------------------------------------------------------------- #

def _is_cache_complete(n_target: int) -> bool:
    if not (VECTORS_PATH.exists() and TEXTS_PATH.exists() and META_PATH.exists()):
        return False
    try:
        meta = json.loads(META_PATH.read_text())
        if int(meta.get("n_chunks", 0)) < n_target:
            return False
        # Sanity: vectors file shape matches meta
        arr = np.load(VECTORS_PATH, mmap_mode="r")
        if arr.shape[0] < n_target or arr.shape[1] != EMBED_DIM:
            return False
        return True
    except Exception:
        return False


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if _is_cache_complete(N_TARGET):
        print(f"cache already complete at {OUTPUT_DIR} (n>={N_TARGET}). nothing to do.")
        return

    bge = BGERetriever(batch_size=BGE_BATCH_SIZE)

    # We allocate the full output array upfront, fill it as we go, then truncate
    # to the actual count if streaming yielded fewer chunks than requested.
    all_vecs = np.zeros((N_TARGET, EMBED_DIM), dtype=np.float32)
    all_texts: list[str] = []

    # Streaming iterator -> buffered into batches of BGE_BATCH_SIZE to keep the
    # GPU saturated. We DON'T materialize all chunks before encoding.
    buf_texts: list[str] = []
    filled = 0
    t0 = time.time()

    def flush_buffer():
        nonlocal filled
        if not buf_texts:
            return
        vecs = bge.encode_docs(buf_texts).astype(np.float32)   # (B, D)
        # bge.encode_docs() L2-normalizes in model dtype then casts; renormalize
        # to fp32 unit-norm at the storage boundary (same fix as build_indexes)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.clip(norms, 1e-9, None)
        n = min(len(buf_texts), N_TARGET - filled)
        all_vecs[filled : filled + n] = vecs[:n].astype(np.float32)
        all_texts.extend(buf_texts[:n])
        filled += n
        buf_texts.clear()

    for chunk in stream_wiki_chunks(N_TARGET):
        buf_texts.append(chunk)
        if len(buf_texts) >= BGE_BATCH_SIZE:
            flush_buffer()
            if filled % ENCODE_FLUSH_EVERY < BGE_BATCH_SIZE:
                elapsed = time.time() - t0
                rate = filled / max(elapsed, 1e-6)
                eta = (N_TARGET - filled) / max(rate, 1e-6)
                print(f"  encoded {filled:>7}/{N_TARGET}  "
                      f"({rate:.0f} chunks/s, eta {eta:.0f}s)")
            if filled >= N_TARGET:
                break

    # final partial batch
    flush_buffer()

    # Truncate (in the rare case streaming ran out before reaching N_TARGET)
    if filled < N_TARGET:
        print(f"  warning: only got {filled} chunks (wanted {N_TARGET})")
        all_vecs = all_vecs[:filled]

    print(f"saving {filled} vectors + texts to {OUTPUT_DIR}")
    np.save(VECTORS_PATH, all_vecs)
    with open(TEXTS_PATH, "w") as fh:
        for t in all_texts:
            fh.write(json.dumps({"text": t}) + "\n")
    META_PATH.write_text(json.dumps({
        "n_chunks":        filled,
        "embed_dim":       EMBED_DIM,
        "model_name":      "BAAI/bge-base-en-v1.5",
        "source_dataset":  f"{WIKI_DATASET}:{WIKI_CONFIG}",
        "chunk_tokens":    CHUNK_TOKENS,
        "chunk_overlap":   CHUNK_OVERLAP,
        "min_chunk_chars": MIN_CHUNK_CHARS,
    }, indent=2))
    elapsed = time.time() - t0
    print(f"done in {elapsed:.0f}s")


if __name__ == "__main__":
    main()