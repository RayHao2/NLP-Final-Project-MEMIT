"""
Estimate C_q -- the natural-query covariance matrix.

C_q captures the
distribution of "what natural queries look like" in embedding space.

    e* proportional to  (C_q + alpha * I)^-1 @ s_e

where s_e is the sum of edit-query embeddings. 

    C_q = (1/N) * sum_i  q_i  q_i^T

where q_i are L2-normalized query embeddings of natural queries.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bge_retriever import BGERetriever, EMBED_DIM


DEFAULT_N = 50_000

def _stream_msmarco_queries(n_target):
    from datasets import load_dataset

    print(f"streaming MS MARCO queries from HF (target: {n_target})...")
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="train", streaming=True)

    seen = set()
    out = []
    for row in ds:
        q = (row.get("query") or "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(q)
        if len(out) >= n_target:
            break
    print(f"  collected {len(out)} unique queries")
    return out

def compute_cq_streaming(bge, queries, chunk_size= 4096):
    """
    Compute C_q = (1/N) sum q_i q_i^T by accumulating outer-product sums in
    chunks.
    """
    D = bge.embed_dim
    accum = np.zeros((D, D), dtype=np.float64)
    n_used = 0

    t0 = time.time()
    for i in range(0, len(queries), chunk_size):
        batch = queries[i : i + chunk_size]
        # encode_queries handles the BGE prefix and L2-normalization
        Q = bge.encode_queries(batch).astype(np.float64)   # (B, D)
        accum += Q.T @ Q
        n_used += Q.shape[0]

        if (i // chunk_size) % 5 == 0:
            elapsed = time.time() - t0
            rate = n_used / max(elapsed, 1e-9)
            eta = (len(queries) - n_used) / max(rate, 1e-9)
            print(f"  {n_used:>7}/{len(queries)}  ({rate:7.1f} q/s, eta {eta:6.1f}s)")


    C_q = (accum / n_used).astype(np.float32)
    return C_q, n_used

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="number of natural queries to encode")
    ap.add_argument("--output", type=Path, required=True,
                    help="output .npz path (e.g. retrieval/cq.npz)")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bs = 64
    cs = 4096

    queries = _stream_msmarco_queries(args.n)
    source_desc = "ms_marco/v2.1/train"

    bge = BGERetriever(batch_size=bs)
    C_q, n_used = compute_cq_streaming(bge, queries, chunk_size=cs)

    eig = np.linalg.eigvalsh(C_q.astype(np.float64)) 
    eig_min, eig_max = float(eig.min()), float(eig.max())
    trace = float(np.trace(C_q))
    sym_err = float(np.max(np.abs(C_q - C_q.T)))
    print(f"  trace(C_q)       = {trace:.6f}   (expected ~= 1.0 for unit-norm vecs)")
    print(f"  symmetry max err = {sym_err:.2e}  (expected ~0)")
    print(f"  eigenvalue range = [{eig_min:.3e}, {eig_max:.3e}]")

    np.savez(args.output, C_q=C_q, N=np.int64(n_used), embed_dim=np.int64(EMBED_DIM), model_name=np.array("BAAI/bge-base-en-v1.5"), source=np.array(source_desc))
    print(f"saved -> {args.output}  ({C_q.nbytes / 1024:.1f} KB)")


if __name__ == "__main__":
    main()