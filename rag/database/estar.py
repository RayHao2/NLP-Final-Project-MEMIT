"""
Build e* for every edit.


e* proportional to  (C_q + alpha * I)^-1 @ s_e
then L2-normalize

where:
    s_e = BGE query embedding of the canonical prompt
    C_q = natural-query covariance (estimated by estimate_cq.py)
    alpha = scalar regularizer trading efficacy vs specificity

"""

import json
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent))

from bge_retriever import BGERetriever, EMBED_DIM


# EDITS_PATH  = Path("..//edit_documents/cf.json") #counterfact
EDITS_PATH  = Path("..//edit_documents/zsre_eval.jsonl") #zsre

CQ_PATH     = Path("./data/cq.npz")
OUTPUT_PATH = Path("./data/zsre_embeddings.npz")

ALPHAS = (0.001, 0.01, 0.1, 1.0)
SKIP_FLAGS = {"missing_target_new"}   # exclude edits whose docs don't assert target_new

BGE_BATCH_SIZE = 64
FLUSH_EVERY = 1000


def make_solvers(C_q, alphas):
    """
    Precompute the  (C_q + alpha I) for each alpha, returning a
    solver function s_e -> (C_q + alpha I)^-1 @ s_e per alpha.
    """
    from scipy.linalg import cho_factor, cho_solve

    D = C_q.shape[0]
    I = np.eye(D, dtype=np.float64)
    solvers = {}
    for alpha in alphas:
        M = (C_q.astype(np.float64) + alpha * I)
        c, low = cho_factor(M, lower=True)
        def _solve(s_e, c=c, low=low):
            return cho_solve((c, low), s_e.astype(np.float64))
        solvers[alpha] = _solve
    return solvers


def compute_e_star_for_alphas(s_e, solvers):
    """Given s_e and pre-factored solvers, return {alpha: e_star_unit_vec}."""
    out = {}
    for alpha, solve_fn in solvers.items():
        e = solve_fn(s_e)
        n = np.linalg.norm(e)
        if n < 1e-12:
            ns = np.linalg.norm(s_e)
            e = (s_e / ns) if ns > 1e-12 else s_e
        else:
            e = e / n
        out[alpha] = e.astype(np.float32)
    return out


def iter_edits(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load C_q 
    print(f"loading C_q from {CQ_PATH}")
    cq_data = np.load(CQ_PATH, allow_pickle=True)
    C_q = cq_data["C_q"]
    cq_N = int(cq_data["N"])
    print(f"  C_q shape={C_q.shape}, estimated from N={cq_N}")

    #  solvers
    alphas = sorted(ALPHAS)
    print(f"factoring (C_q + alpha I) for alphas={alphas}")
    solvers = make_solvers(C_q, alphas)

    #  collect edits
    print(f"scanning {EDITS_PATH}")
    usable = []
    skipped = []
    no_canonical = []
    for rec in iter_edits(EDITS_PATH):
        flags = set(rec.get("flags", []))
        if flags & SKIP_FLAGS:
            skipped.append(rec.get("case_id"))
            continue
        canonical = (rec.get("actual_prompt") or "").strip()
        if not canonical:
            no_canonical.append(rec.get("case_id"))
            continue
        usable.append({"case_id": rec["case_id"], "text": canonical})

    print(f"  {len(usable)} usable edits, "
          f"{len(skipped)} skipped by flag, "
          f"{len(no_canonical)} skippe")


    # Allocate outputs 
    N = len(usable)
    case_ids = np.array([r["case_id"] for r in usable], dtype=np.int64)
    e_star = np.zeros((len(alphas), N, EMBED_DIM), dtype=np.float32)

    # Encode in batches
    bge = BGERetriever(batch_size=BGE_BATCH_SIZE)
    t0 = time.time()

    texts_all = [r["text"] for r in usable]
    for batch_start in range(0, N, BGE_BATCH_SIZE):
        batch = texts_all[batch_start : batch_start + BGE_BATCH_SIZE]
        Q = bge.encode_queries(batch).astype(np.float64)   # (B, D), unit norm
        for j, s_e in enumerate(Q):
            per_alpha = compute_e_star_for_alphas(s_e, solvers)
            for a_idx, alpha in enumerate(alphas):
                e_star[a_idx, batch_start + j, :] = per_alpha[alpha]

        done = min(batch_start + BGE_BATCH_SIZE, N)
        if done % FLUSH_EVERY < BGE_BATCH_SIZE or done == N:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-9)
            eta = (N - done) / max(rate, 1e-9)
            print(f"  {done:>6}/{N}  ({rate:6.1f} edits/s, eta {eta:6.0f}s)")

    np.savez(OUTPUT_PATH, case_ids=case_ids, e_star=e_star, alphas=np.array(alphas, dtype=np.float32), embed_dim=np.int64(EMBED_DIM), model_name=np.array("BAAI/bge-base-en-v1.5"), cq_N=np.int64(cq_N), skipped=np.array(skipped, dtype=np.int64), no_canonical=np.array(no_canonical, dtype=np.int64))
    size_mb = (e_star.nbytes + case_ids.nbytes) / (1024 * 1024)
    print(f"saved {N} edits x {len(alphas)} alphas -> {OUTPUT_PATH} ({size_mb:.1f} MB)")

    # Self-similarity diagnostic 
    print("\nSelf-similarity check (e* should rank its own canonical query high):")
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(N, size=min(200, N), replace=False)
    canonical_texts = [usable[i]["text"] for i in sample_idx]
    Q_canon = bge.encode_queries(canonical_texts)              # (S, D)

    for a_idx, alpha in enumerate(alphas):
        e_sample = e_star[a_idx, sample_idx, :]
        cos = (Q_canon * e_sample).sum(axis=1)
        print(f"  alpha={alpha:<7g}  cos(e*, canonical_q): "
              f"mean={cos.mean():.3f}  median={np.median(cos):.3f}  "
              f"min={cos.min():.3f}  max={cos.max():.3f}")


if __name__ == "__main__":
    main()