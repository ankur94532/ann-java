#!/usr/bin/env python3
"""Why is the Java HNSW faster than FAISS's at every configuration?

Beating FAISS on both recall and latency at all 36 swept configurations is not a credible
outcome for a from-scratch implementation, so the question is which part of the comparison
is unfair. Per-query latency cannot answer it, because it conflates "each distance costs
more" with "we compute more distances".

`faiss.cvar.hnsw_stats.ndis` counts distance evaluations inside HNSW's own search - the
DistanceComputer path, not the blocked routine IndexFlatL2 uses - so dividing latency by it
gives nanoseconds per distance on both sides. That splits the hypotheses cleanly:

  same ndis, FAISS slower per distance   -> kernel / virtual-dispatch quality on aarch64.
                                            The narrow claim is right.
  FAISS more ndis at the same efSearch   -> not a kernel story. The graphs differ despite
                                            matching M and efConstruction, and the recall
                                            edge is the same fact seen from the other side.
  FAISS fewer ndis and still slower      -> the strongest form of the kernel claim.

`nhops` is reported alongside: distances per hop is the effective degree the search walks,
which is a direct read on whether the two graphs have comparable connectivity.

Usage:
    ./.venv/bin/python scripts/hnsw_ndis_compare.py --m 16 --efc 200
"""
import argparse
import os
import time

import numpy as np
import faiss

# Measured by the Java harness at M=16, efConstruction=200 on SIFT1M, full query set.
# Printed by: ./gradlew run --args="hnsw --m 16 --efc 200 --ef ..."  (distances/query)
JAVA_M16_EFC200 = {
    16: (518, 32.7), 32: (797, 53.2), 64: (1323, 92.0),
    128: (2287, 164.7), 256: (3995, 295.7), 512: (6942, 533.2),
}


def read_fvecs(path, limit=None):
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    raw = raw.reshape(-1, dim + 1)
    if limit is not None:
        raw = raw[:limit]
    return np.ascontiguousarray(raw[:, 1:]).view(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--efc", type=int, default=200)
    parser.add_argument("--ef", default="16,32,64,128,256,512")
    parser.add_argument("--queries", type=int, default=10000)
    parser.add_argument("--data", default="data/sift")
    args = parser.parse_args()
    efs = [int(x) for x in args.ef.split(",")]

    faiss.omp_set_num_threads(1)
    base = read_fvecs(os.path.join(args.data, "sift_base.fvecs"))
    queries = read_fvecs(os.path.join(args.data, "sift_query.fvecs"), args.queries)
    print(f"faiss {faiss.__version__}, 1 thread, {base.shape[0]:,} x {base.shape[1]}, "
          f"{queries.shape[0]:,} queries")

    index = faiss.IndexHNSWFlat(base.shape[1], args.m, faiss.METRIC_L2)
    index.hnsw.efConstruction = args.efc
    t0 = time.perf_counter()
    index.add(base)
    print(f"built M={args.m} efC={args.efc} in {time.perf_counter() - t0:.1f}s\n")

    java = JAVA_M16_EFC200 if (args.m, args.efc) == (16, 200) else {}
    print(f"{'ef':>5} {'faiss ndis/q':>13} {'nhops/q':>9} {'dis/hop':>8} "
          f"{'faiss us':>9} {'ns/dis':>8} | {'java ndis/q':>12} {'java us':>8} {'ns/dis':>8}")

    for ef in efs:
        index.hnsw.efSearch = ef
        assert index.hnsw.efSearch == ef, "efSearch did not take"
        for i in range(min(1000, queries.shape[0])):      # warm-up, discarded
            index.search(queries[i:i + 1], 10)

        faiss.cvar.hnsw_stats.reset()
        t0 = time.perf_counter()
        for i in range(queries.shape[0]):
            index.search(queries[i:i + 1], 10)
        elapsed = time.perf_counter() - t0

        nq = queries.shape[0]
        ndis = faiss.cvar.hnsw_stats.ndis / nq
        nhops = faiss.cvar.hnsw_stats.nhops / nq
        us = elapsed / nq * 1e6
        ns_per_dis = us * 1000 / ndis if ndis else float("nan")

        if ef in java:
            jd, ju = java[ef]
            print(f"{ef:5d} {ndis:13,.0f} {nhops:9,.0f} {ndis / max(nhops, 1):8.1f} "
                  f"{us:9.1f} {ns_per_dis:8.1f} | {jd:12,} {ju:8.1f} "
                  f"{ju * 1000 / jd:8.1f}")
        else:
            print(f"{ef:5d} {ndis:13,.0f} {nhops:9,.0f} {ndis / max(nhops, 1):8.1f} "
                  f"{us:9.1f} {ns_per_dis:8.1f} |            -        -        -")

    print("\nRead the ns/dis columns first. If they differ and ndis matches, the gap is the\n"
          "kernel. If ndis differs, the graphs differ and the recall edge is the same fact.")


if __name__ == "__main__":
    main()
