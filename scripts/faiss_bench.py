#!/usr/bin/env python3
"""FAISS baseline for the same protocol as the Java harness.

Emits CSV rows with exactly the columns Measurement.CSV_HEADER defines, so the Java and
FAISS results concatenate into one dataframe with no reconciliation step.

PROTOCOL.md compliance, point by point:
  * k = 10, squared L2, shipped ground truth read from the same .ivecs file (§2, §3).
  * recall@10 is set overlap, computed by the same rule as bench/Recall.java (§3).
  * single-threaded search AND build: faiss.omp_set_num_threads(1) (§5, §6).
  * warm-up pass over the first min(1000, |Q|) queries, discarded; then the full query
    set, three times, median reported (§5).
  * one query per search() call, so the reported latency is per-query and not amortised
    over a batch the Java side never gets to use (§5).
  * IVFPQ uses use_precomputed_table = 0 to match the Java implementation, which rebuilds
    the residual lookup table per probed list rather than storing an nlist x m x 256 table.

Usage:
    python3 scripts/faiss_bench.py --dataset sift --index hnsw --csv docs/results/faiss.csv
"""
import argparse
import csv
import datetime
import os
import sys
import time

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover
    sys.exit("faiss is not installed; see scripts/setup_python.sh")

CSV_HEADER = [
    "harness", "dataset", "index", "params", "k", "recall_at_k",
    "mean_latency_us", "p95_latency_us", "build_seconds", "index_bytes",
    "base_bytes", "queries", "runs", "timestamp",
]

DATASETS = {
    "sift": ("data/sift", "sift_base.fvecs", "sift_query.fvecs", "sift_groundtruth.ivecs"),
    "gist": ("data/gist", "gist_base.fvecs", "gist_query.fvecs", "gist_groundtruth.ivecs"),
}


def read_fvecs(path, limit=None):
    """.fvecs is [int32 d][d float32] repeated, with no file header."""
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    raw = raw.reshape(-1, dim + 1)
    if limit is not None:
        raw = raw[:limit]
    if not np.all(raw[:, 0] == dim):
        raise ValueError(f"{path} has a ragged record")
    return np.ascontiguousarray(raw[:, 1:]).view(np.float32)


def read_ivecs(path, limit=None):
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    raw = raw.reshape(-1, dim + 1)
    if limit is not None:
        raw = raw[:limit]
    return np.ascontiguousarray(raw[:, 1:])


def recall_at_k(found, truth, k):
    """Mean set overlap, identical to bench/Recall.java."""
    hits = 0
    for row_found, row_truth in zip(found[:, :k], truth[:, :k]):
        hits += len(np.intersect1d(row_found, row_truth, assume_unique=False))
    return hits / (found.shape[0] * k)


def measure(index, queries, truth, k, runs):
    """One warm-up pass, then `runs` measured passes over the full query set."""
    nq = queries.shape[0]
    warmup = min(1000, nq)
    for i in range(warmup):
        index.search(queries[i:i + 1], k)

    means, p95s, recalls = [], [], []
    for _ in range(runs):
        found = np.empty((nq, k), dtype=np.int64)
        times = np.empty(nq, dtype=np.float64)
        for i in range(nq):
            t0 = time.perf_counter_ns()
            _, ids = index.search(queries[i:i + 1], k)
            times[i] = time.perf_counter_ns() - t0
            found[i] = ids[0]
        means.append(times.mean() / 1000.0)
        # Nearest-rank, matching BenchHarness.percentile.
        ordered = np.sort(times)
        rank = int(np.ceil(0.95 * nq))
        p95s.append(ordered[min(nq, max(1, rank)) - 1] / 1000.0)
        recalls.append(recall_at_k(found, truth, k))
    return float(np.median(recalls)), float(np.median(means)), float(np.median(p95s))


def index_bytes(index):
    return len(faiss.serialize_index(index))


def row(dataset_label, name, params, k, recall, mean_us, p95_us, build_s, nbytes,
        base_bytes, nq, runs):
    return {
        "harness": "faiss",
        "dataset": dataset_label,
        "index": name,
        "params": params,
        "k": k,
        "recall_at_k": f"{recall:.6f}",
        "mean_latency_us": f"{mean_us:.3f}",
        "p95_latency_us": f"{p95_us:.3f}",
        "build_seconds": f"{build_s:.3f}",
        "index_bytes": nbytes,
        "base_bytes": base_bytes,
        "queries": nq,
        "runs": runs,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def sweep_hnsw(base, queries, truth, args, writer):
    dim = base.shape[1]
    for m in args.m_values:
        for ef_construction in args.efc_values:
            index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_L2)
            index.hnsw.efConstruction = ef_construction
            t0 = time.perf_counter()
            index.add(base)
            build_s = time.perf_counter() - t0
            nbytes = index_bytes(index)
            print(f"  built HNSW M={m} efC={ef_construction} in {build_s:.1f}s "
                  f"({nbytes / 2**20:.1f} MiB)", flush=True)
            for ef_search in args.ef_values:
                index.hnsw.efSearch = ef_search
                recall, mean_us, p95_us = measure(index, queries, truth, args.k, args.runs)
                params = f"M={m},efC={ef_construction},ef={ef_search}"
                print(f"    {params:32} recall={recall:.4f} mean={mean_us:8.1f}us "
                      f"p95={p95_us:8.1f}us", flush=True)
                writer.writerow(row(args.dataset_label, "IndexHNSWFlat", params, args.k,
                                    recall, mean_us, p95_us, build_s, nbytes,
                                    base.nbytes, queries.shape[0], args.runs))


def sweep_ivfpq(base, queries, truth, args, writer):
    dim = base.shape[1]
    for nlist in args.nlist_values:
        for m in args.pq_m_values:
            if dim % m != 0:
                continue
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8, faiss.METRIC_L2)
            # Match the Java implementation: no nlist x m x 256 precomputed table.
            index.use_precomputed_table = 0
            index.cp.niter = args.niter
            index.cp.max_points_per_centroid = args.points_per_centroid
            t0 = time.perf_counter()
            index.train(base)
            index.add(base)
            build_s = time.perf_counter() - t0
            nbytes = index_bytes(index)
            print(f"  built IVFPQ nlist={nlist} m={m} in {build_s:.1f}s "
                  f"({nbytes / 2**20:.1f} MiB)", flush=True)
            for nprobe in args.nprobe_values:
                if nprobe > nlist:
                    continue
                index.nprobe = nprobe
                recall, mean_us, p95_us = measure(index, queries, truth, args.k, args.runs)
                params = f"nlist={nlist},m={m},nprobe={nprobe}"
                print(f"    {params:32} recall={recall:.4f} mean={mean_us:8.1f}us "
                      f"p95={p95_us:8.1f}us", flush=True)
                writer.writerow(row(args.dataset_label, "IndexIVFPQ", params, args.k,
                                    recall, mean_us, p95_us, build_s, nbytes,
                                    base.nbytes, queries.shape[0], args.runs))


def int_list(text):
    return [int(part) for part in text.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="sift")
    parser.add_argument("--index", choices=["hnsw", "ivfpq", "both"], default="both")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--queries", type=int, default=None,
                        help="cap the query set; final numbers must use the full set")
    parser.add_argument("--csv", default="docs/results/faiss.csv")
    parser.add_argument("--m-values", type=int_list, default=[8, 16, 32])
    parser.add_argument("--efc-values", type=int_list, default=[100, 200, 400])
    parser.add_argument("--ef-values", type=int_list, default=[16, 32, 64, 128, 256, 512])
    parser.add_argument("--nlist-values", type=int_list, default=[1024, 4096])
    parser.add_argument("--pq-m-values", type=int_list, default=[8, 16, 32])
    parser.add_argument("--nprobe-values", type=int_list, default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--niter", type=int, default=25)
    parser.add_argument("--points-per-centroid", type=int, default=256)
    args = parser.parse_args()

    faiss.omp_set_num_threads(1)

    directory, base_file, query_file, truth_file = DATASETS[args.dataset]
    print(f"faiss {faiss.__version__}, threads={faiss.omp_get_max_threads()}", flush=True)
    base = read_fvecs(os.path.join(directory, base_file))
    queries = read_fvecs(os.path.join(directory, query_file), args.queries)
    truth = read_ivecs(os.path.join(directory, truth_file), args.queries)
    args.dataset_label = "SIFT1M" if args.dataset == "sift" else "GIST1M"
    print(f"{base.shape[0]:,} base x {base.shape[1]} dims, {queries.shape[0]:,} queries",
          flush=True)

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    exists = os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
        if not exists:
            writer.writeheader()
        if args.index in ("hnsw", "both"):
            sweep_hnsw(base, queries, truth, args, writer)
        if args.index in ("ivfpq", "both"):
            sweep_ivfpq(base, queries, truth, args, writer)
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
