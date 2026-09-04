# Results

Every CSV here is committed so that the plots and the tables in the README can be
regenerated without re-running anything. All of them were produced under
[PROTOCOL.md](../../PROTOCOL.md) on the machine recorded in its §8.

## Sweep results

| file | contents |
|---|---|
| `java-sift1m.csv` | the full Java HNSW and IVF-PQ sweep on SIFT1M |
| `java-gist1m.csv` | the same on GIST1M |
| `faiss-sift1m.csv` | `IndexHNSWFlat` and `IndexIVFPQ`, same protocol, same machine |
| `faiss-gist1m.csv` | the same on GIST1M |

Columns are identical across the Java and the Python harness:

```
harness,dataset,index,params,k,recall_at_k,mean_latency_us,p95_latency_us,
build_seconds,index_bytes,base_bytes,queries,runs,timestamp
```

* `recall_at_k` — mean set overlap against the shipped ground truth, per PROTOCOL.md §3.
  The attainable maximum is **0.999440**, not 1.0, because the dataset contains exactly
  tied neighbours; see PROTOCOL.md §1.
* `mean_latency_us` / `p95_latency_us` — single-threaded, one query per call, median of
  three measured passes over the full query set after a discarded warm-up pass.
* `build_seconds` — single-threaded construction, excluding dataset load. For IVF-PQ
  configurations that shared a coarse quantizer across `m` values during the sweep, this
  still includes the full k-means training cost, so it is what a from-scratch build of
  that configuration would have taken.
* `index_bytes` — the index structures only, excluding the raw base vectors.
  `base_bytes` is those raw vectors, for the compression ratio.

## Optimization series

`hnsw-opt-step*.csv` are the before/after measurements behind
[docs/hnsw-optimization.md](../hnsw-optimization.md). Every one is the same
configuration — SIFT1M, M=16, efConstruction=200, the full 10,000-query set, three runs —
so the only thing that varies between files is the implementation.

## Per-query analysis

`perquery-*.csv` are written by `Main analyse` and hold one row per query: its recall,
the distance to its 1st and 10th true neighbours, the ratio of the two, and how many of
its true neighbours are graph orphans. These are the input to the Phase 6 failure
analysis.

## Checkpoint records

`checkpoint1-oracle-sift1m.txt` is the Checkpoint 1 oracle run: exact search over the
full query set, validated against the shipped ground truth.
