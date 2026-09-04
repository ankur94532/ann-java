# Benchmark protocol

**Frozen.** Every number in `README.md` comes from this protocol. It was fixed before any
index was written and is not revised to suit a result. If something here turns out to be
a bad idea, it stays, and the problem is described in the limitations section instead.

Status: frozen at Checkpoint 1.

---

## 1. Datasets

| dataset | base vectors | dim | queries | ground truth | metric |
|---|---|---|---|---|---|
| SIFT1M (primary) | 1,000,000 | 128 | 10,000 | shipped, exact 100-NN | L2 |
| GIST1M (hard case, Phase 6) | 1,000,000 | 960 | 1,000 | shipped, exact 100-NN | L2 |

Both come from the TEXMEX corpus (`ftp.irisa.fr/local/texmex/corpus/`), fetched by
`scripts/download_sift.sh` and `scripts/download_gist.sh`. Vectors are read from the
`.fvecs`/`.ivecs` format into a single flat `float[]` of length `n * dim`.

The shipped ground truth is used as-is, and was verified against this project's own
brute-force search at Checkpoint 1. The verification is on **distances, not ids**: all
10,000 SIFT1M queries reproduce the shipped 10-NN distance sequence exactly, with zero
queries in which this search returned a vector genuinely farther than one the shipped
truth names.

Ids alone cannot be checked, because the dataset contains vectors that are *exactly*
equidistant from a query. Query 93, for example, has base vectors 196106 and 274922 both
at squared distance 42192.0 — and on SIFT that figure is not a rounding artefact: the
components are integers in [0, 255], so every squared distance is an integer below 2^24
and is represented in float32 with no error at all. Which of the two is "the" 10th
neighbour is arbitrary, and the shipped file and this scan happen to choose differently.

**This puts a ceiling on recall@10 of 0.999440**, not 1.0: 646 of the 10,000 queries
disagree with the shipped ids at some position purely through ties. Every recall figure
in this project is measured against the shipped ids on the same footing, so the ceiling
applies equally to the Java indexes and to FAISS and does not distort the comparison —
but a reported 0.9994 means "as good as exact", and no configuration can do better.

**GIST1M has the same property for a different reason, and the ceiling is 0.999200.** All
1,000 queries return a distance sequence no worse than the shipped one, with zero queries
where this search returned a vector genuinely farther; 64 disagree on ids. But where SIFT's
ties are *exact*, GIST's are not resolvable rather than equal. Summing 960 squared
differences in float32 accumulates relative error of order `sqrt(960) * eps ≈ 1.9e-6`, and
for at least one query (697) the gap between the 10th and 11th neighbours is smaller than
that: this scan finds 67023 at 1.3828166 where the shipped truth names 785940 at 1.3828262,
a relative difference of 7e-6. Neither is wrong; the two vectors are not distinguishable at
this precision.

Comparison against the ground truth therefore allows a relative tolerance of
`16 * sqrt(dim) * eps`, and is **one-sided**: finding a vector *closer* than the shipped
truth names is never an error, only finding one genuinely farther is. The tolerance is
derived from the dimension rather than the dataset, so SIFT — where it is never needed —
is not special-cased.

No normalisation, no dimensionality reduction, no deduplication is applied to any vector.

### 1.1 Oracle kernel

The brute-force oracle uses the **scalar** distance kernel, on both datasets, because the
thing every index is validated against should contain no clever code.

A selectable SIMD kernel exists (`oracle --kernel simd`) and was added on the belief that a
scalar scan would make GIST1M validation a thirteen-hour job. **That estimate was wrong by
about 250x** — it multiplied the dimension into the distance *count*, when the dimension is
already inside the per-distance cost. GIST1M has only 1,000 queries against 1,000,000 base
vectors, so it is 10^9 distance computations, each about 13x more expensive than a SIFT one:
roughly three minutes, not thirteen hours. The scalar oracle is entirely affordable and is
what both datasets use.

The SIMD option is kept because it is useful for larger query sets, and because it forced a
cross-check worth having: when it is used, the run re-verifies a subsample with the scalar
kernel and reports id agreement and the worst relative distance difference. On SIFT1M the
two agree exactly (300/300 queries, relative difference 0.00e+00), since every squared
distance there is an exact integer below 2^24.

## 2. Task

k-nearest-neighbour search with **k = 10**, under squared Euclidean distance. The square
root is never taken; it is monotonic and changes no ordering.

## 3. Recall

For one query, with `R` the 10 ids the index returned and `T` the 10 true nearest ids:

```
recall@10(q) = |R ∩ T| / 10
```

Set intersection, not rank correlation: an index that returns all ten true neighbours in
a different order scores 1.0. The reported figure is the unweighted mean over the whole
query set. This is the definition FAISS's own benchmarks use, which is what makes the
Java-vs-FAISS comparison meaningful.

Where fewer than 10 results are returned, the missing slots count as misses.

The attainable maximum is 0.999440, not 1.0, for the tie reason given in §1.

## 4. Query set

The **full shipped query set** — 10,000 queries for SIFT1M, 1,000 for GIST1M. During
development a prefix may be used for speed, but every number that reaches `README.md`,
`docs/`, or a committed CSV uses the full set.

Queries are issued in file order. No shuffling, no per-run resampling.

## 5. Latency

* **Single-threaded.** One query at a time, on one thread. Java: the measured loop is
  plain sequential code. FAISS: `faiss.omp_set_num_threads(1)`, one query per
  `index.search` call.
* **Warm-up.** Before the measured pass, a warm-up pass runs the first `min(1000, |Q|)`
  queries and discards their timings. The measured pass then runs the *full* query set.
  The warm-up is a separate pass so that no query is dropped from the measured set —
  this matters for GIST1M, whose query set is only 1,000 long.
* **Timing.** `System.nanoTime()` immediately around the search call; results are written
  into pre-allocated caller buffers so no allocation occurs inside the timed region.
* **Reported.** Mean and p95 per query, in microseconds. p95 is the 95th percentile of
  the per-query times of the measured pass, by nearest-rank on the sorted samples.
* **Repeats.** Each configuration is **built once** and its measured pass is **run three
  times**; the reported latency and recall are the **median of the three runs**. Building
  three times would multiply the sweep's wall-clock by an amount that buys nothing:
  construction is deterministic given the fixed seed, so all three builds would be the
  same index.

## 6. Build

Index construction is **single-threaded** on both sides (`faiss.omp_set_num_threads(1)`
covers FAISS build as well as search). Build time is wall-clock seconds from the first
vector inserted to the index being searchable, excluding the time to read the dataset off
disk. Single-threaded is slower than either implementation is capable of, but it is the
only setting in which "my build took X and FAISS's took Y" means anything.

HNSW's level assignment uses a fixed seed (42), so a rebuild of the same configuration
produces the same graph.

## 7. Memory

Two figures are recorded per configuration:

* **Retained heap.** `Runtime.totalMemory() - Runtime.freeMemory()` after three
  `System.gc()` calls with a settle pause, measured before and after construction; the
  index footprint is the difference. The base vectors are loaded before the first
  measurement, so they are outside the difference by construction.
* **Analytic size.** The index's own accounting of the arrays it owns, which is what the
  README's memory plot uses, because it is the figure that is comparable with the number
  FAISS reports for its serialised index.

Both figures **exclude the raw base vectors**. This is the interesting comparison: HNSW
must keep the full-precision vectors to compute distances during a search, so its true
cost is `footprint + raw`; IVF-PQ does not, and can discard them. The README states both
totals explicitly rather than hiding the difference in a footnote.

FAISS memory is `len(faiss.serialize_index(index))`, minus the size of a flat index over
the same vectors where the FAISS index stores them.

## 8. Hardware and software

All numbers, Java and FAISS, come from the same machine, on mains power, with no other
load:

```
cpu       : Apple M4 Pro (14 cores: 10 performance + 4 efficiency)
ram       : 48 GiB
os        : macOS 26.6.2 (Darwin 25.6.0), aarch64
jvm       : Temurin OpenJDK 21.0.9, --add-modules jdk.incubator.vector
simd      : 128-bit vectors (ARM NEON), 4 float lanes per vector
python    : 3.14, faiss-cpu (version recorded in the FAISS CSV header)
```

The 128-bit vector width matters for reading the Phase 2 results: on this machine a float
SIMD kernel has a hard ceiling of 4x over scalar, not the 8x or 16x an AVX-512 x86 box
would allow. The FAISS baseline is subject to the same ceiling, so the comparison is
unaffected — but the absolute kernel numbers are not portable to x86.

## 9. What is swept

* HNSW: `M ∈ {8, 16, 32}`, `efConstruction ∈ {100, 200, 400}`,
  `efSearch ∈ {16, 32, 64, 128, 256, 512}`
* IVF-PQ: `nlist ∈ {1024, 4096}`, `m ∈ {8, 16, 32}`, `nprobe ∈ {1, 4, 8, 16, 32, 64}`
* FAISS: the equivalent parameters on `IndexHNSWFlat` and `IndexIVFPQ`.

`efSearch` and `nprobe` are search-time knobs, so one build serves the whole sweep over
them.

## 10. Output

One CSV row per configuration, identical columns from the Java and the Python harness:

```
harness,dataset,index,params,k,recall_at_k,mean_latency_us,p95_latency_us,
build_seconds,index_bytes,base_bytes,queries,runs,timestamp
```

CSVs are committed under `docs/results/` so that every plot can be regenerated without
re-running the sweep.
