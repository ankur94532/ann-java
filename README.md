# ann-java

**recall@10 of 0.9920 on a million vectors in 165 µs per query, single-threaded, from a
from-scratch Java HNSW** — and an IVF-PQ that indexes the same million vectors in 19.7 MiB,
24.8x smaller than the raw data, matching FAISS's recall to within 0.0004.

Approximate nearest-neighbour indexes — **HNSW** and **IVF-PQ** — written from scratch in
Java 21 with the incubating Vector API, benchmarked against FAISS on SIFT1M under a
protocol frozen before any index was written.

The goal was never to beat FAISS. FAISS is years of C++ and hand-written SIMD by a team.
The goal was to land in the same neighbourhood, understand exactly where the remaining gap
comes from, and be able to point at it.

> **Status: in progress.** Checkpoints 0–4 are complete and every number below is measured.
> The full parameter sweep, the FAISS comparison plots, and the GIST1M high-dimensional
> analysis are still running. This README will grow the plots and the analysis section as
> they land.

---

## Headline numbers so far

SIFT1M, 1,000,000 base vectors × 128 dimensions, the full 10,000-query set, k=10,
single-threaded, on an Apple M4 Pro. Full protocol in [PROTOCOL.md](PROTOCOL.md).

| index | recall@10 | mean latency | index size | vs raw |
|---|---:|---:|---:|---:|
| HNSW, M=16 efC=200 ef=64 | 0.9703 | 92 µs | 138 MiB | — |
| HNSW, M=16 efC=200 ef=128 | 0.9920 | 165 µs | 138 MiB | — |
| IVF-PQ, nlist=1024 m=16 nprobe=64 | 0.5741 | 694 µs | 19.7 MiB | 24.8x |
| IVF-PQ, nlist=1024 m=32 nprobe=64 † | 0.7303 | — | 35.0 MiB | 14.0x |
| IVF-PQ, nlist=1024 m=64 nprobe=64 †‡ | 0.8979 | — | 65.5 MiB | 7.5x |
| exact brute force (the oracle) | 0.9994 | ~14 ms | — | — |

† measured before the codebook-layout change and the switch to FAISS-matching k-means
seeding. That change altered no result at m=16 and the seeding moved recall by ~0.005, so
these recall figures are good to about half a point; the latencies are superseded and are
omitted rather than quoted stale. Refreshed numbers land with the sweep.
‡ m=64 is outside the parameter grid PROTOCOL.md §9 froze. It is reported as a diagnostic,
not as a sweep result — see below.

The two indexes are not competing for the same job, and the table makes that obvious: HNSW
buys recall with memory, IVF-PQ buys memory with recall. HNSW needs the full-precision
vectors alongside its 138 MiB graph to compute distances at all, so its true cost is
626 MiB. IVF-PQ does not keep the vectors, and 19.7 MiB is the whole index.

### The IVF-PQ frontier, and a target that cannot be hit

The three IVF-PQ rows are one curve, and the shape of it is the main IVF-PQ result here.
Recall is capped by the code size, not by how many lists you scan: at m=16 the curve is
flat from nprobe=32 onward at 0.574, and scanning every list in the index would not move
it. The ceiling is quantization error, and the only way to lift it is to spend more bytes
per vector.

That makes recall and compression two ends of one dial, and the project's own Checkpoint 4
asked for **recall > 0.80 together with ≥8x compression**. Measured, those sit on opposite
sides of the frontier: 0.80 recall needs 64-byte codes, and 64-byte codes are 7.5x. The
two conditions cannot both be met on SIFT1M at k=10 with 8-bit product quantization. FAISS
shows the same ceilings at the same code sizes, so this is a property of the method rather
than of this implementation.

The checkpoint is reported as missed on its own terms rather than satisfied by widening the
frozen grid until something passes. An OPQ rotation before quantization is the honest way
to actually move this frontier and is noted under Limitations.

---

## The recall ceiling is 0.9994, not 1.0

Checkpoint 1 compared this project's own exact search against the ground truth shipped with
SIFT1M. All 10,000 queries reproduce the shipped 10-NN **distance** sequence exactly, with
zero queries where the search returned anything genuinely farther. But 646 of them disagree
on **ids**, because the dataset contains vectors that are exactly equidistant from a query.

Query 93 has base vectors 196106 and 274922 both at squared distance 42192.0 — and on SIFT
that is not a rounding artefact. Components are integers in [0, 255], so every squared
distance is an integer below 2²⁴ and float32 represents it with no error at all. Which of
the two is "the" 10th neighbour is arbitrary; the shipped file and this scan happen to
choose differently.

**So no configuration in this project can score above 0.999440.** It applies equally to the
Java indexes and to FAISS, so the comparison is unaffected — but a reported 0.9994 means
"as good as exact", and it is worth knowing that 1.0 is not on the table.

---

## Where the FAISS gap comes from

Measured against `faiss-cpu` 1.15.0, same machine, same protocol, `omp_set_num_threads(1)`,
same full query set on both sides:

| config (IVF-PQ, nlist=1024, m=16) | recall@10 | mine | FAISS | ratio |
|---|---:|---:|---:|---:|
| nprobe=1 | 0.3130 / 0.3113 | 23.4 µs | 24.6 µs | **0.95x** |
| nprobe=8 | 0.5384 / 0.5400 | 100.6 µs | 54.4 µs | 1.85x |
| nprobe=64 | 0.5741 / 0.5737 | 694.3 µs | 273.8 µs | 2.54x |

(Recall column is mine / FAISS.) Recall matches to within 0.0004 at nprobe=64. The gap is
entirely latency, and the shape of it is the informative part: **at nprobe=1 this
implementation is marginally faster than FAISS, and the gap only opens as nprobe grows.**
The fixed per-query cost — coarse search over 1024 centroids, heap setup — is competitive.
The marginal per-list cost is not, and that is where all 2.5x lives.

Within that marginal cost, the list scan is the remainder: it runs at about 1.8 cycles per
table lookup against a load-throughput limit nearer 1.1. The lookup-table half was 62% of a
query before the layout change and is 26% after.

**An earlier version of this README claimed the Java implementation beat FAISS on recall.
That was wrong**, and the cause is worth recording: the two sides had been run on different
query subsets (10,000 against 2,000). On matched query sets FAISS is very slightly ahead.
Recall varies enough between query subsets to manufacture a difference of about a
percentage point, which is the same size as the effect being claimed.

---

## Implementation notes

### Distance kernels — [docs/kernels.md](docs/kernels.md)

Scalar and Vector API kernels, tested against each other across dimensions that are not
multiples of the lane count or the unroll factor.

| | d=128 | d=960 |
|---|---:|---:|
| SIMD speedup over scalar | **4.16x** | **10.17x** |

The machine has 128-bit vectors (ARM NEON, 4 float lanes), so 4x is what the lane count
alone can buy and 4.16x at d=128 is essentially that ceiling. The extra 4% is not extra
parallelism: the scalar loop does not quite sustain one element per cycle either, and the
SIMD version recovers that slack as well as the lanes. **10.17x at d=960, though, is not
reachable by any amount of slack on a 4-lane machine, which means the scalar baseline is
the thing that is broken** — and it is. Java may not reassociate a float sum,
so `sum += d*d` runs at one addition per FP-add *latency* rather than throughput. Four
independent SIMD accumulators win 4x from the lanes and another 2.6x from breaking a
dependency chain the scalar version was never allowed to break.

Recorded honestly alongside it: at d=128 the four-accumulator kernel is 0.5 ns *slower*
than the single-accumulator one, because its reduction tree cannot amortise over 32 steps.

### HNSW optimization — [docs/hnsw-optimization.md](docs/hnsw-optimization.md)

Six implementations of the same algorithm, each measured under the same protocol.

| change | p95 @ ef=64 | build |
|---|---:|---:|
| 0. naive reference | 447.5 µs | 779.3 s |
| 1. flat `int[]` arenas | 357.0 µs | 603.7 s |
| 2. versioned visited stamps | 182.3 µs | 360.4 s |
| 3. primitive `long[]` heaps | 160.4 µs | 282.6 s |
| 4. split traversal / distances | 159.2 µs | 283.1 s |
| 5. software prefetch | **117.0 µs** | **220.2 s** |

**Recall is identical to six decimal places in every row at every one of six `efSearch`
values, and the edge count matches to the digit.** That is enforced, not lucky: a test
asserts the naive and optimized implementations build a *byte-identical* graph. Without it,
a "faster" row could just as easily be a worse graph that searches less thoroughly.

The interesting rows are the ones that did not go as expected. The flat-arena change I
was most confident about was worth 20%; the `HashSet` I had not suspected was worth 49%;
and **step 4 was worth nothing at all** — until step 5, where it turned out to be the
precondition that makes a software prefetch possible. A table that kept only the changes
that paid off immediately would have thrown it away.

### IVF-PQ

The largest single win came from a **layout** change, not a faster kernel. Measuring the
two halves of a query separately showed lookup-table construction at 62% against list scan
at 38% — so the table build, not the scan, was the problem. Centroid-major codebooks make
the table 256 calls to a distance kernel over subvectors of 2–16 elements, where the SIMD
prologue costs more than the arithmetic. Storing the codebooks **dimension-major** and
vectorising across the *codeword* axis instead gave 5–8x on the table and 1.95x end to end.

A second attempt failed instructively. Four accumulators in the code-scanning loop — the
exact fix that wins 2.6x in the L2 kernel — measured as **nothing**, because the caller's
loop over codes already supplies all the instruction-level parallelism the processor needs.
It was reverted, and the reasoning is recorded rather than the code.

---

## Reproducing

```bash
./scripts/download_sift.sh                 # ~168 MB, into data/
./gradlew build
./gradlew run --args="oracle --dataset sift"          # Checkpoint 1: validate the loader
./gradlew run --args="hnsw --ef 16,32,64,128,256,512" # HNSW recall/latency curve
./gradlew run --args="ivfpq --nlist 1024 --m 16"      # IVF-PQ recall/latency curve
./gradlew run --args="sweep --dataset sift"           # the full sweep (hours, resumable)
./gradlew jmh -PjmhArgs="DistanceBenchmark"           # kernel microbenchmarks
```

FAISS baseline and plots:

```bash
./scripts/setup_python.sh
./.venv/bin/python scripts/faiss_bench.py --dataset sift --csv docs/results/faiss-sift1m.csv
./.venv/bin/python scripts/plot_results.py docs/results/*.csv --out docs/plots
```

Requires JDK 21; Gradle resolves the toolchain. The Vector API is an incubator module, so
every JVM invocation needs `--add-modules jdk.incubator.vector` — the build files do this.

**Hardware these numbers come from:** Apple M4 Pro (10 performance + 4 efficiency cores),
48 GiB, macOS 26.6, Temurin OpenJDK 21.0.9, 128-bit float vectors. The SIMD ratios in
particular are not portable: an AVX-512 host runs the same source 16 lanes wide.

---

## Limitations

* **The recall ceiling is 0.9994**, for the tie reason above.
* **Single-threaded throughout**, build and search, on both sides. That is the only setting
  in which "mine took X and FAISS took Y" means anything, but it is not how either would be
  deployed.
* **The IVF-PQ search gap to FAISS is 2.5x** at high `nprobe` and is not closed. It is
  localised to the list scan and quantified, not hand-waved.
* **No OPQ rotation**, so the product quantizer assumes the subspaces are uncorrelated. SIFT
  satisfies that reasonably; GIST, whose 960 dimensions are far more correlated across
  subspace boundaries, is expected not to. OPQ learns a rotation that decorrelates them and
  typically buys real recall at a fixed code size — plausibly enough to clear 0.80 at m=32,
  which would put both halves of Checkpoint 4 within reach honestly rather than by
  redefining it. Not implemented.
* **`m` must divide the dimension in this implementation**, so on SIFT the available code
  sizes are 8, 16, 32, 64 and 128 bytes and nothing between — which is why the Checkpoint 4
  frontier has no point to land on between 32 and 64 bytes. This is an implementation
  choice, not a property of product quantization: FAISS pads the last subvector and accepts
  any `m`. Fixing it would give the intermediate code sizes and a denser frontier.
* Deletion, updates, and persistence are all unimplemented. This indexes a fixed set once.
