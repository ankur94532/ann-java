# ann-java

**recall@10 of 0.9920 on a million vectors in 165 µs per query, single-threaded, from a
from-scratch Java HNSW** — and an IVF-PQ that indexes the same million vectors in 19.7 MiB,
24.8x smaller than the raw data.

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

## The oracle is provably exact, and the recall ceiling is 0.9994

Checkpoint 1 validated this project's own exact search against the ground truth shipped
with SIFT1M — and it validated something stronger than an id-for-id match. **All 10,000
queries reproduce the shipped 10-NN distance sequence exactly, with zero queries where this
search returned a vector genuinely farther than one the shipped truth names.** That is the
well-defined statement of "the search is exact"; matching ids is not, and cannot be, because
the dataset contains vectors exactly equidistant from a query.

646 queries do disagree on ids, entirely through those ties. Checking distances instead of
ids is what makes the disagreement diagnosable rather than alarming: it separates "the
loader or the scan is wrong" from "the question has more than one right answer."

Query 93 has base vectors 196106 and 274922 both at squared distance 42192.0 — and on SIFT
that is not a rounding artefact. Components are integers in [0, 255], so every squared
distance is an integer below 2²⁴ and float32 represents it with no error at all. Which of
the two is "the" 10th neighbour is arbitrary; the shipped file and this scan happen to
choose differently.

**So no configuration in this project can score above 0.999440** — not because anything is
approximate, but because the metric asks for ids and the ids are not unique. It applies
equally to the Java indexes and to FAISS, so the comparison is unaffected; a reported 0.9994
means "as good as exact", and 1.0 was never on the table.

---

## The curves

Three plots, generated from the committed CSVs by `scripts/plot_results.py`. Blue is HNSW,
orange is IVF-PQ; solid is this project, dashed is FAISS. Each line is the Pareto frontier
of its sweep — configurations beaten on both axes at once are dropped rather than joined by
a zig-zag implying a trade nobody would choose.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/plots/recall-latency-sift1m-dark.png">
  <img alt="Recall against latency on SIFT1M. This project's HNSW sits below FAISS's; this project's IVF-PQ sits above FAISS's." src="docs/plots/recall-latency-sift1m-light.png">
</picture>

The headline. The two blue lines are the HNSW pair and the solid one is below — faster at
every recall. The two orange lines are the IVF-PQ pair and the solid one is above — slower
at every recall. **This project wins the graph index and loses the quantized one**, and the
rest of this README is the explanation of both halves.

Note also where the orange lines stop: IVF-PQ cannot be pushed past recall ≈0.74 at these
code sizes no matter how much latency it is given, while HNSW runs all the way to the 0.9994
ceiling. That wall is quantization error, not search effort.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/plots/recall-memory-sift1m-dark.png">
  <img alt="Recall against index memory on SIFT1M. IVF-PQ occupies 12-37 MiB; HNSW occupies 77-260 MiB." src="docs/plots/recall-memory-sift1m-light.png">
</picture>

The two families live in different regions and barely compete. IVF-PQ spans 12–37 MiB;
HNSW spans 77–260 MiB and needs the 488 MiB of raw vectors on top. The HNSW pair is a
single visible line because the two implementations agree to 0.2%.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/plots/build-recall-sift1m-dark.png">
  <img alt="Build time against recall on SIFT1M. IVF-PQ builds split into two tiers by nlist." src="docs/plots/build-recall-sift1m-light.png">
</picture>

One point per built index — a scatter, not a curve, since joining builds that differ in `M`
or `nlist` would draw a trajectory nothing travels along. This is the plot where the two
implementations diverge most sharply, and in opposite directions:

| | mine vs FAISS |
|---|---|
| HNSW build | **1.09–1.23x faster** |
| IVF-PQ build | **4.6–8.7x slower** |

The IVF-PQ build is roughly 95% k-means, and the gap has a specific and unglamorous cause:
**the assignment step is a matrix multiply in disguise.** FAISS computes every
point-to-centroid distance as a GEMM — `‖x‖² + ‖c‖² − 2·X·Cᵀ` — so each vector is fetched
once per *block* of centroids and reused across all of them. This implementation issues
n × k independent distance calls, fetching each vector once per centroid and discarding the
reuse entirely. At nlist=4096 that is 1.0 × 10¹¹ distance computations: FAISS sustains about
750 M/s where this code manages about 75 M/s, which is the factor blocking would be expected
to buy. It is the largest single gap in the project and the best-understood.

## HNSW is faster than FAISS here, and this is why

This project's HNSW beats `IndexHNSWFlat` on latency at all 36 swept configurations, with
equal or marginally better recall. That is not a credible result for a from-scratch
implementation against a decade of tuned C++, so it was treated as a defect in the
comparison until it could be explained. Three checks, in order.

**The graphs are the same.** Corrected for the `IndexFlat` that `IndexHNSWFlat` embeds,
FAISS's graph is 137.6 MiB at M=16 against this project's 137.9 — 0.2% apart, and within
0.7% at every M. Two independent implementations filling the same degree budget to the same
extent.

**The searches do the same work.** `faiss.cvar.hnsw_stats.ndis` counts distance evaluations
inside HNSW's own search. Per query, at M=16 efC=200:

| efSearch | ndis, mine / FAISS | ns per distance, mine / FAISS | gap |
|---:|---|---|---:|
| 16 | 518 / 514 | 63.1 / 71.4 | 8.3 ns |
| 64 | 1,323 / 1,297 | 69.5 / 82.1 | 12.6 ns |
| 512 | 6,942 / 6,781 | 76.8 / 112.2 | 35.4 ns |

**This implementation computes about 2% _more_ distances than FAISS and is still faster.**
The recall edge falls out of the same fact: it finds slightly more true neighbours because
it examines slightly more candidates, not because its graph is better.

**It is not a crippled FAISS build.** The wheel reports compile options
`OPTIMIZE DD ARM_NEON MAC_METAL` with the full ASIMD instruction set available. That was the
obvious explanation and it is wrong.

What is left is execution efficiency. On the shape of the gap I can say less than I would
like, and it is worth being precise about which half is measured.

**Measured, on this side only.** A JFR profile of the search phase splits this
implementation's query time into roughly 47% distance kernel and 53% bookkeeping — the
frontier heap, the result beam, the visited stamps, the arena reads. That number is from
`docs/results/jfr-profile-after-step5.txt`.

**Inferred, about FAISS.** The per-distance gap is not constant: it grows from 13% at ef=16
to 46% at ef=512. A pure distance-kernel difference would be flat, so *something* that scales
with beam width is involved. That is consistent with per-candidate bookkeeping being the
larger part, and it matches the shape of this project's own optimization history — steps 2
and 3 in [docs/hnsw-optimization.md](docs/hnsw-optimization.md) attacked exactly that
bookkeeping, and their benefit also grew with `efSearch` (step 3: −8% at ef=16, −17% at
ef=512).

**But consistent-with is not demonstrated.** Splitting FAISS's per-distance cost into kernel
and bookkeeping would mean instrumenting its `DistanceComputer`, which has not been done. The
obvious cheap substitute — sweeping `M` at fixed `efSearch` — does not work, because `M`
simultaneously changes distances per hop, graph connectivity and therefore hop count, and
neighbour-list size and therefore cache behaviour; a gap that tracked `M` would not say which
of the three caused it. So the decomposition above is a hypothesis the data is compatible
with, not a result.

**The scope of the claim.** Single-threaded, one query per call, on aarch64, against
`faiss-cpu` 1.15.0. FAISS's batched search paths, its threading, and its x86 AVX-512 kernels
are all unexercised, and the kernel comparison in particular would likely reverse on an
AVX-512 host, where FAISS gets 16 lanes and hand-written intrinsics. What this shows is that
the *algorithm* was implemented correctly and the *execution* was optimised carefully — not
that FAISS is slow.

## Where the IVF-PQ gap comes from

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
The marginal per-list cost is not, and that is where all 2.54x lives.

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

The machine has 128-bit vectors (ARM NEON, 4 float lanes), so 4x is all the lane count can
buy. **The two figures differ because they are two different kernels**, and that is the
whole point:

* At **d=128** the fastest kernel is the *single-accumulator* one (12.8 ns). It carries the
  same serial FMA dependency chain the scalar loop has, so it wins the lanes and nothing
  else — 4.16x, essentially the lane ceiling. Adding accumulators makes it *slower* here
  (13.3 ns), because the reduction tree cannot amortise over only 32 vector steps.
* At **d=960** the fastest is the *four-accumulator* one (70.2 ns). The single-accumulator
  version manages only 3.89x — again lanes alone — and the four-accumulator version gets
  another 2.6x on top by breaking the dependency chain. 4 × 2.6 ≈ 10.17x.

The chain is the thing. Java may not reassociate a float sum, so `sum += d*d` advances at
one addition per FP-add *latency* rather than throughput, and any kernel that keeps a single
accumulator inherits that limit no matter how wide its vectors are. **A speedup above the
lane count is never extra parallelism in the SIMD path; it is a serial dependency in the
baseline that the SIMD path was allowed to break and the scalar loop was not.**

### Two implementations, one graph size

A useful accident of the memory accounting: FAISS's `IndexHNSWFlat` serialises a full
`IndexFlat` alongside its graph, so its raw serialised size at M=16 is 625.9 MiB. Subtract
the 488.3 MiB of base vectors that `IndexFlat` holds — which PROTOCOL.md §7 requires anyway,
since the Java index reports its graph without the vectors it borrows — and FAISS's graph is
**137.6 MiB against this project's 137.9 MiB, a difference of 0.2%.**

That is not just an accounting fix. Two independent implementations of HNSW, given the same
M and the same dataset, allocating graph memory within a fifth of a percent of each other is
a correctness signal: it says both are building a graph with the same degree budget actually
filled to the same extent, which a subtly wrong pruning rule or an off-by-one in the degree
cap would not produce.

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

* **The metric's ceiling is 0.999440**, because ids are not unique under exact distance
  ties — the search itself is exact. See the Checkpoint 1 section above.
* **Single-threaded throughout**, build and search, on both sides. That is the only setting
  in which "mine took X and FAISS took Y" means anything, but it is not how either would be
  deployed.
* **The IVF-PQ search gap to FAISS is 2.54x** at nprobe=64 and is not closed. It is
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
