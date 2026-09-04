# Distance kernels

Checkpoint 2. Scalar versus Vector-API (`jdk.incubator.vector`) distance kernels,
measured with JMH at the two dimensions this project benchmarks: 128 (SIFT1M) and 960
(GIST1M).

Reproduce with:

```
./gradlew jmh -PjmhArgs="DistanceBenchmark -rf json -rff build/jmh-distance.json"
python3 scripts/jmh_table.py build/jmh-distance.json
```

## Machine

```
cpu   : Apple M4 Pro (10 performance + 4 efficiency cores), 48 GiB
jvm   : Temurin OpenJDK 21.0.9, --add-modules jdk.incubator.vector
simd  : FloatVector.SPECIES_PREFERRED = 128-bit (ARM NEON), 4 float lanes
jmh   : 1.37, 1 fork, 3x2s warmup, 5x2s measurement, average time
```

**Four lanes is the ceiling here.** On this machine a float SIMD kernel can do four
elements per instruction, so a "SIMD speedup" larger than ~4x is not evidence of wider
vectors — it means the scalar baseline it is being compared against was slow for some
*other* reason, and below that is exactly what happens at d=960. On an AVX-512 host the
same source runs 16 lanes wide and every ratio here changes; the shapes of the conclusions
do not.

## Results

### L2 squared, one L1-resident pair

| kernel | d=128 ns/op | d=128 vec/s | d=128 vs scalar | d=960 ns/op | d=960 vec/s | d=960 vs scalar |
|---| ---: | ---: | ---: | ---: | ---: | ---: |
| scalar | 53.2 ± 0.8 | 18.8 M | 1.00x | 714.2 ± 5.5 | 1.4 M | 1.00x |
| SIMD, 1 accumulator | 12.8 ± 0.1 | 78.2 M | **4.16x** | 183.8 ± 1.8 | 5.4 M | **3.89x** |
| SIMD, 4 accumulators | 13.3 ± 0.9 | 75.2 M | **4.00x** | 70.2 ± 1.1 | 14.2 M | **10.17x** |

### Inner product, one L1-resident pair

| kernel | d=128 ns/op | d=128 vec/s | d=128 vs scalar | d=960 ns/op | d=960 vec/s | d=960 vs scalar |
|---| ---: | ---: | ---: | ---: | ---: | ---: |
| scalar | 46.7 ± 0.3 | 21.4 M | 1.00x | 671.7 ± 16.6 | 1.5 M | 1.00x |
| SIMD, 4 accumulators | 12.3 ± 0.2 | 81.6 M | **3.81x** | 63.3 ± 0.4 | 15.8 M | **10.62x** |

### L2 squared, scan of a 4 MiB block (fits in L2)

| kernel | d=128 ns/op | d=128 vec/s | d=128 vs scalar | d=960 ns/op | d=960 vec/s | d=960 vs scalar |
|---| ---: | ---: | ---: | ---: | ---: | ---: |
| scalar | 497,303.1 ± 7,045.9 | 16.5 M | 1.00x | 797,910.4 ± 9,526.5 | 1.4 M | 1.00x |
| SIMD, 4 accumulators | 109,902.7 ± 2,525.3 | 74.5 M | **4.52x** | 83,523.7 ± 3,457.2 | 13.1 M | **9.55x** |

### L2 squared, scan of a 64 MiB block (comes from DRAM)

| kernel | d=128 ns/op | d=128 vec/s | d=128 vs scalar | d=960 ns/op | d=960 vec/s | d=960 vs scalar |
|---| ---: | ---: | ---: | ---: | ---: | ---: |
| scalar | 8,099,273.1 ± 153,829.5 | 16.2 M | 1.00x | 12,817,930.4 ± 459,026.0 | 1.4 M | 1.00x |
| SIMD, 4 accumulators | 1,742,969.2 ± 11,392.0 | 75.2 M | **4.65x** | 1,353,994.5 ± 18,079.9 | 12.9 M | **9.47x** |

### Derived: cost per SIMD iteration and DRAM bandwidth

| kernel | d=128 ns per 4-lane step | d=960 ns per 4-lane step | d=128 DRAM GB/s | d=960 DRAM GB/s |
|---| ---: | ---: | ---: | ---: |
| scalar | 1.664 | 2.976 | 8.3 | 5.2 |
| SIMD, 1 accumulator | 0.400 | 0.766 | - | - |
| SIMD, 4 accumulators | 0.416 | 0.293 | 38.5 | 49.6 |

## What the numbers say

**At d=128 the SIMD kernel hits the lane ceiling and stops.** 4.16x against a 4-lane
machine is as good as this gets, and the four-accumulator variant is very slightly
*slower* than the single-accumulator one (13.3 vs 12.8 ns). That is the first row of the
table that is worth more than its speedup: unrolling is not free. Four accumulators cost
four vector zeroings up front and a three-add reduction tree plus a horizontal
`reduceLanes` at the end, and at d=128 there are only 32 vector steps to amortise that
over. The fixed cost is about 4 ns; the loop is only 13.

**At d=960 the four-accumulator kernel goes 10.2x, which is impossible for a 4-lane
machine — so the scalar baseline must be the thing that is broken.** It is. The scalar
loop is

```java
sum += d * d;
```

and every iteration's addition needs the previous iteration's `sum`. Java is strict about
float semantics, so neither javac nor HotSpot may reassociate that sum, and the loop runs
at one addition per FP-add *latency* rather than per FP-add *throughput*, no matter how
many FP units the core has. The measured cost is 2.98 ns per four elements at d=960 —
roughly 13 cycles for what is 4 multiplies and 4 adds of real work.

The single-accumulator SIMD kernel has exactly the same problem one level up: it is one
FMA chain, so it is latency-bound too, and its 0.766 ns per 4-lane step at d=960 is about
3.4 cycles — an FMA latency, not an FMA throughput. Splitting into four independent
accumulators is what removes the dependency: 0.293 ns per step, about 1.3 cycles, which is
close to the machine's load-issue limit of two 128-bit loads per step. So the 10x is two
separate wins stacked — 4x from the lanes, ~2.6x from breaking a serial dependency that
the scalar version was never allowed to break.

**The one-accumulator kernel does not show that penalty at d=128** (0.400 ns per step
there against 0.766 at d=960, for the same compiled loop). I do not have a confident
explanation; the plausible ones are different JIT unrolling decisions in the two forks —
JMH runs each parameter in its own JVM, so the two are separately profiled and separately
compiled — or the shorter loop being better covered by out-of-order execution across the
call boundary. It is recorded here as measured rather than explained away, and it does not
affect the choice of kernel: the four-accumulator version is the only one whose per-step
cost does not degrade with dimension, so it is the one the indexes use.

**A sequential scan is compute-bound, not bandwidth-bound — even from DRAM.** The
64 MiB block cannot be in any cache on this machine, and the SIMD scan still runs at
75.2 M vec/s at d=128, which is the same 75.2 M vec/s the L1-resident pair benchmark
gives. The hardware prefetcher sees a perfectly linear stride and hides the entire memory
latency; the kernel never waits. This is worth knowing before optimising an index: for
SIFT-shaped data, making the distance calculation faster helps, and making the memory
access pattern nicer does not, because the memory system was never the constraint.

At d=960 that stops being quite true. The SIMD scan reaches 49.6 GB/s from DRAM against
14.2 M vec/s in L1 versus 12.9 M vec/s from RAM — a 9% loss, so at 960 dimensions the
kernel has become fast enough to start feeling the memory system. That is the first hint
of what Phase 6 is about: the high-dimensional case is not merely "the same thing but
slower", it is a different bottleneck.

**Inner product is marginally cheaper than L2** (12.3 vs 13.3 ns at d=128): it is one FMA
per step where L2 needs a subtract and an FMA. Both metrics are implemented, but every
number in this project uses L2, because that is the metric SIFT1M and GIST1M are
distributed with ground truth for.

## Correctness

The SIMD kernels are tested against the scalar ones in `DistanceKernelTest` across
dimensions that are not multiples of the lane count or of the unroll factor (1, 3, 13, 17,
31, 33, 63, 65, 129, 961), at non-zero offsets into a flat array, and on SIFT-shaped
integer-valued inputs.

They do **not** agree bit for bit, and are not expected to. FMA rounds once where a
separate multiply and add round twice, and four accumulators sum in a different order from
one. The tolerance is relative, and for inner product it is scaled by the sum of the term
magnitudes rather than by the result — a sign-mixed inner product can cancel down to near
zero, at which point an error that is negligible next to the work done is enormous next to
the answer. Squared L2 needs no such treatment because every term is non-negative and
nothing cancels.

On SIFT specifically the question does not arise: components are integers in [0, 255], so
every squared distance is an integer below 2^24 and float32 represents it exactly. Both
kernels are exact on that dataset, which is also why the tie discussed in `PROTOCOL.md` §1
is a real tie and not a rounding artefact.
