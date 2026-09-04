# HNSW optimization

Checkpoint 3b. Every row is the same index at the same settings — SIFT1M, M=16,
efConstruction=200, k=10, the full 10,000-query set, three measured passes after a
discarded warm-up, single-threaded — so the only thing that changes between rows is the
implementation.

Reproduce any row with:

```
./gradlew run --args="hnsw --impl fast --m 16 --efc 200 --ef 16,32,64,128,256,512 --runs 3"
python3 scripts/opt_table.py docs/results/hnsw-opt-step*.csv
```

## The rule this table depends on

**Recall is identical to six decimal places in every row, at every one of the six
`efSearch` values, and the edge count is identical to the digit.** That is not a happy
accident, it is enforced. `HnswIndex` (the naive reference) and `FastHnswIndex` draw
levels from the same seeded RNG in the same order and order candidates by the same total
`(distance, id)` comparator, and `FastHnswIndexTest` asserts that they build a
**byte-identical graph** and return byte-identical results for every query.

Without that, a "35% faster" row could just as easily be a slightly worse graph that
searches less thoroughly, and the whole table would mean nothing. The tie-break on id in
the packed representation exists for this reason as much as for speed: without a total
order, two implementations can make different arbitrary choices and diverge.

### p95 latency per query, microseconds

| change | ef=16 | ef=32 | ef=64 | ef=128 | ef=256 | ef=512 |
|---| ---: | ---: | ---: | ---: | ---: | ---: |
| 0. naive reference | 163.8 | 261.8 | 447.5 | 782.2 | 1,479.3 | 2,810.2 |
| 1. flat int[] arenas | 131.6  (-20%) | 211.1  (-19%) | 357.0  (-20%) | 633.6  (-19%) | 1,159.5  (-22%) | 2,290.9  (-18%) |
| 2. versioned visited stamps | 66.6  (-49%) | 105.1  (-50%) | 182.3  (-49%) | 337.2  (-47%) | 627.9  (-46%) | 1,163.2  (-49%) |
| 3. primitive long[] heaps | 61.1  (-8%) | 94.7  (-10%) | 160.4  (-12%) | 289.8  (-14%) | 532.8  (-15%) | 969.0  (-17%) |
| 4. split traversal / distances | 61.2  (=) | 94.0  (=) | 159.2  (=) | 288.0  (=) | 525.3  (-1%) | 964.5  (=) |
| 5. software prefetch | 46.2  (-24%) | 69.6  (-26%) | 117.0  (-27%) | 211.8  (-26%) | 384.5  (-27%) | 694.5  (-28%) |

### Build time and memory

| change | build (s) | index (MiB) | recall@10 at ef=64 |
|---| ---: | ---: | ---: |
| 0. naive reference | 779.3 | 494.8 | 0.9704 |
| 1. flat int[] arenas | 603.7  (-23%) | 137.9 | 0.9704 |
| 2. versioned visited stamps | 360.4  (-40%) | 137.9 | 0.9704 |
| 3. primitive long[] heaps | 282.6  (-22%) | 137.9 | 0.9704 |
| 4. split traversal / distances | 283.1  (=) | 137.9 | 0.9704 |
| 5. software prefetch | 220.2  (-22%) | 137.9 | 0.9704 |

### Cumulative

* p95 at ef=64: 447.5 us -> 117.0 us  (**3.83x**)
* p95 at ef=512: 2,810.2 us -> 694.5 us  (**4.05x**)
* build: 779.3 s -> 220.2 s  (**3.54x**)
* index memory: 494.8 MiB -> 137.9 MiB  (**3.59x**)

## What each change was, and what it actually did

### 1. Flat `int[]` arenas — the one I expected to be biggest, and wasn't

Neighbour lists were `ArrayList<Integer>`, one per node per layer. Reading a single edge
meant loading an `ArrayList`, then its backing `Object[]`, then an `Integer` box, then the
`int` inside it: four dependent loads on four unrelated cache lines to deliver four bytes.
The replacement is one fixed-stride `int[]` for layer 0, addressed arithmetically as
`node * stride` so finding a node's edges is a multiply rather than a load, with the whole
list arriving in one or two cache lines; the sparser upper layers share a packed arena
with a per-node offset.

**−20% on p95, −23% on build, and 3.6x less memory.** Real, but far short of what the
description suggests, and the arithmetic says why. The naive build spent 779 µs per
insertion, of which the distance computations account for about 55 µs. Of the remaining
724 µs, the neighbour-list pointer chase turned out to be roughly a fifth. The object
heaps were most of the rest.

The memory number is the unambiguous part: 494.8 MiB to 137.9 MiB, for a graph with
23.4 million directed edges. The boxed representation was spending about 21 bytes per
edge to store 4 bytes of payload.

### 2. Versioned visited stamps — the one that was biggest

`HashSet<Integer>`, allocated fresh per layer per query, became one `int` per node holding
the number of the search that last touched it. Membership is an array read and a compare;
clearing the set between searches is `visitGeneration++` rather than touching a million
entries.

**−49%, uniformly across every `efSearch` value.** Half of everything left was going into
a hash set that never held more than a few thousand ids and looks entirely harmless in the
source. What makes it expensive is that its cost scales with exactly the quantity
`efSearch` controls: a query at ef=64 examines about 1,300 nodes and offers every one of
them to the set, each costing an `Integer` allocation, a hash, and a probe into a table
whose buckets are scattered across the heap.

The uniformity of the −49% across ef=16 through ef=512 is the tell that this was a
per-node cost rather than a fixed overhead.

Cost: 4 MiB of scratch at a million nodes. It is excluded from `estimatedBytes()` because
it is scratch and not index, which is why the retained-heap delta sits 4 MiB above the
analytic figure. The class documents the gap rather than quietly reporting the smaller
number.

### 3. Primitive `long[]` heaps

Both `PriorityQueue<Candidate>` instances became heaps over packed longs: the distance in
the high 32 bits, transformed so that the signed order of the longs is the order of the
distances, and the id in the low 32. A comparison is one `long` compare of values already
in registers; the heaps are `long[]`; the search loop allocates nothing at all.

**−8% at ef=16 rising to −17% at ef=512.** The gradient is the interesting part. Heap
traffic grows faster than linearly in `efSearch` — a wider beam means more candidates
offered *and* a deeper heap to sift them through — so a change that makes each heap
operation cheaper pays more the wider the beam gets. Any row in this table whose effect
varies with `efSearch` is telling you it touched something the beam width multiplies.

### 4. Split traversal from distance computation — no effect

Phase 1 collects the unvisited ids out of the arena; phase 2 computes their distances. The
intent was to let several random 512-byte vector reads be in flight at once instead of one
at a time.

**Nothing. −0.7% at ef=64, which is noise.**

The reason is that there was no stall to remove. In the original interleaved loop the
distance calls had no data dependency on one another either — only the heap update depends
on a result — so the out-of-order engine was already overlapping those loads across
iterations. The optimization was aimed at a serialisation that was not happening.

This is the row I would have reverted, and reverting it would have been a mistake. See
step 5.

### 5. Software prefetch — the one the profiler pointed at

A JFR profile of the search phase attributed **54.8%** of query time to the distance
kernel. At ef=64 a query computes 1,323 distances in a 125.3 µs query, so that 54.8% works
out to **52 ns per distance** — against the JMH figure of **13.3 ns** for the same kernel
on L1-resident data. Three quarters of the cost of a distance was the cache miss, not the
arithmetic. Each one reads 512 bytes from a random position in 512 MB of base vectors, and
unlike the sequential scan in `docs/kernels.md` — where the hardware prefetcher hides the
memory system entirely — a graph walk gives the prefetcher a pointer chase it cannot
predict.

Java has no prefetch intrinsic. What works is reading one float from each pending
neighbour's vector into a field the JIT cannot prove is dead, before the distance loop
runs. Those reads are mutually independent, so they queue in the memory system together
while the first distance is being computed.

**−27%, and −22% on build.**

And this is what step 4 was for. Splitting traversal from distance computation is what
makes the *whole set* of pending addresses known before any distance is computed, which is
the precondition for issuing their loads together. Step 4 shows no benefit in isolation
and would have looked like dead weight on its own; it is half of an optimization whose
other half arrived a step later. A table that only kept the changes that paid off
immediately would have thrown it away.

## Where the time goes now

Search-phase profile before and after step 5, from
[`docs/results/jfr-profile-before-step5.txt`](results/jfr-profile-before-step5.txt) and
[`docs/results/jfr-profile-after-step5.txt`](results/jfr-profile-after-step5.txt):

| search-phase self time | before step 5 | after step 5 |
|---| ---: | ---: |
| distance kernel (`SimdDistance` + the `Distance` facade) | 54.8% | 47.1% |
| `searchLayer` itself (arena reads, visited stamps, **and the prefetch loop**) | 13.2% | 22.1% |
| `LongMinHeap.push` (the frontier) | 16.5% | 13.4% |
| `BoundedMaxHeap` offer / siftUp / siftDown (the result beam) | 12.6% | 13.8% |
| `LongMinHeap.pop` | 1.3% | 2.0% |
| sorting the drained results | 1.4% | 1.1% |
| mean query at ef=64 | 125.3 µs | 91.9 µs |

Read the first two rows together. The kernel's share fell, but `searchLayer`'s own share
nearly doubled — because the prefetch loop lives inside `searchLayer`. The stall did not
disappear, it moved to a place where it overlaps with useful work instead of blocking the
kernel. Applying the shares to the query times, the kernel went from about 68.7 µs to
43.3 µs per query at ef=64, or from 52 ns to roughly 33 ns per distance against the
13.3 ns the same kernel costs on L1-resident data.

The heap operations are now collectively the second-largest bucket at about 29%, and they
are the obvious next target — the frontier receives a push for nearly every candidate the
beam accepts, and most of those are never popped because the search terminates first.
That is where I would go next; it is not in this table because I have not measured it.

A caveat on reading absolute microseconds out of these shares: JFR execution sampling
attributes a sample to whichever method the thread was in when the timer fired, and a
thread stalled on memory is disproportionately likely to be caught there. The shares are
sound for ranking; the derived per-method microseconds are indicative, not measurements.
The measurements are the wall-clock latencies in the table above.

## Things worth knowing that are not in the table

**A profiler can lie to you by default.** `jfr print` truncates displayed stack traces to
five frames unless given `--stack-depth`. At five frames a build sample and a search
sample are both `[kernel, searchLayer, …]` and cannot be told apart, so the first
profile I took attributed the two phases to each other. The numbers looked plausible,
which is the dangerous part. `scripts/jfr_hotspots.py` now refuses to report quietly on a
truncated recording.

**The distance counter changed what the profile meant.** Knowing that a query at ef=64
computes 1,323 distances turns "54.8% of time in the kernel" into "52 ns per distance",
which can be compared against the 13.3 ns the microbenchmark says the same kernel costs.
Without the count there is no way to tell a kernel that is slow from a kernel that is
waiting, and those want opposite fixes.

**The build is the same code as the search.** Insertion runs the same layer search a query
does, at `efConstruction` width instead of `efSearch`, so every row improves build time
and query time together and roughly in proportion. There is no separate build path to
tune.
