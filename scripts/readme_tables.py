#!/usr/bin/env python3
"""Generate every sweep-derived number in README.md from the committed CSVs.

The README tables were once hand-written and drifted away from the data they
described. Nothing numeric in README.md is typed by hand any more: this script owns
the regions between

    <!-- BEGIN GENERATED: name -->   ...   <!-- END GENERATED: name -->

and rewrites them from docs/results/. Prose outside those markers is untouched.

    python3 scripts/readme_tables.py            # rewrite README.md in place
    python3 scripts/readme_tables.py --check    # exit 1 if it would change (for CI)
    python3 scripts/readme_tables.py --audit    # print the suspect-row evidence

SUSPECT ROWS
------------
The FAISS GIST1M sweep contains a measurement artefact: partway through the run the
machine slowed, inflating FAISS's wall clock in both build and search. Those rows are
not deleted — they are marked, and every generated range says whether it used them.
See `suspect_blocks` for the detection rule.
"""
import argparse
import csv
import os
import re
import math
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'docs', 'results')
README = os.path.join(ROOT, 'README.md')
ANALYSIS = os.path.join(ROOT, 'docs', 'analysis.md')

MIB = 1024.0 * 1024.0

# ---------------------------------------------------------------- loading


def read(name):
    with open(os.path.join(RESULTS, name)) as handle:
        return list(csv.DictReader(handle))


def is_hnsw(row):
    return row['index'].startswith('hnsw') or row['index'] == 'IndexHNSWFlat'


def is_ivfpq(row):
    return row['index'].startswith('ivfpq') or row['index'] == 'IndexIVFPQ'


def hnsw_key(row):
    """(M, efConstruction, efSearch) parsed out of the params string."""
    p = row['params']
    return (int(re.search(r'\bM=(\d+)', p).group(1)),
            int(re.search(r'efC=(\d+)', p).group(1)),
            int(re.search(r'ef=(\d+)$', p).group(1)))


def block_key(row):
    """The (M, efC) build this search row was measured on."""
    return hnsw_key(row)[:2]


class Sweep:
    """One dataset's four result sets, indexed for lookup."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.java = read(f'java-{dataset}.csv')
        self.faiss = read(f'faiss-{dataset}.csv')
        self.jh = {r['params']: r for r in self.java if is_hnsw(r)}
        self.fh = {r['params']: r for r in self.faiss if is_hnsw(r)}
        self.jp = {r['params']: r for r in self.java if is_ivfpq(r)}
        self.fp = {r['params']: r for r in self.faiss if is_ivfpq(r)}
        self.suspect = suspect_blocks(self)
        self.cutoff = suspect_cutoff(self)

    def hnsw_pairs(self, skip_suspect=False):
        for params, j in self.jh.items():
            f = self.fh.get(params)
            if f is None:
                continue
            if skip_suspect and is_suspect(self, f):
                continue
            yield params, j, f

    def ivfpq_pairs(self, skip_suspect=False):
        for params, j in self.jp.items():
            f = self.fp.get(params)
            if f is None:
                continue
            if skip_suspect and is_suspect(self, f):
                continue
            yield params, j, f

    def n_suspect(self):
        return sum(1 for _, _, f in self.hnsw_pairs() if is_suspect(self, f))

    def ivfpq_all_suspect(self):
        pairs = list(self.ivfpq_pairs())
        return bool(pairs) and all(is_suspect(self, f) for _, _, f in pairs)


def us(row):
    return float(row['mean_latency_us'])


def recall(row):
    return float(row['recall_at_k'])


def build_s(row):
    return float(row['build_seconds'])


def mib(row):
    return int(row['index_bytes']) / MIB


# ---------------------------------------------------------------- suspect rows

#: A build block is flagged when the FAISS/Java *build-time* ratio for that block
#: departs from the sweep's median by more than this factor. Build and search are
#: separate code paths, so a build-time step that lands on the same block as a
#: search-time step is evidence about the machine rather than about either
#: implementation. 1.25 separates the GIST1M artefact cleanly and flags nothing on
#: SIFT1M, where the ratio holds within 1.09-1.23 across all nine blocks.
SUSPECT_THRESHOLD = 1.25


def suspect_cutoff(sweep):
    """Earliest FAISS timestamp inside a flagged block, or None.

    Once the machine has slowed, everything measured afterwards is affected -
    including the IVF-PQ family, which the block detector above never sees because
    it has no (M, efC) grid. So the flag is promoted to a wall-clock cutoff and
    applied to every FAISS row at or after it.
    """
    stamps = [f['timestamp'] for _, _, f in sweep.hnsw_pairs()
              if block_key(f) in sweep.suspect]
    return min(stamps) if stamps else None


def is_suspect(sweep, row):
    """True if this FAISS row was measured after the machine slowed."""
    return (sweep.cutoff is not None
            and row['harness'] == 'faiss'
            and row['timestamp'] >= sweep.cutoff)


def suspect_blocks(sweep):
    """(M, efC) build blocks whose FAISS timings look machine-affected."""
    ratios = {}
    for params, j, f in sweep.hnsw_pairs():
        ratios.setdefault(block_key(j), build_s(f) / build_s(j))
    if not ratios:
        return set()
    median = statistics.median(ratios.values())
    return {k for k, v in ratios.items()
            if max(v / median, median / v) > SUSPECT_THRESHOLD}


# ---------------------------------------------------------------- helpers


def cheapest(rows, threshold):
    """Lowest-latency row reaching `threshold` recall, or None."""
    ok = [r for r in rows if recall(r) >= threshold]
    return min(ok, key=us) if ok else None


def best_recall(rows):
    return max(rows, key=recall)


def span(values, fmt='{:.2f}'):
    lo, hi = min(values), max(values)
    return fmt.format(lo) + '–' + fmt.format(hi)


def fnum(value, digits=0):
    return f'{value:,.{digits}f}'


# ---------------------------------------------------------------- blocks

def block_headline(sift, gist):
    target = cheapest([r for r in sift.jh.values()], 0.99)
    faiss_same = sift.fh[target['params']]
    pairs = list(sift.hnsw_pairs())
    wins = sum(1 for _, j, f in pairs if us(j) < us(f))
    ndis = read_ndis()
    lo = min(j / f for _, f, j in ndis)
    hi = max(j / f for _, f, j in ndis)
    clean = [us(j) / us(f) for _, j, f in gist.hnsw_pairs(skip_suspect=True)]
    return f"""The headline: on SIFT1M this HNSW reaches **recall@10 of {recall(target):.4f} in \
{us(target):.0f} µs per query**
single-threaded — matching the exact answer on {recall(target) * 100:.1f}% of neighbours over a \
million vectors —
and beats `IndexHNSWFlat` on latency at all {wins} swept configurations \
({fnum(us(faiss_same), 0)} µs against {fnum(us(target), 0)} at
the same settings) while computing {(lo - 1) * 100:.1f}–{(hi - 1) * 100:.1f}% *more* distances. On GIST1M \
at 960 dimensions the advantage
reverses: on the {len(clean)} configurations measured before the machine artefact described below, \
this
implementation is {span(clean, '{:.2f}')}x FAISS's latency. The more useful result is why: \
**this implementation's
bookkeeping is faster and its distance kernel is slower**, so which one wins depends on how much
arithmetic sits behind each candidate."""


def block_memory(sift):
    pq = [mib(r) for r in sift.jp.values()]
    hn = [mib(r) for r in sift.jh.values()]
    raw = int(next(iter(sift.jh.values()))['base_bytes']) / MIB
    agree = max(abs(mib(j) - mib(f)) / mib(f) for _, j, f in sift.hnsw_pairs()) * 100
    return f"""The families barely compete on memory. IVF-PQ spans {min(pq):.1f}–{max(pq):.1f} MiB; \
HNSW spans {min(hn):.0f}–{max(hn):.0f} MiB **and
needs the {raw:.0f} MiB of raw vectors on top**, because it computes real distances during the
search. The two HNSW lines are one visible line: the implementations agree to within {math.ceil(agree * 10) / 10:.1f}%."""


def block_build(sift, gist):
    def block_ratios(sweep, which):
        out = {}
        pairs = sweep.hnsw_pairs() if which == 'hnsw' else sweep.ivfpq_pairs()
        for _, j, f in pairs:
            key = j['params'].rsplit(',ef=', 1)[0].rsplit(',nprobe=', 1)[0]
            out.setdefault(key, (build_s(j), build_s(f)))
        return out

    sh = block_ratios(sift, 'hnsw')
    sp = block_ratios(sift, 'ivfpq')
    hnsw = span([f / j for j, f in sh.values()])
    pq = span([j / f for j, f in sp.values()], '{:.1f}')
    return f"""One point per built index — a scatter, since joining builds that differ in `M` or \
`nlist`
would draw a trajectory nothing travels along. The two implementations diverge here most
sharply and in opposite directions: on SIFT1M, HNSW builds **{hnsw}x faster**, IVF-PQ builds
**{pq}x slower**."""


def results_table(sweep, targets, note='', brute=None):
    lines = ['| target | index | configuration | recall@10 | mean latency | FAISS | index size |',
             '|---|---|---|---:|---:|---:|---:|']
    for threshold in targets:
        j = cheapest(list(sweep.jh.values()), threshold)
        f = cheapest(list(sweep.fh.values()), threshold)
        if j is None:
            continue
        lines.append(row_line(f'≥{threshold:.2f}', 'HNSW', sweep, j, f))
    for label, table_j, table_f in (('max', sweep.jh, sweep.fh), ('max', sweep.jp, sweep.fp)):
        j = best_recall(list(table_j.values()))
        f = best_recall(list(table_f.values()))
        kind = 'HNSW' if table_j is sweep.jh else 'IVF-PQ'
        lines.append(row_line(label, kind, sweep, j, f))
    if brute is not None:
        rec, secs, nq, threads = brute
        ms = secs / nq * 1000
        lines.append(f'| \u2014 | exact brute force | scalar kernel, {threads} threads | '
                     f'{rec:.4f} | ~{ms:.0f} ms | \u2014 | \u2014 |')
        note += (f'\n\nThe brute-force row is the Checkpoint 1 oracle: {secs:.0f} s for {nq:,} '
                 f'queries on {threads} threads. Every other row in this table is '
                 f'single-threaded, so it is **not** a like-for-like latency.')
    return '\n'.join(lines) + note


def row_line(label, kind, sweep, j, f):
    """One results row. The FAISS cell names its own configuration when it differs."""
    faiss_cell = f'{us(f):,.0f} µs'
    if f['params'] != j['params']:
        faiss_cell += f'<br><sub>{f["params"]}, r={recall(f):.4f}</sub>'
    tainted = is_suspect(sweep, f)
    if tainted:
        faiss_cell += ' †'
    # An inflated FAISS row cannot be used to declare a win for *this* project. It can
    # still show a FAISS win, which the inflation only understates.
    jw = '**' if us(j) < us(f) and not tainted else ''
    fw = '**' if us(f) < us(j) else ''
    return (f'| {label} | {kind} | {j["params"]} | {recall(j):.4f} | '
            f'{jw}{us(j):,.0f} µs{jw} | {fw}{faiss_cell}{fw} | {mib(j):.1f} MiB |')


def block_results_sift(sift):
    return ('**SIFT1M** — 1M × 128, the full 10,000-query set, k=10, single-threaded.\n\n'
            + results_table(sift, (0.90, 0.95, 0.99), brute=oracle('sift1m')))


def block_results_gist(gist):
    note = ('\n\n† FAISS row from a build block affected by the measurement artefact — see '
            '[Limitations](#limitations). Its latency is inflated, so this comparison '
            'understates FAISS.')
    return ('**GIST1M** — 1M × 960, **the full 1,000-query set shipped with GIST1M**, k=10,\n'
            'single-threaded. (SIFT1M ships 10,000 queries and GIST1M ships 1,000; both tables use\n'
            'the whole shipped set, and both sides of each comparison use the same one.)\n\n'
            + results_table(gist, (0.90, 0.95), note))


def block_ceiling(sift, gist):
    jb = best_recall(list(sift.jh.values()))
    fb = best_recall(list(sift.fh.values()))
    sc, gc = ceilings()
    gap = (sc - recall(jb)) * 100000
    return f"""**{sc:.6f} and {gc:.6f} are the ceilings, not 1.0.** The datasets contain vectors \
equidistant
from a query, so the top-10 *ids* are not unique — see below. This HNSW reaches \
{recall(jb):.6f},
{gap:.0f} slots in 100,000 short of it. FAISS reaches {recall(fb):.6f} — *above* the figure
PROTOCOL.md §1 calls a hard bound, which is itself a result: see [Limitations](#limitations)."""


def block_faiss_checks(sift):
    sizes = {}
    for _, j, f in sift.hnsw_pairs():
        m = hnsw_key(j)[0]
        sizes.setdefault(m, (mib(j), mib(f)))
    m16j, m16f = sizes[16]
    worst = max(abs(a - b) / b * 100 for a, b in sizes.values())
    ndis = read_ndis()
    lo_ef, lo_f, lo_j = ndis[0]
    hi_ef, hi_f, hi_j = ndis[-1]
    pct = [(j / f - 1) * 100 for _, f, j in ndis]
    return f"""1. **The graphs are the same size.** Corrected for the `IndexFlat` that `IndexHNSWFlat`
   embeds, FAISS's graph is {m16f:.1f} MiB at M=16 against this project's {m16j:.1f} — \
{abs(m16j - m16f) / m16f * 100:.1f}% apart, and
   within {math.ceil(worst * 10) / 10:.1f}% at every M.
2. **The searches do the same work.** `hnsw_stats.ndis` per query at M=16/efC=200: \
{fnum(lo_j)} vs
   {fnum(lo_f)} at ef={lo_ef}, {fnum(hi_j)} vs {fnum(hi_f)} at ef={hi_ef}. \
**This implementation computes {min(pct):.1f}–{max(pct):.1f}% *more*
   distances and is still faster** — and that also explains its slightly higher recall.
3. **FAISS is not a crippled build.** The wheel reports `OPTIMIZE DD ARM_NEON MAC_METAL`
   with the full ASIMD instruction set."""


def block_ratio_gist(gist):
    cells = {}
    for _, j, f in gist.hnsw_pairs():
        m, efc, ef = hnsw_key(j)
        cells.setdefault((ef, m), {})[efc] = (us(j) / us(f), is_suspect(gist, f))
    lines = ['| ef | M=8 | M=16 | M=32 |', '|---:|---:|---:|---:|']
    for ef in (16, 64, 512):
        out = []
        for m in (8, 16, 32):
            vals = cells[(ef, m)]
            clean = [v for v, s_ in vals.values() if not s_]
            dirty = [v for v, s_ in vals.values() if s_]
            if not clean:
                out.append(f'*({statistics.mean(dirty):.2f})* \u2020')
            elif dirty:
                out.append(f'{statistics.mean(clean):.2f} \u2020')
            else:
                out.append(f'{statistics.mean(clean):.2f}')
        lines.append(f'| {ef} | ' + ' | '.join(out) + ' |')
    return ('\n'.join(lines) + f"""

Mean over the three `efConstruction` values, **counting only rows the measurement artefact
did not touch**. \u2020 marks a cell that had to drop at least one row; a cell in
*(parentheses)* had no clean rows at all and is shown for reference only, not relied on.
{gist.n_suspect()} of {len(list(gist.hnsw_pairs()))} GIST1M configurations are affected, and **every configuration in which this
project beats FAISS on GIST1M is one of them** \u2014 see [Limitations](#limitations).""")


def block_ratio_sift(sift, gist):
    s = [us(j) / us(f) for _, j, f in sift.hnsw_pairs()]
    clean = [us(j) / us(f) for _, j, f in gist.hnsw_pairs(skip_suspect=True)]
    return f"""On SIFT1M the same ratio runs {span(s)} across all {len(s)} configurations — this
project is faster everywhere. On GIST1M, restricted to the {len(clean)} configurations the artefact
did not touch, it runs {span(clean)} — FAISS is faster everywhere. The two datasets bracket
the crossover rather than contradicting each other. Two gradients hold across both, and a
third runs opposite to what this README once claimed:"""


def block_analysis_ratio(gist, sift):
    """docs/analysis.md section 6 - same data, that document's wording."""
    table = block_ratio_gist(gist).split('\n\n')[0].replace('| ef |', '| GIST1M, ef |', 1)
    s_r = [us(j) / us(f) for _, j, f in sift.hnsw_pairs()]
    clean = [us(j) / us(f) for _, j, f in gist.hnsw_pairs(skip_suspect=True)]
    return f"""{table}

Mean over the three `efConstruction` values, counting only rows the measurement artefact
described in [README.md](../README.md#limitations) did not touch. \u2020 marks a cell that had
to drop at least one row; a cell in *(parentheses)* had no clean rows at all. {gist.n_suspect()} of {len(list(gist.hnsw_pairs()))} GIST1M
configurations are affected, and every configuration in which this project beats FAISS on GIST1M
is one of them \u2014 so **no GIST1M cell below 1.0 here is trustworthy**, and the M=32 column in
particular cannot support the claim it was once used for.

On SIFT1M the same ratio runs {span(s_r)} across all {len(s_r)} configurations, and on the
{len(clean)} unaffected GIST1M configurations it runs {span(clean)}. Two consistent gradients and one reversal
explain both tables at once \u2014 the third having reversed once the affected rows were
excluded:"""


def gradient_stats(sweep, skip_suspect):
    """Mean latency ratio grouped by M and by efSearch, over usable rows."""
    by_m, by_ef = {}, {}
    for _, j, f in sweep.hnsw_pairs(skip_suspect=skip_suspect):
        m, _efc, ef = hnsw_key(j)
        by_m.setdefault(m, []).append(us(j) / us(f))
        by_ef.setdefault(ef, []).append(us(j) / us(f))
    mean = lambda d: {k: statistics.mean(v) for k, v in sorted(d.items())}
    return mean(by_m), mean(by_ef)


def block_gradients(sift, gist):
    """The gradient bullets, with each direction taken from the data rather than asserted."""
    sm, se = gradient_stats(sift, False)
    gm, ge = gradient_stats(gist, True)
    s_dim, g_dim = statistics.mean(sm.values()), statistics.mean(gm.values())
    ef_lo, ef_hi = min(se), max(se)
    gef_lo, gef_hi = min(ge), max(ge)
    m_lo, m_hi = min(sm), max(sm)
    gm_lo, gm_hi = min(gm), max(gm)
    return f"""* **\u2191 dimension \u2192 FAISS gains.** More arithmetic per candidate, bookkeeping unchanged.
  Mean ratio {s_dim:.2f} on SIFT1M against {g_dim:.2f} on GIST1M's unaffected rows. The distance
  kernel is theirs.
* **\u2191 efSearch \u2192 this project gains.** More candidates through the visited stamps and the two
  heaps, which is the bookkeeping steps 2 and 3 bought. Mean ratio falls from {se[ef_lo]:.2f} at
  ef={ef_lo} to {se[ef_hi]:.2f} at ef={ef_hi} on SIFT1M, and from {ge[gef_lo]:.2f} to {ge[gef_hi]:.2f} on GIST1M.
* **\u2191 M \u2192 FAISS gains \u2014 the opposite of what this README used to claim.** The earlier
  text argued that a higher degree gives the step-5 software prefetch more to issue at once, and
  read a win at M=32 on GIST1M as confirmation. Those M=32 rows turned out to be the measurement
  artefact. On data the artefact never touched the gradient runs the other way: mean ratio
  {sm[m_lo]:.2f} \u2192 {sm[m_hi]:.2f} from M={m_lo} to M={m_hi} on SIFT1M, and {gm[gm_lo]:.2f} \u2192 {gm[gm_hi]:.2f} from M={gm_lo} to
  M={gm_hi} on GIST1M. Whatever the prefetch buys against this project's own earlier versions, it
  does not show up as a gain against FAISS as degree rises."""


def block_opt_table():
    steps = [('0. naive reference', 'hnsw-opt-step0-naive.csv'),
             ('1. flat `int[]` arenas', 'hnsw-opt-step1-arenas.csv'),
             ('2. versioned visited stamps', 'hnsw-opt-step2-visited.csv'),
             ('3. primitive `long[]` heaps', 'hnsw-opt-step3-primitive-heaps.csv'),
             ('4. split traversal / distances', 'hnsw-opt-step4-split-traversal.csv'),
             ('5. software prefetch', 'hnsw-opt-step5-prefetch.csv')]
    lines = ['| change | p95 @ ef=64 | build |', '|---|---:|---:|']
    first = last = None
    for label, name in steps:
        row = [r for r in read(name) if r['params'].endswith('ef=64')][0]
        p95, build = float(row['p95_latency_us']), build_s(row)
        if first is None:
            first = (p95, build)
        last = (p95, build)
        bold = '**' if name.endswith('prefetch.csv') else ''
        lines.append(f'| {label} | {bold}{p95:.1f} µs{bold} | {bold}{build:.1f} s{bold} |')
    lines.append('')
    lines.append(f'Cumulative: **{first[0] / last[0]:.2f}x** on p95 at ef=64, '
                 f'**{first[1] / last[1]:.2f}x** on build.')
    return '\n'.join(lines)


def block_limits(sift, gist):
    def pq_search(sweep):
        hi = [us(j) / us(f) for p, j, f in sweep.ivfpq_pairs() if p.endswith('nprobe=64')]
        return span(hi, '{:.1f}')

    def pq_build(sweep):
        blocks = {}
        for _, j, f in sweep.ivfpq_pairs():
            blocks.setdefault(j['params'].rsplit(',nprobe=', 1)[0], (build_s(j), build_s(f)))
        return span([a / b for a, b in blocks.values()], '{:.1f}')

    def hnsw_build(sweep, skip_suspect, invert=False):
        """Span of the build-time ratio. invert=True reports java/faiss instead."""
        blocks = {}
        for _, j, f in sweep.hnsw_pairs(skip_suspect=skip_suspect):
            blocks.setdefault(block_key(j), (build_s(j), build_s(f)))
        vals = [(a / b if invert else b / a) for a, b in blocks.values()]
        return span(vals)

    sc, gc = ceilings()
    fb = best_recall(list(sift.fh.values()))
    n_susp, n_all = gist.n_suspect(), len(list(gist.hnsw_pairs()))
    n_clean = len(list(gist.hnsw_pairs(skip_suspect=True)))
    pq_caveat = ('  Every FAISS GIST1M IVF-PQ row was measured after the step, so both GIST1M\n'
                 '  figures here are **lower bounds on the gap**: the artefact inflates FAISS,\n'
                 '  which flatters this implementation.'
                 if gist.ivfpq_all_suspect() else '')
    return f"""* **The metric's ceiling is {sc:.6f} on SIFT and {gc:.6f} on GIST**, because ids are not
  unique under ties. The searches themselves are exact. PROTOCOL.md \u00a71 calls this a hard
  bound that "no configuration can do better" than, and **that is wrong**: FAISS scores
  {recall(fb):.6f} on SIFT1M, one slot in 100,000 above it. The figure is the tie-breaking
  this project's oracle happened to choose, not an upper bound on any index. The protocol is
  frozen, so the error is recorded here rather than edited there.
* **The FAISS GIST1M sweep contains a measurement artefact, and it favours this project.**
  Partway through the run \u2014 between the `M=16,efC=200` and `M=16,efC=400` builds, around
  03:30\u201304:00 on 5 Sep \u2014 FAISS's wall clock steps up by roughly 1.5\u20132x and stays there.
  Search latency and *build* time step together on the same block, and the two share no code,
  so the cause is the machine rather than either implementation. {n_susp} of {n_all} GIST1M HNSW
  configurations and **every** GIST1M IVF-PQ configuration sit after the step; they are marked
  \u2020 and excluded from every GIST1M range quoted above. **Every configuration in which this
  project beat FAISS on GIST1M lies in the affected region**, so the earlier claim that this
  implementation is level or ahead at M=32 on GIST1M is withdrawn \u2014 on the {n_clean} unaffected
  configurations FAISS is faster in all of them. Re-running the four affected builds is the fix
  and has not been done; until it is, no GIST1M latency comparison here should be read as
  favouring this project.
* **Single-threaded throughout**, build and search, on both sides. It is the only setting in
  which "mine took X and FAISS took Y" means anything, but it is not how either would be
  deployed, and it excludes FAISS's batched search paths entirely.
* **Results are aarch64-specific** and would plausibly reverse on AVX-512.
* **IVF-PQ search is {pq_search(sift)}x slower than FAISS** at `nprobe`=64 on SIFT1M
  ({pq_search(gist)}x on GIST1M), localised to the list scan at ~1.8 cycles per table lookup
  against a load-throughput limit nearer 1.1.
{pq_caveat}
* **IVF-PQ build is {pq_build(sift)}x slower on SIFT1M and {pq_build(gist)}x slower on GIST1M**,
  and the cause is understood: closing it needs a GEMM microkernel with register-level tiling
  written against the Vector API, since rule 4 forbids linking a BLAS. Not attempted.
* **HNSW build is {hnsw_build(sift, False)}x faster than FAISS on SIFT1M, but
  {hnsw_build(gist, True, invert=True)}x *slower* on GIST1M** (unaffected blocks only) \u2014 the same
  dimension gradient that governs the search comparison, showing up in construction as well.
* **No OPQ rotation**, so the product quantizer assumes the subspaces are uncorrelated. GIST
  shows what that costs. A learned rotation is the standard fix and is the single most
  valuable thing missing here \u2014 plausibly enough to clear 0.80 recall at m=32, which would
  satisfy both halves of the Checkpoint 4 target honestly rather than by redefining it.
* **`m` must divide the dimension in this implementation**, so code sizes are 8/16/32/64/128
  bytes and nothing between \u2014 which is why the Checkpoint 4 frontier has no point to land on
  between 32 and 64. FAISS pads the last subvector and accepts any `m`.
* **Deletion, updates and persistence are unimplemented.** This indexes a fixed set once."""


# ---------------------------------------------------------------- side inputs


def read_ndis():
    """(ef, faiss_ndis, java_ndis) from the committed comparison output."""
    out = []
    path = os.path.join(RESULTS, 'hnsw-ndis-java-vs-faiss.txt')
    with open(path) as handle:
        for line in handle:
            m = re.match(r'\s*(\d+)\s+([\d,]+)\s+[\d,]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s*\|'
                         r'\s*([\d,]+)', line)
            if m:
                out.append((int(m.group(1)),
                            int(m.group(2).replace(',', '')),
                            int(m.group(3).replace(',', ''))))
    if not out:
        raise SystemExit('could not parse hnsw-ndis-java-vs-faiss.txt')
    return out


def oracle(dataset):
    """(recall, wall_clock_s, queries, threads) from a committed oracle run."""
    text = open(os.path.join(RESULTS, f'checkpoint1-oracle-{dataset}.txt')).read()
    rec = float(re.search(r'recall@10\s*:\s*([\d.]+)', text).group(1))
    secs = float(re.search(r'done\s*:\s*([\d.]+)s', text).group(1))
    scan = re.search(r'scanning\s*:\s*([\d,]+) queries.*?(\d+) threads', text)
    return rec, secs, int(scan.group(1).replace(',', '')), int(scan.group(2))


def ceilings():
    """The oracle's recall@10 against the shipped ids, per dataset."""
    out = []
    for name in ('checkpoint1-oracle-sift1m.txt', 'checkpoint1-oracle-gist1m.txt'):
        text = open(os.path.join(RESULTS, name)).read()
        out.append(float(re.search(r'recall@10\s*:\s*([\d.]+)', text).group(1)))
    return tuple(out)


# ---------------------------------------------------------------- rewriting


def build_blocks():
    sift, gist = Sweep('sift1m'), Sweep('gist1m')
    return {ANALYSIS: {'analysis-ratio': block_analysis_ratio(gist, sift),
                      'analysis-gradients': block_gradients(sift, gist)}, README: {
        'headline': block_headline(sift, gist),
        'memory-spans': block_memory(sift),
        'build-divergence': block_build(sift, gist),
        'results-sift': block_results_sift(sift),
        'results-gist': block_results_gist(gist),
        'ceiling': block_ceiling(sift, gist),
        'faiss-checks': block_faiss_checks(sift),
        'ratio-gist': block_ratio_gist(gist),
        'ratio-sift': block_ratio_sift(sift, gist),
        'gradients': block_gradients(sift, gist),
        'opt-table': block_opt_table(),
        'limitations': block_limits(sift, gist),
    }}


def apply_blocks(text, blocks, label):
    missing = []
    for name, body in blocks.items():
        pattern = re.compile(
            r'(<!-- BEGIN GENERATED: ' + re.escape(name) + r' -->\n).*?'
            r'(\n<!-- END GENERATED: ' + re.escape(name) + r' -->)',
            re.DOTALL)
        if not pattern.search(text):
            missing.append(name)
            continue
        text = pattern.sub(lambda m: m.group(1) + body + m.group(2), text, count=1)
    if missing:
        raise SystemExit(f'{label} has no markers for: ' + ', '.join(missing))
    return text


def audit():
    for name in ('sift1m', 'gist1m'):
        sweep = Sweep(name)
        ratios = {}
        for _, j, f in sweep.hnsw_pairs():
            ratios.setdefault(block_key(j), (build_s(j), build_s(f)))
        median = statistics.median(f / j for j, f in ratios.values())
        print(f'=== {name}: FAISS/Java build-time ratio per block (median {median:.2f}, '
              f'flag beyond {SUSPECT_THRESHOLD}x)')
        for key in sorted(ratios):
            j, f = ratios[key]
            r = f / j
            dev = max(r / median, median / r)
            print(f'    M={key[0]:2} efC={key[1]:3}  java {j:8.1f}s  faiss {f:8.1f}s  '
                  f'ratio {r:5.2f}  dev {dev:4.2f}  '
                  f'{"SUSPECT" if key in sweep.suspect else ""}')
        pairs = list(sweep.hnsw_pairs())
        wins = [(p, j) for p, j, f in pairs if us(j) < us(f)]
        in_susp = sum(1 for p, j in wins if block_key(j) in sweep.suspect)
        print(f'    java-faster configurations: {len(wins)}/{len(pairs)}, '
              f'{in_susp} of them in suspect blocks\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='exit 1 if README would change')
    ap.add_argument('--audit', action='store_true', help='print suspect-row evidence')
    args = ap.parse_args()

    if args.audit:
        audit()
        return 0

    stale = 0
    written = []
    for path, blocks in build_blocks().items():
        name = os.path.relpath(path, ROOT)
        original = open(path).read()
        updated = apply_blocks(original, blocks, name)
        if args.check:
            if original != updated:
                print(f'{name} is out of date; run scripts/readme_tables.py', file=sys.stderr)
                stale += 1
            continue
        if original != updated:
            with open(path, 'w') as handle:
                handle.write(updated)
        written.append(f'{len(blocks)} blocks in {name}')
    if args.check:
        if stale:
            return 1
        print('README.md and docs/analysis.md are up to date with docs/results/.')
        return 0
    print('regenerated ' + ', '.join(written))
    return 0


if __name__ == '__main__':
    sys.exit(main())
