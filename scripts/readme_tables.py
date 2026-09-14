#!/usr/bin/env python3
"""Generate every measured number in the project's prose from the committed results.

Nothing numeric inside a generated region is typed by hand: this script owns the regions
between

    <!-- BEGIN GENERATED: name -->   ...   <!-- END GENERATED: name -->

in README.md, docs/analysis.md, docs/kernels.md, docs/results/README.md and
docs/guide/guide.html, and rewrites them from docs/results/ (the sweep CSVs, the oracle
output and the JMH JSON). Text outside those markers is untouched.

    python3 scripts/readme_tables.py            # rewrite every owned file in place
    python3 scripts/readme_tables.py --check    # exit 1 if anything would change (for CI)
    python3 scripts/readme_tables.py --audit    # print the suspect-row evidence

Where prose asserts a direction ("FAISS gains as M rises", "every win is in the affected
region"), the block checks it against the data and refuses to generate if it has stopped
holding, so the words cannot go stale while the numbers stay fresh.

SUSPECT ROWS
------------
The FAISS GIST1M sweep contains a measurement artefact: partway through the run the
machine slowed, inflating FAISS's wall clock in both build and search. Those rows are
not deleted — they are marked, and every generated range says whether it used them.
See `suspect_blocks` for the detection rule.
"""
import argparse
import csv
import datetime
import json
import os
import re
import math
import statistics
import sys

import jmh_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'docs', 'results')
README = os.path.join(ROOT, 'README.md')
ANALYSIS = os.path.join(ROOT, 'docs', 'analysis.md')
KERNELS = os.path.join(ROOT, 'docs', 'kernels.md')
RESULTS_README = os.path.join(RESULTS, 'README.md')
GUIDE = os.path.join(ROOT, 'docs', 'guide', 'guide.html')

MIB = 1024.0 * 1024.0
DAGGER = '†'
ARTEFACT_ANCHOR = 'a-timing-artefact-in-the-faiss-gist1m-sweep'

#: Clock the cycle estimates in docs/kernels.md are expressed against. Not measured
#: here; it converts nanoseconds to an approximate cycle count and is stated in the text.
CLOCK_GHZ = 4.4

#: The k-means assignment loop behind the "distances per second" figure: iterations and
#: training points, as configured for the SIFT1M IVF-PQ builds.
KMEANS_ITERATIONS = 25
KMEANS_TRAINING_POINTS = 1_000_000

#: Build time of the cache-blocked k-means experiment. The run was reverted and its CSV
#: row not kept, so this is the one figure here with no file behind it: it is quoted from
#: commit c5c4dae, and the prose says so.
BLOCKED_KMEANS_BUILD_S = 1119.8
BLOCKED_KMEANS_COMMIT = 'c5c4dae'

WORDS = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven',
         8: 'eight', 9: 'nine'}

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


def jmh(*names):
    """{(benchmark, param): score} from committed JMH JSON; later files win."""
    out = {}
    for name in names:
        with open(os.path.join(RESULTS, name)) as handle:
            for entry in json.load(handle):
                (param,) = entry['params'].values()
                bench = entry['benchmark'].rsplit('.', 1)[-1]
                out[(bench, int(param))] = entry['primaryMetric']['score']
    return out


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


def require(condition, message):
    """Refuse to generate prose whose claim the data no longer supports."""
    if not condition:
        raise SystemExit('generated prose is stale: ' + message)


def spaced(params):
    return params.replace(',', ', ')


# ---------------------------------------------------------------- shared facts


def build_blocks_of(sweep):
    """{(M, efC): {'java': s, 'faiss': s, 'first': earliest FAISS row timestamp}}."""
    blocks = {}
    for _, j, f in sweep.hnsw_pairs():
        b = blocks.setdefault(block_key(j), {'java': build_s(j), 'faiss': build_s(f),
                                             'first': f['timestamp']})
        b['first'] = min(b['first'], f['timestamp'])
    return blocks


def artefact_facts(sift, gist):
    """Everything the timing-artefact prose says, derived and checked."""
    blocks = build_blocks_of(gist)
    order = sorted(blocks, key=lambda k: blocks[k]['first'])
    clean = [k for k in order if k not in gist.suspect]
    affected = [k for k in order if k in gist.suspect]
    require(affected, 'no GIST1M block is flagged any more')
    require(order == clean + affected, 'flagged GIST1M blocks are not a contiguous tail of the run')
    first = affected[0]
    started = (datetime.datetime.fromisoformat(blocks[first]['first'])
               - datetime.timedelta(seconds=blocks[first]['faiss']))
    ratio = {k: blocks[k]['faiss'] / blocks[k]['java'] for k in order}
    median = statistics.median(ratio.values())
    sift_blocks = build_blocks_of(sift)
    require(not sift.suspect, 'SIFT1M now has flagged blocks')

    pairs = list(gist.hnsw_pairs())
    wins = [f for _, j, f in pairs if us(j) < us(f)]
    require(all(is_suspect(gist, f) for f in wins),
            'a GIST1M configuration where this project is faster lies outside the affected region')
    clean_lat = [us(j) / us(f) for _, j, f in gist.hnsw_pairs(skip_suspect=True)]
    require(min(clean_lat) > 1, 'this project is faster at an unaffected GIST1M configuration')
    lat = {}
    for _, j, f in pairs:
        lat.setdefault(block_key(j), []).append(us(j) / us(f))
    lat = {k: statistics.mean(v) for k, v in lat.items()}
    fm = first[0]
    before = [k for k in clean if k[0] == fm]
    require(before and lat[first] < min(lat[k] for k in before),
            'search latency no longer shifts at the first flagged index')
    all_affected_m = sorted(m for m in {k[0] for k in order}
                            if all(k in gist.suspect for k in order if k[0] == m))
    require(all_affected_m, 'no M value is wholly affected any more')
    return {
        'order': order, 'blocks': blocks, 'suspect': set(affected), 'ratio': ratio,
        'first': first, 'started': started, 'date': f'{started.day} {started:%b}',
        'hhmm': f'{started:%H:%M}',
        'n_clean_blocks': len(clean), 'n_affected_blocks': len(affected),
        'clean_build': span([ratio[k] for k in clean]),
        'affected_build': span([ratio[k] for k in affected]),
        'dev': span([max(ratio[k] / median, median / ratio[k]) for k in affected]),
        'sift_build': span([b['faiss'] / b['java'] for b in sift_blocks.values()]),
        'n_sift_blocks': len(sift_blocks),
        'n_wins': len(wins), 'n_all': len(pairs), 'n_suspect': gist.n_suspect(),
        'n_clean': len(clean_lat), 'clean_lat': span(clean_lat),
        'ivfpq_all': gist.ivfpq_all_suspect(),
        'whole_m': '/'.join(map(str, all_affected_m)),
        'lat_first': f'{lat[first]:.2f}', 'lat_before': span([lat[k] for k in before]),
        'n_before': len(before),
    }


def gradient_stats(sweep, skip_suspect):
    """Mean latency ratio grouped by M and by efSearch, over usable rows."""
    by_m, by_ef = {}, {}
    for _, j, f in sweep.hnsw_pairs(skip_suspect=skip_suspect):
        m, _efc, ef = hnsw_key(j)
        by_m.setdefault(m, []).append(us(j) / us(f))
        by_ef.setdefault(ef, []).append(us(j) / us(f))
    mean = lambda d: {k: statistics.mean(v) for k, v in sorted(d.items())}
    return mean(by_m), mean(by_ef)


def require_trend(means, rising, label):
    """Fail generation if a gradient the prose asserts does not hold monotonically."""
    values = [means[k] for k in sorted(means)]
    ok = all((b > a) if rising else (b < a) for a, b in zip(values, values[1:]))
    require(ok, f'{label} is no longer {"rising" if rising else "falling"}: {means}')


def gradient_facts(sift, gist):
    sm, se = gradient_stats(sift, False)
    gm, ge = gradient_stats(gist, True)
    s_dim, g_dim = statistics.mean(sm.values()), statistics.mean(gm.values())
    require(g_dim > s_dim, 'GIST1M ratio no longer exceeds SIFT1M')
    require_trend(se, False, 'SIFT1M ratio by efSearch')
    require_trend(ge, False, 'GIST1M ratio by efSearch')
    require_trend(sm, True, 'SIFT1M ratio by M')
    require_trend(gm, True, 'GIST1M ratio by M')
    return {
        's_dim': f'{s_dim:.2f}', 'g_dim': f'{g_dim:.2f}',
        'ef_lo': min(se), 'ef_hi': max(se),
        's_ef': (f'{se[min(se)]:.2f}', f'{se[max(se)]:.2f}'),
        'g_ef': (f'{ge[min(ge)]:.2f}', f'{ge[max(ge)]:.2f}'),
        's_m': ' → '.join(f'{sm[m]:.2f}' for m in sorted(sm)),
        'g_m': ' → '.join(f'{gm[m]:.2f}' for m in sorted(gm)),
        's_m_range': (min(sm), max(sm)), 'g_m_range': (min(gm), max(gm)),
        'g_missing': '/'.join(map(str, sorted(set(sm) - set(gm)))),
    }


def ratio_cells(gist):
    """{(ef, M): (text, state)} with state 'clean', 'partial' or 'none'."""
    cells = {}
    for _, j, f in gist.hnsw_pairs():
        m, efc, ef = hnsw_key(j)
        cells.setdefault((ef, m), {})[efc] = (us(j) / us(f), is_suspect(gist, f))
    out = {}
    for key, vals in cells.items():
        clean = [v for v, s_ in vals.values() if not s_]
        dirty = [v for v, s_ in vals.values() if s_]
        if not clean:
            out[key] = (f'{statistics.mean(dirty):.2f}', 'none')
        else:
            out[key] = (f'{statistics.mean(clean):.2f}', 'partial' if dirty else 'clean')
    return out


def faiss_check_facts(sift):
    sizes = {}
    for _, j, f in sift.hnsw_pairs():
        sizes.setdefault(hnsw_key(j)[0], (mib(j), mib(f)))
    m16j, m16f = sizes[16]
    worst = max(abs(a - b) / b * 100 for a, b in sizes.values())
    ndis = read_ndis()
    pct = [(j / f - 1) * 100 for _, f, j in ndis]
    require(min(pct) > 0, 'this project no longer computes more distances than FAISS')
    return {
        'm16j': f'{m16j:.1f}', 'm16f': f'{m16f:.1f}',
        'apart': f'{abs(m16j - m16f) / m16f * 100:.1f}',
        'worst': f'{math.ceil(worst * 10) / 10:.1f}',
        'ndis': ndis, 'pct': f'{min(pct):.1f}–{max(pct):.1f}',
    }


def kernel_facts():
    k = jmh('jmh-distance.json', 'jmh-scan.json', 'jmh-scanram.json')
    step = lambda bench, d: k[(bench, d)] / (d / jmh_table.LANES)

    def ram_rate(d):
        vectors = max(1, jmh_table.RAM_BLOCK // (d * 4))
        return vectors / (k[('scanRamSimdL2Unrolled', d)] / 1e9)

    f = {
        'sc': {d: k[('pairScalarL2', d)] for d in (128, 960)},
        's1': {d: k[('pairSimdL2', d)] for d in (128, 960)},
        's4': {d: k[('pairSimdL2Unrolled', d)] for d in (128, 960)},
        'ip4_128': k[('pairSimdInnerProduct', 128)],
        'step_sc_960': step('pairScalarL2', 960),
        'step_s1': {d: step('pairSimdL2', d) for d in (128, 960)},
        'step_s4_960': step('pairSimdL2Unrolled', 960),
        'ram_rate': {d: ram_rate(d) / 1e6 for d in (128, 960)},
        'pair_rate': {d: 1e9 / k[('pairSimdL2Unrolled', d)] / 1e6 for d in (128, 960)},
        'gbps_960': ram_rate(960) * 960 * 4 / 1e9,
    }
    f['lanes_960'] = f['sc'][960] / f['s1'][960]
    f['chain'] = f['step_s1'][960] / f['step_s4_960']
    f['fixed_loop'] = (128 / jmh_table.LANES) * f['step_s4_960']
    f['fixed'] = f['s4'][128] - f['fixed_loop']
    require(f['s4'][128] > f['s1'][128], 'four accumulators are no longer slower at d=128')
    require(f['ip4_128'] < f['s4'][128], 'inner product is no longer cheaper than L2 at d=128')
    require(f['step_s1'][128] < f['step_s1'][960], 'one-accumulator step no longer degrades at d=960')
    require(abs(f['ram_rate'][128] / f['pair_rate'][128] - 1) < 0.02,
            'the d=128 DRAM scan no longer matches the L1-resident pair rate')
    require(f['ram_rate'][960] < f['pair_rate'][960], 'the d=960 DRAM scan is no longer slower')
    return f


def pq_scan_facts():
    p = jmh('jmh-pqscan.json')
    four, serial = p[('scanFourAccumulators', 16)], p[('scanSerialAccumulator', 16)]
    require(abs(four / serial - 1) < 0.02, 'the four-accumulator PQ scan is no longer a wash')
    return {'four': f'{four:.3f}', 'serial': f'{serial:.3f}'}


def kmeans_facts(sift, kf):
    row = sift.jp['nlist=4096,m=16,nprobe=1']
    base = build_s(row)
    rate = KMEANS_ITERATIONS * KMEANS_TRAINING_POINTS * 4096 / base / 1e6
    require(rate > kf['pair_rate'][128], 'k-means no longer runs above the L1-resident pair rate')
    return {'base': f'{base:.1f}', 'blocked': f'{BLOCKED_KMEANS_BUILD_S:.1f}',
            'rate': f'{rate:.0f}', 'pair': f'{kf["pair_rate"][128]:.0f}',
            'commit': BLOCKED_KMEANS_COMMIT}


# ---------------------------------------------------------------- README blocks

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
reverses: on the {len(clean)} configurations unaffected by the timing artefact described below, \
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


def result_rows(sweep, targets):
    """(label, kind, java row, faiss row) for each results-table line."""
    for threshold in targets:
        j = cheapest(list(sweep.jh.values()), threshold)
        f = cheapest(list(sweep.fh.values()), threshold)
        if j is not None:
            yield f'≥{threshold:.2f}', 'HNSW', j, f
    for table_j, table_f, kind in ((sweep.jh, sweep.fh, 'HNSW'), (sweep.jp, sweep.fp, 'IVF-PQ')):
        yield 'max', kind, best_recall(list(table_j.values())), best_recall(list(table_f.values()))


def win_marks(sweep, j, f):
    """(java is shown winning, faiss is shown winning, faiss row is suspect).

    An inflated FAISS row cannot be used to declare a win for *this* project. It can
    still show a FAISS win, which the inflation only understates.
    """
    tainted = is_suspect(sweep, f)
    return us(j) < us(f) and not tainted, us(f) < us(j), tainted


def results_table(sweep, targets, note='', brute=None):
    lines = ['| target | index | configuration | recall@10 | mean latency | FAISS | index size |',
             '|---|---|---|---:|---:|---:|---:|']
    for label, kind, j, f in result_rows(sweep, targets):
        lines.append(row_line(label, kind, sweep, j, f))
    if brute is not None:
        rec, secs, nq, threads = brute
        ms = secs / nq * 1000
        lines.append(f'| — | exact brute force | scalar kernel, {threads} threads | '
                     f'{rec:.4f} | ~{ms:.0f} ms | — | — |')
        note += (f'\n\nThe brute-force row is the Checkpoint 1 oracle: {secs:.0f} s for {nq:,} '
                 f'queries on {threads} threads. Every other row in this table is '
                 f'single-threaded, so it is **not** a like-for-like latency.')
    return '\n'.join(lines) + note


def row_line(label, kind, sweep, j, f):
    """One results row. The FAISS cell names its own configuration when it differs."""
    faiss_cell = f'{us(f):,.0f} µs'
    if f['params'] != j['params']:
        faiss_cell += f'<br><sub>{f["params"]}, r={recall(f):.4f}</sub>'
    java_wins, faiss_wins, tainted = win_marks(sweep, j, f)
    if tainted:
        faiss_cell += ' ' + DAGGER
    jw = '**' if java_wins else ''
    fw = '**' if faiss_wins else ''
    return (f'| {label} | {kind} | {j["params"]} | {recall(j):.4f} | '
            f'{jw}{us(j):,.0f} µs{jw} | {fw}{faiss_cell}{fw} | {mib(j):.1f} MiB |')


def block_results_sift(sift):
    return ('**SIFT1M** — 1M × 128, the full 10,000-query set, k=10, single-threaded.\n\n'
            + results_table(sift, (0.90, 0.95, 0.99), brute=oracle('sift1m')))


def block_results_gist(gist):
    note = (f'\n\n{DAGGER} FAISS row recorded after the machine slowed partway through the GIST1M '
            f'sweep — see [the timing artefact](#{ARTEFACT_ANCHOR}). Its latency is inflated, '
            f'so this comparison understates FAISS.')
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


def block_artefact(sift, gist):
    a = artefact_facts(sift, gist)
    m, efc = a['first']
    return f"""The FAISS GIST1M sweep ran as a single job on {a['date']}, building its HNSW indexes in order of
`M` and `efConstruction` and then the IVF-PQ ones. From the `M={m},efC={efc}` build onward —
started at about {a['hhmm']} UTC — FAISS's timings are inflated, and they stay inflated to the end
of the run. Two measurements that share no code step up at the same point:

* **Build time.** FAISS needs {a['clean_build']} of this project's build time on the first {WORDS[a['n_clean_blocks']]}
  GIST1M indexes and {a['affected_build']} on the last {WORDS[a['n_affected_blocks']]}. On SIFT1M the same ratio stays
  within {a['sift_build']} across all {WORDS[a['n_sift_blocks']]}.
* **Search latency.** At `M={m}`, this project's latency is {a['lat_before']}x FAISS's on the
  {WORDS[a['n_before']]} indexes built before the step and {a['lat_first']}x on the `efC={efc}` index built after
  it. Across the whole sweep, this project is faster than FAISS at {a['n_wins']} of {a['n_all']} GIST1M
  configurations, and all {a['n_wins']} are on the last {WORDS[a['n_affected_blocks']]} indexes.

A step in both at once, tied to a moment in the run rather than to any parameter, points to
the machine slowing down rather than to either implementation.

**How the numbers here handle it.** `scripts/readme_tables.py` flags an index whose
FAISS/Java build-time ratio sits more than {SUSPECT_THRESHOLD}x from the sweep's median (the
affected ones sit {a['dev']}x from it), and treats every FAISS row recorded from the first flagged
index onward as affected. On GIST1M that is {a['n_suspect']} of {a['n_all']} HNSW configurations — every
one at M={a['whole_m']} —{' plus all IVF-PQ configurations' if a['ivfpq_all'] else ''}; on SIFT1M it flags nothing. Affected rows
are marked {DAGGER}, left out of every GIST1M range, and never counted as a win for this project.
`python3 scripts/readme_tables.py --audit` prints the evidence index by index.

**What the unaffected data shows.** FAISS is faster at all {a['n_clean']} unaffected GIST1M
configurations, by {a['clean_lat']}x. No M={a['whole_m']} index is unaffected, so on GIST1M the
comparison at M={a['whole_m']} has not been measured cleanly; re-running the {WORDS[a['n_affected_blocks']]} affected FAISS
builds would settle it."""


def block_faiss_checks(sift):
    c = faiss_check_facts(sift)
    lo_ef, lo_f, lo_j = c['ndis'][0]
    hi_ef, hi_f, hi_j = c['ndis'][-1]
    return f"""1. **The graphs are the same size.** Corrected for the `IndexFlat` that `IndexHNSWFlat`
   embeds, FAISS's graph is {c['m16f']} MiB at M=16 against this project's {c['m16j']} — \
{c['apart']}% apart, and
   within {c['worst']}% at every M.
2. **The searches do the same work.** `hnsw_stats.ndis` per query at M=16/efC=200: \
{fnum(lo_j)} vs
   {fnum(lo_f)} at ef={lo_ef}, {fnum(hi_j)} vs {fnum(hi_f)} at ef={hi_ef}. \
**This implementation computes {c['pct']}% *more*
   distances and is still faster** — and that also explains its slightly higher recall.
3. **FAISS is not a crippled build.** The wheel reports `OPTIMIZE DD ARM_NEON MAC_METAL`
   with the full ASIMD instruction set."""


def ratio_table_md(gist, first_header='ef'):
    cells = ratio_cells(gist)
    lines = [f'| {first_header} | M=8 | M=16 | M=32 |', '|---:|---:|---:|---:|']
    for ef in (16, 64, 512):
        out = []
        for m in (8, 16, 32):
            text, state = cells[(ef, m)]
            out.append({'clean': text, 'partial': f'{text} {DAGGER}',
                        'none': f'*({text})* {DAGGER}'}[state])
        lines.append(f'| {ef} | ' + ' | '.join(out) + ' |')
    return '\n'.join(lines)


def block_ratio_gist(gist):
    return ratio_table_md(gist) + f"""

Mean over the three `efConstruction` values, **counting only rows unaffected by the
[timing artefact](#{ARTEFACT_ANCHOR})**. {DAGGER} marks a cell that had to drop at least one
row; a cell in *(parentheses)* had no unaffected rows at all and is shown for reference only,
not relied on."""


def block_ratio_sift(sift, gist):
    s = [us(j) / us(f) for _, j, f in sift.hnsw_pairs()]
    clean = [us(j) / us(f) for _, j, f in gist.hnsw_pairs(skip_suspect=True)]
    require(max(s) < 1 < min(clean), 'the datasets no longer bracket the crossover')
    return f"""On SIFT1M the same ratio runs {span(s)} across all {len(s)} configurations — this
project is faster everywhere. On GIST1M's {len(clean)} unaffected configurations it runs
{span(clean)} — FAISS is faster everywhere. The two datasets bracket the crossover rather than
contradicting each other, and three gradients account for both:"""


def block_gradients(sift, gist):
    """The gradient bullets. Each asserted direction is checked against the data."""
    g = gradient_facts(sift, gist)
    missing = f'\n  (no GIST1M M={g["g_missing"]} build is unaffected)' if g['g_missing'] else ''
    return f"""* **↑ dimension → FAISS gains.** More arithmetic per candidate, bookkeeping unchanged.
  Mean ratio {g['s_dim']} on SIFT1M against {g['g_dim']} on GIST1M's unaffected rows. The distance
  kernel is theirs.
* **↑ efSearch → this project gains.** More candidates through the visited stamps and the two
  heaps, which is the bookkeeping steps 2 and 3 bought. Mean ratio falls from {g['s_ef'][0]} at
  ef={g['ef_lo']} to {g['s_ef'][1]} at ef={g['ef_hi']} on SIFT1M, and from {g['g_ef'][0]} to {g['g_ef'][1]} on GIST1M.
* **↑ M → FAISS gains.** Mean ratio {g['s_m']} from M={g['s_m_range'][0]} to M={g['s_m_range'][1]} on SIFT1M, and
  {g['g_m']} from M={g['g_m_range'][0]} to M={g['g_m_range'][1]} on GIST1M's unaffected rows{missing}. A higher degree gives the step-5 software prefetch more
  neighbours to issue per hop, but on this data that does not turn into a gain against FAISS."""


def opt_steps():
    steps = [('0. naive reference', 'naive reference', 'hnsw-opt-step0-naive.csv'),
             ('1. flat `int[]` arenas', 'flat <code>int[]</code> neighbour arenas',
              'hnsw-opt-step1-arenas.csv'),
             ('2. versioned visited stamps', 'versioned visited stamps',
              'hnsw-opt-step2-visited.csv'),
             ('3. primitive `long[]` heaps', 'primitive <code>long[]</code> heaps',
              'hnsw-opt-step3-primitive-heaps.csv'),
             ('4. split traversal / distances', 'split traversal from distances',
              'hnsw-opt-step4-split-traversal.csv'),
             ('5. software prefetch', 'software prefetch', 'hnsw-opt-step5-prefetch.csv')]
    for md_label, html_label, name in steps:
        row = [r for r in read(name) if r['params'].endswith('ef=64')][0]
        yield md_label, html_label, float(row['p95_latency_us']), build_s(row), mib(row)


def block_opt_table():
    lines = ['| change | p95 @ ef=64 | build |', '|---|---:|---:|']
    rows = list(opt_steps())
    for i, (label, _, p95, build, _) in enumerate(rows):
        bold = '**' if i == len(rows) - 1 else ''
        lines.append(f'| {label} | {bold}{p95:.1f} µs{bold} | {bold}{build:.1f} s{bold} |')
    lines.append('')
    lines.append(f'Cumulative: **{rows[0][2] / rows[-1][2]:.2f}x** on p95 at ef=64, '
                 f'**{rows[0][3] / rows[-1][3]:.2f}x** on build.')
    return '\n'.join(lines)


def block_failed(sift):
    kf = kernel_facts()
    p = pq_scan_facts()
    km = kmeans_facts(sift, kf)
    return f"""* **Four accumulators in the PQ scan.** The exact fix that wins {kf['chain']:.1f}x in the L2 kernel
  measured {p['four']} µs against {p['serial']} at m=16 — nothing. The caller's loop over codes already supplies all
  the instruction-level parallelism the processor needs; the L2 kernel had no outer loop to
  hide behind. Reverted.
* **Cache-blocking the k-means assignment.** {km['blocked']} s against {km['base']} for `nlist=4096` — a wash.
  (The blocked build was reverted without keeping its CSV row; its time is recorded in commit
  {km['commit']}.) The loop was already running at {km['rate']} M distances/s, *above* the {km['pair']} M/s the
  microbenchmark gives for an L1-resident pair, because the point stays in L1 across all 4096
  centroids. There was no memory traffic to remove. FAISS's 8x is **register** blocking via
  `sgemm` — a tile of pairs computed at once so each load feeds many FMAs — not cache blocking.
  Reverted."""


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
    above = (recall(fb) - sc) * 100000
    require(above > 0, 'FAISS no longer scores above the SIFT1M oracle figure')
    a = artefact_facts(sift, gist)
    pq_caveat = ('  Every FAISS GIST1M IVF-PQ row was recorded after the machine slowed, so both\n'
                 '  GIST1M figures here are **lower bounds on the gap**: the artefact inflates\n'
                 '  FAISS, which flatters this implementation.'
                 if a['ivfpq_all'] else '')
    return f"""* **The metric's ceiling is {sc:.6f} on SIFT and {gc:.6f} on GIST**, because ids are not
  unique under ties. The searches themselves are exact. PROTOCOL.md calls {sc:.6f} a hard bound
  — "no configuration can do better" ([line 39](PROTOCOL.md#L39)) and "the attainable maximum"
  ([line 98](PROTOCOL.md#L98)) — and **that is wrong**: FAISS scores {recall(fb):.6f} on SIFT1M,
  {WORDS.get(round(above), f'{above:.0f}')} slot{'s' if round(above) != 1 else ''} in 100,000 above it. The figure is the tie-breaking this project's oracle
  happened to choose, not an upper bound on any index. The protocol is frozen, so the error is
  recorded here rather than edited there.
* **{WORDS[a['n_affected_blocks']].capitalize()} FAISS GIST1M builds need re-running.** The
  [timing artefact](#{ARTEFACT_ANCHOR}) leaves {a['n_suspect']} of {a['n_all']} GIST1M HNSW
  configurations (every one at M={a['whole_m']}){' and all GIST1M IVF-PQ configurations' if a['ivfpq_all'] else ''} without a
  trustworthy FAISS latency. GIST1M latency comparisons here rest on the {a['n_clean']} that remain.
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
  {hnsw_build(gist, True, invert=True)}x *slower* on GIST1M** (unaffected blocks only) — the same
  dimension gradient that governs the search comparison, showing up in construction as well.
* **No OPQ rotation**, so the product quantizer assumes the subspaces are uncorrelated. GIST
  shows what that costs. A learned rotation is the standard fix and is the single most
  valuable thing missing here — plausibly enough to clear 0.80 recall at m=32, which would
  satisfy both halves of the Checkpoint 4 target honestly rather than by redefining it.
* **`m` must divide the dimension in this implementation**, so code sizes are 8/16/32/64/128
  bytes and nothing between — which is why the Checkpoint 4 frontier has no point to land on
  between 32 and 64. FAISS pads the last subvector and accepts any `m`.
* **Deletion, updates and persistence are unimplemented.** This indexes a fixed set once."""


# ---------------------------------------------------------------- docs/analysis.md


def block_analysis_ratio(gist, sift):
    """docs/analysis.md section 6 - same data, that document's wording."""
    s_r = [us(j) / us(f) for _, j, f in sift.hnsw_pairs()]
    clean = [us(j) / us(f) for _, j, f in gist.hnsw_pairs(skip_suspect=True)]
    a = artefact_facts(sift, gist)
    return f"""{ratio_table_md(gist, 'GIST1M, ef')}

Mean over the three `efConstruction` values, counting only rows unaffected by the timing
artefact in the FAISS GIST1M sweep, described in
[README.md](../README.md#{ARTEFACT_ANCHOR}). {DAGGER} marks a cell that had to drop at least
one row. A cell in *(parentheses)* had no unaffected rows — every FAISS M={a['whole_m']} GIST1M index was
built after the machine slowed — and is not evidence either way.

On SIFT1M the same ratio runs {span(s_r)} across all {len(s_r)} configurations, and on the
{len(clean)} unaffected GIST1M configurations it runs {span(clean)}. Three gradients explain both
tables at once:"""


# ---------------------------------------------------------------- docs/kernels.md


def block_kernel_tables():
    return jmh_table.render(jmh_table.load(jmh_table.KERNEL_RUNS))


def block_kernel_findings():
    k = kernel_facts()
    cyc = lambda ns: ns * CLOCK_GHZ
    return f"""**At d=128 the SIMD kernel hits the lane ceiling and stops.** {k['sc'][128] / k['s1'][128]:.2f}x against a 4-lane
machine is as good as this gets, and the four-accumulator variant is very slightly
*slower* than the single-accumulator one ({k['s4'][128]:.1f} vs {k['s1'][128]:.1f} ns).

Note which kernel that {k['sc'][128] / k['s1'][128]:.2f}x belongs to: the **single-accumulator** one. It keeps the same
serial FMA chain the scalar loop has, so it wins the lanes and nothing else — which is
exactly why it lands at 4x and not beyond. The {k['sc'][960] / k['s4'][960]:.2f}x figure at d=960 below belongs to the
**four-accumulator** kernel, which wins the lanes *and* breaks the chain. Comparing the two
headline numbers without noticing they come from different kernels makes them look
contradictory; they are not.

The d=128 row is also worth more than its speedup for a second reason: unrolling is not
free. Four accumulators cost
four vector zeroings up front and a three-add reduction tree plus a horizontal
`reduceLanes` at the end, and at d=128 there are only 32 vector steps to amortise that
over. The fixed cost is about {k['fixed']:.0f} ns of the {k['s4'][128]:.1f} ns call: the 32 steps themselves, at the
d=960 per-step cost below, account for only {k['fixed_loop']:.1f}.

**At d=960 the four-accumulator kernel goes {k['sc'][960] / k['s4'][960]:.1f}x, which is impossible for a 4-lane
machine — so the scalar baseline must be the thing that is broken.** It is. The scalar
loop is

```java
sum += d * d;
```

and every iteration's addition needs the previous iteration's `sum`. Java is strict about
float semantics, so neither javac nor HotSpot may reassociate that sum, and the loop runs
at one addition per FP-add *latency* rather than per FP-add *throughput*, no matter how
many FP units the core has. The measured cost is {k['step_sc_960']:.2f} ns per four elements at d=960 —
roughly {cyc(k['step_sc_960']):.0f} cycles for what is 4 multiplies and 4 adds of real work. (Cycle counts in this
section take the clock as about {CLOCK_GHZ} GHz.)

The single-accumulator SIMD kernel has exactly the same problem one level up: it is one
FMA chain, so it is latency-bound too, and its {k['step_s1'][960]:.3f} ns per 4-lane step at d=960 is about
{cyc(k['step_s1'][960]):.1f} cycles — an FMA latency, not an FMA throughput. Splitting into four independent
accumulators is what removes the dependency: {k['step_s4_960']:.3f} ns per step, about {cyc(k['step_s4_960']):.1f} cycles, which is
close to the machine's load-issue limit of two 128-bit loads per step. So the {k['sc'][960] / k['s4'][960]:.1f}x is two
separate wins stacked — {k['lanes_960']:.1f}x from the lanes, ~{k['chain']:.1f}x from breaking a serial dependency that
the scalar version was never allowed to break.

**The one-accumulator kernel does not show that penalty at d=128** ({k['step_s1'][128]:.3f} ns per step
there against {k['step_s1'][960]:.3f} at d=960, for the same compiled loop). I do not have a confident
explanation; the plausible ones are different JIT unrolling decisions in the two forks —
JMH runs each parameter in its own JVM, so the two are separately profiled and separately
compiled — or the shorter loop being better covered by out-of-order execution across the
call boundary. It is recorded here as measured rather than explained away, and it does not
affect the choice of kernel: the four-accumulator version is the only one whose per-step
cost does not degrade with dimension, so it is the one the indexes use.

**A sequential scan is compute-bound, not bandwidth-bound — even from DRAM.** The
64 MiB block cannot be in any cache on this machine, and the SIMD scan still runs at
{k['ram_rate'][128]:.1f} M vec/s at d=128, matching the {k['pair_rate'][128]:.1f} M vec/s the L1-resident pair benchmark
gives. The hardware prefetcher sees a perfectly linear stride and hides the entire memory
latency; the kernel never waits. This is worth knowing before optimising an index: for
SIFT-shaped data, making the distance calculation faster helps, and making the memory
access pattern nicer does not, because the memory system was never the constraint.

At d=960 that stops being quite true. The SIMD kernel manages {k['pair_rate'][960]:.1f} M vec/s on an
L1-resident pair but {k['ram_rate'][960]:.1f} M vec/s scanning from DRAM — a {(1 - k['ram_rate'][960] / k['pair_rate'][960]) * 100:.0f}% loss while moving
{k['gbps_960']:.1f} GB/s — so at 960 dimensions the kernel has become fast enough to start feeling the
memory system. That is the first hint of what Phase 6 is about: the high-dimensional case
is not merely "the same thing but slower", it is a different bottleneck.

**Inner product is marginally cheaper than L2** ({k['ip4_128']:.1f} vs {k['s4'][128]:.1f} ns at d=128): it is one FMA
per step where L2 needs a subtract and an FMA. Both metrics are implemented, but every
number in this project uses L2, because that is the metric SIFT1M and GIST1M are
distributed with ground truth for."""


# ---------------------------------------------------------------- docs/results/README.md


def block_results_readme_recall(sift):
    sc, _ = ceilings()
    fb = best_recall(list(sift.fh.values()))
    require(recall(fb) > sc, 'FAISS no longer scores above the SIFT1M oracle figure')
    return f"""* `recall_at_k` — mean set overlap against the shipped ground truth, per PROTOCOL.md §3.
  Exact search scores **{sc:.6f}** on SIFT1M, not 1.0, because the dataset contains exactly
  tied neighbours and the shipped ids break those ties one particular way (PROTOCOL.md §1).
  That is not an upper bound: an index that happens to break ties the way the shipped file
  does can score higher, and FAISS's `IndexHNSWFlat` reaches **{recall(fb):.6f}** in
  `faiss-sift1m.csv`."""


# ---------------------------------------------------------------- docs/guide/guide.html


def html_num(text, strong=False):
    return f'<td class="n">{"<strong>" if strong else ""}{text}{"</strong>" if strong else ""}</td>'


def block_guide_ceiling(sift):
    sc, gc = ceilings()
    fb = best_recall(list(sift.fh.values()))
    ties = oracle_ties('sift1m')
    return f"""<p style="margin-bottom:0">{ties:,} of the 10,000 SIFT queries have such ties. So even a
perfect search, scored against the shipped file, gets <strong>{sc:.6f}, not 1.0</strong>. That
figure is not a hard limit either: an index that happens to break ties the way the file does
can score slightly above it, and FAISS does ({recall(fb):.6f}, Part 9.1). GIST has the same
problem for a different reason (Part 9), with a figure of {gc:.6f}.</p>"""


def block_guide_kernel_table():
    k = kernel_facts()
    rows = [('scalar', k['sc']), ('SIMD, 1 accumulator', k['s1']), ('SIMD, 4 accumulators', k['s4'])]
    best = {d: max(rows[1:], key=lambda r: k['sc'][d] / r[1][d])[0] for d in (128, 960)}
    out = ['<table>',
           '<tr><th>Kernel</th><th class="n">d=128</th><th class="n">speedup</th>'
           '<th class="n">d=960</th><th class="n">speedup</th></tr>']
    for label, vals in rows:
        cells = []
        for d in (128, 960):
            speed = k['sc'][d] / vals[d]
            cells.append(html_num(f'{vals[d]:.1f} ns'))
            cells.append(html_num(f'{speed:.2f}×', strong=label == best[d]))
        out.append(f'<tr><td>{label}</td>' + ''.join(cells) + '</tr>')
    out.append('</table>')
    s128 = k['sc'][128] / k['s1'][128]
    s960 = k['sc'][960] / k['s4'][960]
    out.append('')
    out.append(f"""<p>The {s128:.2f}× at d=128 is expected — four lanes, roughly four times faster. But
<strong>{s960:.2f}× is impossible on a four-lane machine.</strong> You cannot get more than 4× from
having 4 lanes. So something else must be going on, and it must be that the <em>scalar
baseline</em> is unusually slow at 960 dimensions.</p>""")
    return '\n'.join(out)


def block_guide_kernel_lesson():
    k = kernel_facts()
    return f"""<p style="margin:0">A speedup <em>larger than the lane count</em> is never extra parallelism in
the SIMD code — it is a serial dependency in the baseline that the SIMD version was allowed to
break. At d=960 the four-accumulator kernel wins {k['lanes_960']:.1f}× from lanes and another {k['chain']:.1f}× from breaking
the chain. At d=128 it wins only the lanes, and the extra accumulators actually cost {k['s4'][128] - k['s1'][128]:.1f} ns
because setting up and combining four running totals cannot be amortised over just 32 steps.</p>"""


def block_guide_scan():
    k = kernel_facts()
    return f"""<p>A second measurement worth knowing: scanning a large block of vectors sequentially runs at
the same speed whether the data is in the fastest cache or out in main memory ({k['ram_rate'][128]:.1f} million
vectors/second from main memory against {k['pair_rate'][128]:.1f} million in cache, at d=128). The CPU's
prefetcher sees the predictable pattern and fetches ahead perfectly. <strong>This means a
sequential scan is limited by arithmetic, not by memory</strong> — which turns out to matter a
lot in Part 6, because a graph walk is <em>not</em> sequential and does not get that benefit.</p>"""


def block_guide_naive():
    _, _, p95, build, _ = next(opt_steps())
    return f"""<p>The reference implementation uses the obvious Java data structures:
<code>ArrayList&lt;Integer&gt;</code> for neighbour lists, <code>HashSet&lt;Integer&gt;</code>
to remember which nodes have been visited, and <code>PriorityQueue&lt;Candidate&gt;</code> for
both heaps. On SIFT1M it builds in {build:.0f} seconds and answers a query in {p95:.1f} µs (p95) at
<code>ef=64</code>.</p>"""


def block_guide_opt_table():
    rows = list(opt_steps())
    out = ['<table>',
           '<tr><th>#</th><th>Change</th><th class="n">p95 @ ef=64</th><th class="n">build</th>'
           '<th class="n">memory</th></tr>']
    for i, (_, label, p95, build, mem) in enumerate(rows):
        last = i == len(rows) - 1
        out.append(f'<tr><td>{i}</td><td>{label}</td>{html_num(f"{p95:.1f} µs", last)}'
                   f'{html_num(f"{build:.1f} s", last)}{html_num(f"{mem:.1f} MiB")}</tr>')
    out.append('</table>')
    out.append('')
    out.append(f'<p class="small"><strong>Cumulative: {rows[0][2] / rows[-1][2]:.1f}× faster queries, '
               f'{rows[0][3] / rows[-1][3]:.1f}× faster builds, {rows[0][4] / rows[-1][4]:.1f}× less\n'
               f'memory</strong> — with byte-identical results.</p>')
    return '\n'.join(out)


def block_guide_pq_lookup():
    p = jmh('jmh-pq2.json')
    out = ['<table>',
           '<tr><th>m</th><th class="n">before</th><th class="n">after</th><th class="n">speedup</th></tr>']
    for m in (8, 16, 32, 64):
        before, after = p[('lookupTableSimd', m)], p[('lookupTableTransposed', m)]
        out.append(f'<tr><td>{m}</td>{html_num(f"{before:.2f} µs")}{html_num(f"{after:.2f} µs")}'
                   f'{html_num(f"{before / after:.1f}×")}</tr>')
    out.append('</table>')
    return '\n'.join(out)


def block_guide_results_sift(sift):
    out = ['<table>',
           '<tr><th>target</th><th>index</th><th>configuration</th><th class="n">recall@10</th>'
           '<th class="n">latency</th><th class="n">FAISS</th><th class="n">size</th></tr>']
    for label, kind, j, f in result_rows(sift, (0.90, 0.99)):
        java_wins, faiss_wins, tainted = win_marks(sift, j, f)
        faiss = f'{us(f):,.0f} µs'
        if f['params'] != j['params']:
            faiss += f'<br><span class="small">{spaced(f["params"])}, r={recall(f):.4f}</span>'
        if tainted:
            faiss += ' ' + DAGGER
        out.append(f'<tr><td>{label}</td><td>{kind}</td><td>{spaced(j["params"])}</td>'
                   f'{html_num(f"{recall(j):.4f}")}{html_num(f"{us(j):,.0f} µs", java_wins)}'
                   f'{html_num(faiss, faiss_wins)}{html_num(f"{mib(j):.1f} MiB")}</tr>')
    rec, secs, nq, threads = oracle('sift1m')
    out.append(f'<tr><td>—</td><td>exact brute force</td><td>{threads} threads</td>'
               f'{html_num(f"{rec:.4f}")}{html_num(f"~{secs / nq * 1000:.0f} ms")}'
               f'<td class="n">—</td><td class="n">—</td></tr>')
    out.append('</table>')

    sc, _ = ceilings()
    jb = best_recall(list(sift.jh.values()))
    fb = best_recall(list(sift.fh.values()))
    gap = (sc - recall(jb)) * 100000
    target = cheapest(list(sift.jh.values()), 0.99)
    pq = best_recall(list(sift.jp.values()))
    raw = int(target['base_bytes']) / MIB
    out.append('')
    out.append(f"""<p>At its best setting this HNSW reaches <strong>{recall(jb):.6f}</strong>, {gap:.0f} slots in
100,000 short of the {sc:.6f} figure from Part 2.4; FAISS reaches {recall(fb):.6f}, just above it.
The brute-force row is the exact search from Part 2.4, and it is <strong>not a like-for-like
latency</strong>: it ran on {threads} threads ({secs:.0f} s for {nq:,} queries), while every other row is
single-threaded.</p>

<p>The two families barely compete. HNSW needs {mib(target):.0f} MiB of graph <em>plus</em> the {raw:.0f} MiB of raw
vectors, because it computes real distances during the search. IVF-PQ's {mib(pq):.1f} MiB is the entire
index. If the data fits in memory, use the graph; if it does not, the quantizer is not a
compromise, it is the only option that runs.</p>""")
    return '\n'.join(out)


def block_guide_recall_drop(sift, gist):
    rows = [('HNSW, this project', 'jh', False), ('HNSW, FAISS', 'fh', False),
            ('IVF-PQ, this project', 'jp', True), ('IVF-PQ, FAISS', 'fp', True)]
    out = ['<table>',
           '<tr><th>best recall@10</th><th class="n">SIFT1M</th><th class="n">GIST1M</th>'
           '<th class="n">change</th></tr>']
    for label, attr, strong in rows:
        s = recall(best_recall(list(getattr(sift, attr).values())))
        g = recall(best_recall(list(getattr(gist, attr).values())))
        change = f'{(g - s) * 100:+.1f} points'.replace('-', '−')
        out.append(f'<tr><td>{label}</td>{html_num(f"{s:.4f}")}{html_num(f"{g:.4f}", strong)}'
                   f'{html_num(change, strong)}</tr>')
    out.append('</table>')
    return '\n'.join(out)


def block_guide_artefact(sift, gist):
    a = artefact_facts(sift, gist)
    m, efc = a['first']
    rows = ['<table>',
            '<tr><th>index</th><th class="n">build, this project</th><th class="n">build, FAISS</th>'
            '<th class="n">FAISS ÷ this project</th></tr>']
    for key in a['order']:
        java, faiss, ratio = a['blocks'][key]['java'], a['blocks'][key]['faiss'], a['ratio'][key]
        affected = key in a['suspect']
        mark = f' {DAGGER}' if affected else ''
        rows.append(f'<tr><td>M={key[0]}, efC={key[1]}{mark}</td>{html_num(f"{java:,.0f} s")}'
                    f'{html_num(f"{faiss:,.0f} s")}{html_num(f"{ratio:.2f}", affected)}</tr>')
    rows.append('</table>')
    table = '\n'.join(rows)
    ivfpq = (' Every GIST1M IVF-PQ row was recorded after the slowdown too.'
             if a['ivfpq_all'] else '')
    return f"""<p>The FAISS side of GIST1M ran as one long job on {a['date']}, building its HNSW indexes one after
another in the order below. Partway through — from the <code>M={m}, efC={efc}</code> build, which started
at about {a['hhmm']} UTC — the machine slowed down, and every FAISS timing recorded after that point is
inflated. Nothing in the result files says so. It has to be read out of the numbers.</p>

<p>The giveaway is build time. Building an index and answering queries are timed by different
code, so a change in how long FAISS takes to <em>build</em> cannot be caused by anything in the
search:</p>

{table}
<p class="small">{DAGGER} recorded after the slowdown. On SIFT1M the same ratio stays within
{a['sift_build']} for all {WORDS[a['n_sift_blocks']]} indexes.</p>

<p>FAISS builds GIST1M indexes in {a['clean_build']} of this project's time on the first
{WORDS[a['n_clean_blocks']]}, then {a['affected_build']} on the last {WORDS[a['n_affected_blocks']]}. Search latency shifts at the same point: at
M={m}, this project's latency is {a['lat_before']}× FAISS's on the {WORDS[a['n_before']]} indexes built before
the step and {a['lat_first']}× on the one built after it. A jump in both at once, tied to a moment in
the run rather than to any parameter, points to the machine rather than to either
implementation.{ivfpq}</p>

<div class="box warn">
<h4>All of this project's GIST1M "wins" came from the slow stretch</h4>
<p style="margin:0">In the raw numbers this project beats FAISS at {a['n_wins']} of the {a['n_all']} GIST1M HNSW
configurations, and all {a['n_wins']} are on the {WORDS[a['n_affected_blocks']]} affected indexes. On the {a['n_clean']} configurations measured
before the slowdown, FAISS is faster at every one, by {a['clean_lat']}×. No M={a['whole_m']} index was built before
it, so how the two compare at M={a['whole_m']} on GIST1M is simply not known yet — re-running the
{WORDS[a['n_affected_blocks']]} affected FAISS builds would settle it.</p>
</div>

<p>The rest of this part uses only unaffected rows for GIST1M and marks the others with
{DAGGER}. The tables that follow are produced from the result files by
<code>scripts/readme_tables.py</code>, which detects the slowdown with this build-time test
rather than by date, and <code>--audit</code> prints the evidence.</p>"""


def block_guide_gradients(sift, gist):
    s = [us(j) / us(f) for _, j, f in sift.hnsw_pairs()]
    clean = [us(j) / us(f) for _, j, f in gist.hnsw_pairs(skip_suspect=True)]
    require(max(s) < 1 < min(clean), 'the datasets no longer bracket the crossover')
    g = gradient_facts(sift, gist)
    cells = ratio_cells(gist)
    rows = ['<table>',
            '<tr><th class="n">GIST, ef</th><th class="n">M=8</th><th class="n">M=16</th>'
            '<th class="n">M=32</th></tr>']
    for ef in (16, 64, 512):
        tds = []
        for m in (8, 16, 32):
            text, state = cells[(ef, m)]
            tds.append(html_num({'clean': text, 'partial': f'{text} {DAGGER}',
                                 'none': f'<em>({text})</em> {DAGGER}'}[state]))
        rows.append(f'<tr><td class="n">{ef}</td>' + ''.join(tds) + '</tr>')
    rows.append('</table>')
    missing = (f'\n      No GIST1M M={g["g_missing"]} index is unaffected, so that column cannot say.'
               if g['g_missing'] else '')
    return f"""<p>Against FAISS, this project's HNSW is faster on SIFT at all {len(s)} configurations and slower on
GIST at all {len(clean)} unaffected ones. Read as a single ratio that looks contradictory. It is not.</p>

{chr(10).join(rows)}
<p class="small">Ratio of this project's latency to FAISS's. Below 1.0 means this project is
faster. Mean over the three <code>efConstruction</code> values, unaffected rows only: {DAGGER} marks a
cell that lost at least one row to the slowdown, and a value in <em>(parentheses)</em> had none left
and is shown for reference only. On SIFT the same ratio runs {span(s)}; on GIST's unaffected
rows, {span(clean)}.</p>

<p>Three consistent trends explain both datasets with one model:</p>
<ul>
  <li><strong>More dimensions → FAISS gains.</strong> More arithmetic per candidate while
      bookkeeping per candidate is unchanged: mean ratio {g['s_dim']} on SIFT against {g['g_dim']} on GIST.
      The distance kernel is FAISS's advantage.</li>
  <li><strong>More <code>efSearch</code> → this project gains.</strong> More candidates pass
      through the visited stamps and the heaps: the ratio falls from {g['s_ef'][0]} to {g['s_ef'][1]} on SIFT and
      from {g['g_ef'][0]} to {g['g_ef'][1]} on GIST between ef={g['ef_lo']} and ef={g['ef_hi']}. That bookkeeping is this
      project's advantage — exactly what steps 2 and 3 bought.</li>
  <li><strong>More <code>M</code> → FAISS gains.</strong> The ratio goes {g['s_m']} from M={g['s_m_range'][0]} to
      M={g['s_m_range'][1]} on SIFT, and {g['g_m']} from M={g['g_m_range'][0]} to M={g['g_m_range'][1]} on GIST.{missing}
      A higher degree gives the step-5 prefetch more neighbours to fetch per hop, but on this
      data that does not turn into a gain against FAISS.</li>
</ul>"""


def block_guide_checks(sift):
    c = faiss_check_facts(sift)
    lo_ef, lo_f, lo_j = c['ndis'][0]
    hi_ef, hi_f, hi_j = c['ndis'][-1]
    return f"""<ol>
  <li><strong>Graph sizes match.</strong> {c['m16f']} MiB against {c['m16j']} at M=16 — {c['apart']}% apart, within
      {c['worst']}% at every M. Two independent implementations filling the same degree budget.</li>
  <li><strong>Search work matches.</strong> Distance computations per query: {fnum(lo_j)} against FAISS's
      {fnum(lo_f)} at ef={lo_ef}; {fnum(hi_j)} against {fnum(hi_f)} at ef={hi_ef}. <strong>This project computes {c['pct']}% more
      distances and is still faster</strong> — which also explains its slightly higher recall.</li>
  <li><strong>FAISS is not a crippled build.</strong> The installed package reports
      <code>OPTIMIZE ARM_NEON</code> with the full instruction set available.</li>
</ol>"""


def block_guide_failed(sift):
    kf = kernel_facts()
    p = pq_scan_facts()
    km = kmeans_facts(sift, kf)
    return f"""<tr><td>Four accumulators in the PQ scan</td>
    <td>The same dependency-chain fix that wins {kf['chain']:.1f}× in the distance kernel.</td>
    <td><strong>Nothing</strong> — {p['four']} µs against {p['serial']}. The caller's loop over codes already
        supplied all the parallelism; the distance kernel had no outer loop to hide behind.
        Reverted.</td></tr>
<tr><td>Cache-blocking k-means</td>
    <td>Reuse data while it is in cache instead of re-reading it.</td>
    <td><strong>Nothing</strong> — {km['blocked']} s against {km['base']} (the first figure is from commit
        <code>{km['commit']}</code>; the reverted run's result file was not kept). The loop already ran at
        {km['rate']} million distances/second, <em>above</em> the rate for cached data, because the point being
        assigned stays in cache across all centroids. There was no memory traffic to remove.
        Reverted.</td></tr>"""


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


def oracle_ties(dataset):
    text = open(os.path.join(RESULTS, f'checkpoint1-oracle-{dataset}.txt')).read()
    return int(re.search(r'ties\s*:\s*([\d,]+) queries', text).group(1).replace(',', ''))


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
    return {
        README: {
            'headline': block_headline(sift, gist),
            'memory-spans': block_memory(sift),
            'build-divergence': block_build(sift, gist),
            'results-sift': block_results_sift(sift),
            'results-gist': block_results_gist(gist),
            'ceiling': block_ceiling(sift, gist),
            'artefact': block_artefact(sift, gist),
            'faiss-checks': block_faiss_checks(sift),
            'ratio-gist': block_ratio_gist(gist),
            'ratio-sift': block_ratio_sift(sift, gist),
            'gradients': block_gradients(sift, gist),
            'opt-table': block_opt_table(),
            'failed-optimizations': block_failed(sift),
            'limitations': block_limits(sift, gist),
        },
        ANALYSIS: {
            'analysis-ratio': block_analysis_ratio(gist, sift),
            'analysis-gradients': block_gradients(sift, gist),
        },
        KERNELS: {
            'kernel-tables': block_kernel_tables(),
            'kernel-findings': block_kernel_findings(),
        },
        RESULTS_README: {
            'recall-column': block_results_readme_recall(sift),
        },
        GUIDE: {
            'guide-ceiling': block_guide_ceiling(sift),
            'guide-kernel-table': block_guide_kernel_table(),
            'guide-kernel-lesson': block_guide_kernel_lesson(),
            'guide-scan': block_guide_scan(),
            'guide-naive': block_guide_naive(),
            'guide-opt-table': block_guide_opt_table(),
            'guide-pq-lookup': block_guide_pq_lookup(),
            'guide-results-sift': block_guide_results_sift(sift),
            'guide-recall-drop': block_guide_recall_drop(sift, gist),
            'guide-artefact': block_guide_artefact(sift, gist),
            'guide-gradients': block_guide_gradients(sift, gist),
            'guide-checks': block_guide_checks(sift),
            'guide-failed': block_guide_failed(sift),
        },
    }


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
    ap.add_argument('--check', action='store_true', help='exit 1 if any owned file would change')
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
        print('every generated region is up to date with docs/results/.')
        return 0
    print('regenerated ' + ', '.join(written))
    return 0


if __name__ == '__main__':
    sys.exit(main())
