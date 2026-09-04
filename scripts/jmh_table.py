#!/usr/bin/env python3
"""Merges JMH JSON result files into the tables used in docs/kernels.md.

    python3 scripts/jmh_table.py build/jmh-distance.json build/jmh-scan.json ...

Later files win on duplicate (benchmark, dim) keys, so a re-run of one benchmark can be
layered over an older full sweep.
"""
import json
import sys

LANES = 4          # float lanes in a 128-bit NEON vector
L2_BLOCK = 4 << 20
RAM_BLOCK = 64 << 20


def load(paths):
    results = {}
    for path in paths:
        with open(path) as f:
            for entry in json.load(f):
                name = entry["benchmark"].rsplit(".", 1)[-1]
                dim = int(entry["params"]["dim"])
                pm = entry["primaryMetric"]
                results[(name, dim)] = (pm["score"], pm["scoreError"])
    return results


def table(results, dims, title, rows, per_call_vectors=None):
    out = [f"### {title}\n"]
    out.append("| kernel |" + "".join(f" d={d} ns/op | d={d} vec/s | d={d} vs scalar |" for d in dims))
    out.append("|---|" + "".join(" ---: | ---: | ---: |" for _ in dims))
    base_key = rows[0][1]
    for label, key in rows:
        cells = []
        for d in dims:
            if (key, d) not in results:
                cells += ["-", "-", "-"]
                continue
            score, err = results[(key, d)]
            vectors = per_call_vectors(d) if per_call_vectors else 1
            rate = vectors / (score / 1e9)
            speedup = results[(base_key, d)][0] / score
            cells.append(f"{score:,.1f} ± {err:,.1f}")
            cells.append(f"{rate / 1e6:,.1f} M")
            cells.append("1.00x" if key == base_key else f"**{speedup:.2f}x**")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def main(paths):
    results = load(paths)
    dims = sorted({d for _, d in results})
    out = []
    out += table(results, dims, "L2 squared, one L1-resident pair", [
        ("scalar", "pairScalarL2"),
        ("SIMD, 1 accumulator", "pairSimdL2"),
        ("SIMD, 4 accumulators", "pairSimdL2Unrolled"),
    ])
    out += table(results, dims, "Inner product, one L1-resident pair", [
        ("scalar", "pairScalarInnerProduct"),
        ("SIMD, 4 accumulators", "pairSimdInnerProduct"),
    ])
    out += table(results, dims, "L2 squared, scan of a 4 MiB block (fits in L2)", [
        ("scalar", "scanL2ScalarL2"),
        ("SIMD, 4 accumulators", "scanL2SimdL2Unrolled"),
    ], per_call_vectors=lambda d: max(1, L2_BLOCK // (d * 4)))
    out += table(results, dims, "L2 squared, scan of a 64 MiB block (comes from DRAM)", [
        ("scalar", "scanRamScalarL2"),
        ("SIMD, 4 accumulators", "scanRamSimdL2Unrolled"),
    ], per_call_vectors=lambda d: max(1, RAM_BLOCK // (d * 4)))

    out.append("### Derived: cost per SIMD iteration and DRAM bandwidth\n")
    out.append("| kernel | d=128 ns per 4-lane step | d=960 ns per 4-lane step | "
               "d=128 DRAM GB/s | d=960 DRAM GB/s |")
    out.append("|---| ---: | ---: | ---: | ---: |")
    for label, pair_key, ram_key in [
        ("scalar", "pairScalarL2", "scanRamScalarL2"),
        ("SIMD, 1 accumulator", "pairSimdL2", None),
        ("SIMD, 4 accumulators", "pairSimdL2Unrolled", "scanRamSimdL2Unrolled"),
    ]:
        cells = []
        for d in dims:
            cells.append(f"{results[(pair_key, d)][0] / (d / LANES):.3f}")
        for d in dims:
            if ram_key is None:
                cells.append("-")
            else:
                vectors = max(1, RAM_BLOCK // (d * 4))
                rate = vectors / (results[(ram_key, d)][0] / 1e9)
                cells.append(f"{rate * d * 4 / 1e9:.1f}")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1:] or ["build/jmh-distance.json"])
