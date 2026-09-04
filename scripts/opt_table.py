#!/usr/bin/env python3
"""Builds the optimization table in docs/hnsw-optimization.md from the step CSVs.

    python3 scripts/opt_table.py docs/results/hnsw-opt-step*.csv

Every step CSV is the same configuration (SIFT1M, M=16, efConstruction=200, full query
set, three runs), so the only thing that varies between files is the implementation.
"""
import csv
import glob
import os
import re
import sys

LABELS = {
    "step0": "0. naive reference",
    "step1": "1. flat int[] arenas",
    "step2": "2. versioned visited stamps",
    "step3": "3. primitive long[] heaps",
    "step4": "4. split traversal / distances",
    "step5": "5. software prefetch",
}


def load(paths):
    steps = {}
    for path in sorted(paths):
        match = re.search(r"(step\d+)", os.path.basename(path))
        if not match:
            continue
        key = match.group(1)
        rows = {}
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                ef = int(row["params"].split("ef=")[-1])
                rows[ef] = row
        steps[key] = rows
    return dict(sorted(steps.items()))


def delta(current, previous):
    if previous is None or previous == 0:
        return ""
    change = 100 * (current - previous) / previous
    if abs(change) < 1.0:
        return "  (=)"
    return f"  ({change:+.0f}%)"


def main(paths):
    steps = load(paths)
    if not steps:
        raise SystemExit("no step CSVs found")
    efs = sorted(next(iter(steps.values())).keys())

    print("### p95 latency per query, microseconds\n")
    print("| change | " + " | ".join(f"ef={ef}" for ef in efs) + " |")
    print("|---|" + " ---: |" * len(efs))
    previous = None
    for key, rows in steps.items():
        cells = []
        for ef in efs:
            value = float(rows[ef]["p95_latency_us"])
            before = float(previous[ef]["p95_latency_us"]) if previous else None
            cells.append(f"{value:,.1f}{delta(value, before)}")
        print(f"| {LABELS.get(key, key)} | " + " | ".join(cells) + " |")
        previous = rows

    print("\n### Build time and memory\n")
    print("| change | build (s) | index (MiB) | recall@10 at ef=64 |")
    print("|---| ---: | ---: | ---: |")
    previous = None
    for key, rows in steps.items():
        row = rows[efs[0]]
        build = float(row["build_seconds"])
        mib = int(row["index_bytes"]) / 2 ** 20
        recall = float(rows[64]["recall_at_k"]) if 64 in rows else float("nan")
        before = float(previous["build_seconds"]) if previous else None
        print(f"| {LABELS.get(key, key)} | {build:,.1f}{delta(build, before)} "
              f"| {mib:,.1f} | {recall:.4f} |")
        previous = row

    first = next(iter(steps.values()))
    last = list(steps.values())[-1]
    print("\n### Cumulative\n")
    for ef in (64, 512):
        if ef in first and ef in last:
            a = float(first[ef]["p95_latency_us"])
            b = float(last[ef]["p95_latency_us"])
            print(f"* p95 at ef={ef}: {a:,.1f} us -> {b:,.1f} us  (**{a / b:.2f}x**)")
    a = float(first[efs[0]]["build_seconds"])
    b = float(last[efs[0]]["build_seconds"])
    print(f"* build: {a:,.1f} s -> {b:,.1f} s  (**{a / b:.2f}x**)")
    a = int(first[efs[0]]["index_bytes"]) / 2 ** 20
    b = int(last[efs[0]]["index_bytes"]) / 2 ** 20
    print(f"* index memory: {a:,.1f} MiB -> {b:,.1f} MiB  (**{a / b:.2f}x**)")


if __name__ == "__main__":
    main(sys.argv[1:] or glob.glob("docs/results/hnsw-opt-step*.csv"))
