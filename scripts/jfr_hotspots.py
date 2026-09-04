#!/usr/bin/env python3
"""Aggregates JFR execution samples into a hot-method report.

    jfr print --events jdk.ExecutionSample build/ann.jfr > build/samples.txt
    python3 scripts/jfr_hotspots.py build/samples.txt

Two views, because they answer different questions:

  self  - the method the sample landed *in*. Where the cycles actually go.
  total - every method on the stack. Which call path is responsible for them.

A method with high self time is the thing to make faster; a method with high total
but low self time is the thing to call less often.
"""
import argparse
import re
from collections import Counter

FRAME = re.compile(r"^\s+(\S+?)\.(\w+)\(")


def parse(path, package_filter, under=None):
    self_counts = Counter()
    total_counts = Counter()
    samples = 0
    frames = []

    def flush():
        nonlocal samples
        if not frames:
            return
        # A build and a search phase land in one recording; `under` keeps only the
        # samples taken inside the phase being profiled.
        if under and not any(under in frame for frame in frames):
            return
        samples += 1
        for i, frame in enumerate(frames):
            if package_filter and package_filter not in frame:
                continue
            if i == 0:
                self_counts[frame] += 1
        seen = set()
        for frame in frames:
            if package_filter and package_filter not in frame:
                continue
            if frame not in seen:
                seen.add(frame)
                total_counts[frame] += 1

    with open(path, errors="replace") as handle:
        in_stack = False
        for line in handle:
            if line.startswith("jdk.ExecutionSample"):
                flush()
                frames.clear()
                in_stack = False
            elif "stackTrace = [" in line:
                in_stack = True
            elif in_stack:
                match = FRAME.match(line)
                if match:
                    frames.append(f"{match.group(1)}.{match.group(2)}")
                elif line.strip().startswith("]"):
                    in_stack = False
    flush()
    return self_counts, total_counts, samples


def report(title, counts, samples, limit):
    print(f"\n{title}  ({samples:,} samples)")
    print(f"{'%':>7}  {'samples':>8}  method")
    for method, count in counts.most_common(limit):
        print(f"{100 * count / samples:6.1f}%  {count:8,}  {method}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("samples")
    parser.add_argument("--package", default="", help="only count frames containing this")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--under", default=None,
                        help="only count samples whose stack contains this frame")
    args = parser.parse_args()

    self_counts, total_counts, samples = parse(args.samples, args.package, args.under)
    if samples == 0:
        raise SystemExit("no execution samples found; was the recording made with "
                         "settings=profile?")
    report("SELF time (the method the sample landed in)", self_counts, samples, args.limit)
    report("TOTAL time (anywhere on the stack)", total_counts, samples, args.limit)


if __name__ == "__main__":
    main()
