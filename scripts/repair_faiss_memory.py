#!/usr/bin/env python3
"""Applies PROTOCOL.md section 7's memory rule to a FAISS CSV written before
scripts/faiss_bench.py did it at source.

IndexHNSWFlat serialises a full IndexFlat alongside its graph, so its serialised size
includes the raw base vectors. The Java HNSW reports its graph without the vectors it
borrows, so leaving FAISS's number uncorrected compares a graph against a graph plus half
a gigabyte, and the memory plot would show a 4.5x difference that does not exist.

The correction is exact arithmetic on two recorded columns - index_bytes minus base_bytes -
and touches no measured quantity. IndexIVFPQ rows are left alone: that index stores codes,
not vectors.

    python3 scripts/repair_faiss_memory.py docs/results/faiss-sift1m.csv
"""
import csv
import sys


def main(paths):
    for path in paths:
        with open(path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            print(f"{path}: empty")
            continue
        fields = list(rows[0].keys())
        repaired = 0
        for row in rows:
            if "HNSW" not in row["index"]:
                continue
            total = int(row["index_bytes"])
            base = int(row["base_bytes"])
            if total <= base:
                continue  # already corrected
            row["index_bytes"] = str(total - base)
            repaired += 1
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{path}: {repaired} HNSW rows corrected to exclude the raw vectors")


if __name__ == "__main__":
    main(sys.argv[1:] or ["docs/results/faiss-sift1m.csv"])
