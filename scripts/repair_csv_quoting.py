#!/usr/bin/env python3
"""One-off repair for result CSVs written before Measurement.toCsvRow() quoted the
`index` column.

Those rows look like

    java,SIFT1M,hnsw-naive(M=16,efC=200,ef=16),"M=16,efC=200,ef=16",10,...

where the unquoted commas inside the index name shift every later column. The repair is
purely structural: the row is split at the quoted params field, the index name is
reassembled from the fragments before it and re-emitted quoted. No measured value is
read, altered, or recomputed.

    python3 scripts/repair_csv_quoting.py docs/results/*.csv
"""
import sys


def repair_line(line):
    if '"' not in line:
        return line, False
    head, _, rest = line.partition(',"')
    parts = head.split(",")
    if len(parts) <= 3:
        return line, False  # already well formed: harness,dataset,index
    harness, dataset = parts[0], parts[1]
    index = ",".join(parts[2:])
    return f'{harness},{dataset},"{index}",\"{rest}', True


def main(paths):
    for path in paths:
        with open(path) as handle:
            lines = handle.read().splitlines()
        out = []
        repaired = 0
        for line in lines:
            if line.startswith("harness,") or not line.strip():
                out.append(line)
                continue
            fixed, changed = repair_line(line)
            repaired += changed
            out.append(fixed)
        if repaired:
            with open(path, "w") as handle:
                handle.write("\n".join(out) + "\n")
        print(f"{path}: {repaired} rows repaired")


if __name__ == "__main__":
    main(sys.argv[1:])
