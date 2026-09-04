#!/usr/bin/env python3
"""Phase 6: does recall loss concentrate on a particular kind of query?

    python3 scripts/plot_failures.py docs/results/perquery-hnsw-sift1m.csv \
        --label "HNSW M=16 ef=64" --out docs/plots

Input is the per-query CSV written by `Main analyse`. The x axis is the distance from
the query to its 10th true neighbour, which is the natural measure of how isolated a
query is: small means its neighbourhood is packed tight, large means it sits in a sparse
region of the space.

Queries are bucketed by decile of that distance rather than scattered, because 10,000
points at recall values that are mostly exactly 1.0 draw a solid bar, not a relationship.
The bucket mean is the thing the question is actually about.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THEMES = {
    "light": {"surface": "#fcfcfb", "text_primary": "#0b0b0b", "text_secondary": "#52514e",
              "muted": "#8a8983", "grid": "#e5e4df",
              "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#1a1a19", "text_primary": "#ffffff", "text_secondary": "#c3c2b7",
             "muted": "#8a8983", "grid": "#33322f",
             "series": ["#3987e5", "#d95926", "#199e70"]},
}
MARKERS = ["o", "s", "^"]


def load(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (np.array([float(r["dk"]) for r in rows]),
            np.array([float(r["recall"]) for r in rows]),
            np.array([int(r["orphan_neighbours"]) for r in rows]))


def deciles(dk, recall, buckets=10):
    edges = np.quantile(dk, np.linspace(0, 1, buckets + 1))
    edges[-1] += 1e-6
    centres, means, counts = [], [], []
    for i in range(buckets):
        mask = (dk >= edges[i]) & (dk < edges[i + 1])
        if not mask.any():
            continue
        centres.append(float(np.median(dk[mask])))
        means.append(float(recall[mask].mean()))
        counts.append(int(mask.sum()))
    return np.array(centres), np.array(means), np.array(counts)


def style_axes(ax, theme, xlabel, ylabel, title, subtitle):
    ax.set_facecolor(theme["surface"])
    ax.figure.set_facecolor(theme["surface"])
    ax.grid(True, color=theme["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=theme["text_secondary"], labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=theme["text_secondary"], fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, color=theme["text_secondary"], fontsize=10, labelpad=8)
    ax.set_title(title, color=theme["text_primary"], fontsize=13, loc="left", pad=18)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=theme["muted"],
            fontsize=9, va="bottom", ha="left")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--label", action="append", default=None,
                        help="one label per CSV, in order")
    parser.add_argument("--dataset", default="SIFT1M")
    parser.add_argument("--out", default="docs/plots")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    labels = args.label or [os.path.basename(p).replace(".csv", "") for p in args.csv]
    if len(labels) != len(args.csv):
        raise SystemExit("give one --label per CSV")

    for path, label in zip(args.csv, labels):
        dk, recall, orphans = load(path)
        affected = int((orphans > 0).sum())
        print(f"{label}: {len(dk):,} queries, mean recall {recall.mean():.4f}, "
              f"{(recall < 1).sum():,} imperfect, "
              f"{affected:,} with an unreachable true neighbour")

    for mode in ("light", "dark"):
        theme = THEMES[mode]
        fig, ax = plt.subplots(figsize=(8, 5.2))
        style_axes(ax, theme,
                   "distance from the query to its 10th true neighbour",
                   "mean recall@10",
                   f"Where recall is lost, {args.dataset}",
                   "queries bucketed into deciles of neighbourhood distance; "
                   "right-hand buckets are queries in sparse regions")
        for i, (path, label) in enumerate(zip(args.csv, labels)):
            dk, recall, _ = load(path)
            centres, means, counts = deciles(dk, recall)
            colour = theme["series"][i % len(theme["series"])]
            ax.plot(centres, means, linewidth=2.0, color=colour,
                    marker=MARKERS[i % len(MARKERS)], markersize=6.5,
                    markerfacecolor=colour, markeredgecolor=theme["surface"],
                    markeredgewidth=1.5, label=label, zorder=3)
        handles, legend_labels = ax.get_legend_handles_labels()
        if len(legend_labels) >= 2:
            box = ax.legend(loc="lower left", frameon=False, fontsize=9, handlelength=2.6)
            for text in box.get_texts():
                text.set_color(theme["text_secondary"])
        os.makedirs(args.out, exist_ok=True)
        name = args.name or f"recall-by-sparsity-{args.dataset.lower()}"
        path = os.path.join(args.out, f"{name}-{mode}.png")
        fig.savefig(path, dpi=200, facecolor=theme["surface"], bbox_inches="tight")
        plt.close(fig)
        print(f"  {path}")


if __name__ == "__main__":
    main()
