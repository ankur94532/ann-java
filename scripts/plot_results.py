#!/usr/bin/env python3
"""Phase 5.3: the three plots, from the committed CSVs.

    python3 scripts/plot_results.py docs/results/*.csv --out docs/plots

Every plot is emitted twice, light and dark, so the README can serve the right one with
a <picture> element instead of showing white-on-white to dark-mode readers.

Encoding, chosen so the comparison the plots exist to make is the easy read:
  * hue      = index family   (blue = HNSW, orange = IVF-PQ)
  * style    = implementation (solid + filled marker = this project, dashed + hollow =
               FAISS)
so a Java/FAISS pair for the same index family shares a hue and differs only in stroke,
and the gap between them can be read directly. Colours are the validated categorical
slots 1 and 2; the pair clears every all-pairs gate in both modes
(`node scripts/validate_palette.js "#2a78d6,#eb6834" --pairs all`).
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "muted": "#8a8983",
        "grid": "#e5e4df",
        "hnsw": "#2a78d6",
        "ivfpq": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "muted": "#8a8983",
        "grid": "#33322f",
        "hnsw": "#3987e5",
        "ivfpq": "#d95926",
    },
}

# (harness, family) -> (label, hue key, linestyle, marker, filled)
SERIES = {
    ("java", "hnsw"): ("HNSW (this project)", "hnsw", "-", "o", True),
    ("java", "ivfpq"): ("IVF-PQ (this project)", "ivfpq", "-", "s", True),
    ("faiss", "hnsw"): ("FAISS IndexHNSWFlat", "hnsw", "--", "o", False),
    ("faiss", "ivfpq"): ("FAISS IndexIVFPQ", "ivfpq", "--", "s", False),
}
ORDER = [("java", "hnsw"), ("faiss", "hnsw"), ("java", "ivfpq"), ("faiss", "ivfpq")]


def family_of(index_name):
    lowered = index_name.lower()
    if "hnsw" in lowered:
        return "hnsw"
    if "ivfpq" in lowered or "ivf_pq" in lowered:
        return "ivfpq"
    return None


def load(paths, dataset):
    rows = defaultdict(list)
    for path in paths:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                if dataset and row["dataset"] != dataset:
                    continue
                family = family_of(row["index"])
                if family is None:
                    continue
                rows[(row["harness"], family)].append({
                    "params": row["params"],
                    "recall": float(row["recall_at_k"]),
                    "mean_us": float(row["mean_latency_us"]),
                    "p95_us": float(row["p95_latency_us"]),
                    "build_s": float(row["build_seconds"]),
                    "index_mib": int(row["index_bytes"]) / 2 ** 20,
                    "base_mib": int(row["base_bytes"]) / 2 ** 20,
                })
    return rows


def pareto(points, x_key, y_key, maximise_x=True, minimise_y=True):
    """The frontier: points not beaten on both axes at once.

    A sweep contains many configurations that are simply worse than another
    configuration on both recall and latency. Plotting all of them draws a cloud whose
    upper edge is the only part anyone reads, so the line is the frontier and the
    dominated points are dropped rather than joined by a zig-zag that implies a
    trade-off nobody would ever choose.
    """
    ordered = sorted(points, key=lambda p: (-p[x_key] if maximise_x else p[x_key]))
    frontier = []
    best = None
    for point in ordered:
        value = point[y_key]
        if best is None or (value < best if minimise_y else value > best):
            frontier.append(point)
            best = value
    return sorted(frontier, key=lambda p: p[x_key])


def style_axes(ax, theme, xlabel, ylabel, title, subtitle=None):
    ax.set_facecolor(theme["surface"])
    ax.figure.set_facecolor(theme["surface"])
    ax.grid(True, which="major", color=theme["grid"], linewidth=0.8, linestyle="-")
    ax.grid(True, which="minor", color=theme["grid"], linewidth=0.4, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=theme["text_secondary"], labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=theme["text_secondary"], fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, color=theme["text_secondary"], fontsize=10, labelpad=8)
    ax.set_title(title, color=theme["text_primary"], fontsize=13, loc="left", pad=18 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=theme["muted"],
                fontsize=9, va="bottom", ha="left")


def draw_series(ax, rows, theme, x_key, y_key, frontier=True, maximise_x=True,
                minimise_y=True):
    drawn = []
    for key in ORDER:
        points = rows.get(key)
        if not points:
            continue
        label, hue, linestyle, marker, filled = SERIES[key]
        colour = theme[hue]
        plotted = (pareto(points, x_key, y_key, maximise_x, minimise_y)
                   if frontier else sorted(points, key=lambda p: p[x_key]))
        ax.plot([p[x_key] for p in plotted], [p[y_key] for p in plotted],
                linestyle=linestyle, linewidth=2.0, color=colour,
                marker=marker, markersize=6.5,
                markerfacecolor=colour if filled else theme["surface"],
                markeredgecolor=colour, markeredgewidth=1.8,
                label=label, zorder=3, clip_on=False)
        drawn.append((label, colour))
    return drawn


def legend(ax, theme, loc="best"):
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) < 2:
        return
    # "best" rather than a fixed corner: these plots differ in where the data sits, and a
    # legend parked on top of a curve is worse than one that moves between figures.
    box = ax.legend(handles, labels, loc=loc, frameon=False, fontsize=9,
                    labelcolor=theme["text_secondary"], handlelength=2.6,
                    borderaxespad=0.6)
    for text in box.get_texts():
        text.set_color(theme["text_secondary"])


def save(fig, out_dir, name, mode, theme):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}-{mode}.png")
    fig.savefig(path, dpi=200, facecolor=theme["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")
    return path


def microseconds(value, _pos):
    if value >= 1000:
        return f"{value / 1000:g} ms"
    return f"{value:g} \u00b5s"


def log_ticks(ax, formatter):
    """Label 1/2/5 per decade and nothing else.

    Matplotlib's default log axis labels the decades with FuncFormatter and then labels
    the minor ticks with its own scientific-notation formatter, so the axis ends up
    carrying two incompatible number formats at once.
    """
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=20))
    ax.yaxis.set_major_formatter(FuncFormatter(formatter))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs="auto", numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot_recall_latency(rows, out_dir, dataset, mode):
    theme = THEMES[mode]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    style_axes(ax, theme,
               "recall@10", "mean latency per query (log scale)",
               f"Recall against latency, {dataset}",
               "single-threaded, k=10, full query set; each line is the Pareto frontier "
               "of its sweep")
    ax.set_yscale("log")
    log_ticks(ax, microseconds)
    draw_series(ax, rows, theme, "recall", "mean_us")
    legend(ax, theme, loc="upper left")
    return save(fig, out_dir, f"recall-latency-{dataset.lower()}", mode, theme)


def plot_recall_memory(rows, out_dir, dataset, mode):
    theme = THEMES[mode]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    base_mib = next((p["base_mib"] for points in rows.values() for p in points), None)
    style_axes(ax, theme,
               "recall@10", "index size (log scale, MiB)",
               f"Recall against index memory, {dataset}",
               "index structures only, excluding the raw base vectors")
    ax.set_yscale("log")
    log_ticks(ax, lambda v, _p: f"{v:g}")
    draw_series(ax, rows, theme, "recall", "index_mib")
    if base_mib:
        ax.axhline(base_mib, color=theme["muted"], linewidth=1.0, linestyle=":", zorder=1)
        ax.text(ax.get_xlim()[1], base_mib * 1.05,
                f"the raw base vectors alone are {base_mib:,.0f} MiB",
                color=theme["muted"], fontsize=8, va="bottom", ha="right")
    legend(ax, theme)
    return save(fig, out_dir, f"recall-memory-{dataset.lower()}", mode, theme)


def plot_build_recall(rows, out_dir, dataset, mode):
    theme = THEMES[mode]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    style_axes(ax, theme,
               "best recall@10 reached by the build", "build time (log scale)",
               f"Build cost against the recall it buys, {dataset}",
               "single-threaded construction; one point per built index, at its best "
               "search setting")
    ax.set_yscale("log")
    log_ticks(ax, lambda v, _p: f"{v / 60:g} min" if v >= 60 else f"{v:g} s")
    # One point per distinct build: group by build_seconds and take the best recall.
    grouped = {}
    for key, points in rows.items():
        by_build = defaultdict(list)
        for point in points:
            by_build[round(point["build_s"], 3)].append(point)
        grouped[key] = [{"recall": max(p["recall"] for p in group),
                         "build_s": build}
                        for build, group in by_build.items()]
    draw_series(ax, grouped, theme, "recall", "build_s", frontier=False)
    legend(ax, theme)
    return save(fig, out_dir, f"build-recall-{dataset.lower()}", mode, theme)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--out", default="docs/plots")
    parser.add_argument("--dataset", default="SIFT1M")
    args = parser.parse_args()

    rows = load(args.csv, args.dataset)
    if not rows:
        raise SystemExit(f"no rows for dataset {args.dataset} in {args.csv}")
    for key, points in sorted(rows.items()):
        print(f"{key[0]:6} {key[1]:6} {len(points):4} configurations")

    for mode in ("light", "dark"):
        plot_recall_latency(rows, args.out, args.dataset, mode)
        plot_recall_memory(rows, args.out, args.dataset, mode)
        plot_build_recall(rows, args.out, args.dataset, mode)


if __name__ == "__main__":
    main()
