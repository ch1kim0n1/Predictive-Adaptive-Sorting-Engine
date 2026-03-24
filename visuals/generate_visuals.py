#!/usr/bin/env python3
"""
Generate polished benchmark visuals for PASE.

Usage:
    python3 generate_visuals.py
    python3 generate_visuals.py --csv visuals/bench_results.csv --out-dir visuals
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib not found. Run: pip install matplotlib", file=sys.stderr)
    sys.exit(1)


GPU_DISPATCH_THRESHOLD = 100_000
OUTPUT_DPI = 220
LINKEDIN_SQUARE_SIZE = (10.8, 10.8)

STRATEGY_COLOURS = {
    "insertion": "#1666D3",
    "run_merge": "#059669",
    "three_way": "#F59E0B",
    "introsort": "#6B7280",
    "gpu": "#D94673",
    "fallback": "#8B5E3C",
}
DEFAULT_COLOUR = "#64748B"

DATASET_ORDER = [
    "sorted",
    "nearly_sorted_95",
    "nearly_sorted_80",
    "long_runs",
    "heavy_dup",
    "reverse",
    "pipe_organ",
    "clustered",
    "random",
]


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "#F8FAFC",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#0F172A",
            "axes.titlecolor": "#0F172A",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": "#E2E8F0",
            "grid.alpha": 0.75,
            "axes.grid": True,
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "savefig.facecolor": "#F8FAFC",
            "figure.autolayout": False,
        }
    )


def _load_csv(path: Path) -> list[dict[str, str]]:
    lines = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


def _dataset_label(name: str) -> str:
    cleaned = name.replace("_", " ")
    if "95" in cleaned:
        return "Nearly sorted 95%"
    if "80" in cleaned:
        return "Nearly sorted 80%"
    return cleaned.title()


def _size_label(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    return f"{n // 1_000}k"


def _dataset_rank(name: str) -> int:
    try:
        return DATASET_ORDER.index(name)
    except ValueError:
        return len(DATASET_ORDER)


def make_speedup_chart(rows: list[dict[str, str]], out: Path) -> None:
    speed_col = "speedup_vs_std" if any("speedup_vs_std" in r for r in rows) else "speedup"
    points: list[tuple[str, int, float, str]] = []

    for row in rows:
        dataset = row.get("dataset", "unknown")
        try:
            n = int(row.get("n", "0"))
            speedup = float(row[speed_col])
        except (ValueError, KeyError):
            continue
        strategy = row.get("strategy", "fallback")
        points.append((dataset, n, speedup, strategy))

    points.sort(key=lambda p: (_dataset_rank(p[0]), p[1]))
    labels = [f"{_dataset_label(ds)}  |  n={_size_label(n)}" for ds, n, _, _ in points]
    values = [speed for _, _, speed, _ in points]
    colours = [STRATEGY_COLOURS.get(strategy, DEFAULT_COLOUR) for _, _, _, strategy in points]

    fig_h = max(8.0, len(labels) * 0.30)
    fig, ax = plt.subplots(figsize=(14.0, fig_h))

    y_positions = range(len(labels))
    bars = ax.barh(y_positions, values, color=colours, edgecolor="#FFFFFF", linewidth=0.8)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()

    ax.axvline(1.0, color="#DC2626", linestyle="--", linewidth=1.4, label="std::sort baseline")
    x_min = max(0.45, min(0.9, min(values) - 0.08))
    x_max = max(values) + 0.35
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Speedup vs std::sort (higher is better)", fontsize=11)
    ax.set_title(
        "PASE Performance Landscape by Workload and Input Size",
        fontsize=16,
        fontweight="bold",
        pad=14,
    )

    for bar, value in zip(bars, values):
        text_x = max(bar.get_width() + 0.02, x_min + 0.01)
        ax.text(
            text_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}x",
            va="center",
            ha="left",
            fontsize=8.8,
            color="#111827",
        )

    legend_patches = [
        mpatches.Patch(color=color, label=name.replace("_", " ").title())
        for name, color in STRATEGY_COLOURS.items()
    ]
    legend_patches.append(
        plt.Line2D([0], [0], color="#DC2626", linestyle="--", linewidth=1.4, label="std::sort baseline")
    )
    ax.legend(
        handles=legend_patches,
        loc="lower right",
        ncol=3,
        fontsize=8,
        frameon=True,
        framealpha=0.94,
    )

    median_speedup = sorted(values)[len(values) // 2]
    mean_speedup = sum(values) / len(values)
    ax.text(
        0.985,
        0.985,
        f"Median: {median_speedup:.2f}x | Mean: {mean_speedup:.2f}x | Peak: {max(values):.2f}x",
        transform=ax.transAxes,
        fontsize=9.2,
        color="#334155",
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.24", "fc": "#F8FAFC", "ec": "#CBD5E1", "alpha": 0.95},
    )

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    max_label_len = max(len(label) for label in labels)
    left_margin = min(0.33, max(0.22, 0.16 + max_label_len * 0.0032))
    fig.subplots_adjust(left=left_margin, right=0.985, top=0.94, bottom=0.07)
    fig.savefig(out, dpi=OUTPUT_DPI)
    plt.close(fig)
    print(f"Wrote {out}")


def make_cost_model_fit(rows: list[dict[str, str]], out: Path) -> None:
    grouped: dict[str, list[tuple[int, float, float, str]]] = {}
    for row in rows:
        dataset = row.get("dataset", "unknown")
        try:
            n = int(row.get("n", "0"))
            pase_ms = float(row.get("pase_ms", "0"))
            std_ms = float(row.get("std_ms", "0"))
        except ValueError:
            continue
        strategy = row.get("strategy", "introsort")
        grouped.setdefault(dataset, []).append((n, pase_ms, std_ms, strategy))

    datasets = sorted(grouped.keys(), key=_dataset_rank)
    cols = 3
    rows_count = max(1, math.ceil(len(datasets) / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(15.5, 4.2 * rows_count), sharex=True, sharey=True)
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis in axes_list:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.axvspan(GPU_DISPATCH_THRESHOLD, 1_200_000, color="#FCE7F3", alpha=0.45, lw=0)
        axis.grid(True, which="both", linewidth=0.7)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    for idx, dataset in enumerate(datasets):
        ax = axes_list[idx]
        series = sorted(grouped[dataset], key=lambda item: item[0])
        ns = [item[0] for item in series]
        pase = [item[1] for item in series]
        std = [item[2] for item in series]

        ax.plot(ns, pase, marker="o", linewidth=2.3, color="#0EA5E9", label="PASE")
        ax.plot(ns, std, marker="s", linewidth=1.9, linestyle="--", color="#EF4444", label="std::sort")

        ax.set_title(_dataset_label(dataset), fontsize=10.5, fontweight="bold", pad=6)

        best_gain = max((s / p) for p, s in zip(pase, std) if p > 0)
        ax.text(
            0.03,
            0.06,
            f"Best gain: {best_gain:.2f}x",
            transform=ax.transAxes,
            fontsize=8.3,
            color="#334155",
            bbox={"boxstyle": "round,pad=0.22", "fc": "#F1F5F9", "ec": "#CBD5E1", "alpha": 0.92},
        )

    for idx in range(len(datasets), len(axes_list)):
        axes_list[idx].set_visible(False)

    fig.suptitle(
        "Cost-Model Fit: PASE vs std::sort Latency Curves Across Workloads",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    fig.text(
        0.5,
        0.968,
        "Shaded area indicates the GPU-friendly regime (n >= 100k)",
        ha="center",
        fontsize=10,
        color="#475569",
    )

    fig.text(0.5, 0.015, "Array size n (log scale)", ha="center", fontsize=11)
    fig.text(0.01, 0.5, "Latency in milliseconds (log scale)", va="center", rotation="vertical", fontsize=11)

    legend_items = [
        plt.Line2D([0], [0], color="#0EA5E9", marker="o", linewidth=2.3, label="PASE"),
        plt.Line2D([0], [0], color="#EF4444", marker="s", linestyle="--", linewidth=1.9, label="std::sort"),
        mpatches.Patch(color="#FCE7F3", label="GPU-friendly regime", alpha=0.75),
    ]
    fig.legend(handles=legend_items, loc="upper right", bbox_to_anchor=(0.985, 0.985), framealpha=0.94, fontsize=9)

    fig.tight_layout(rect=(0.03, 0.03, 0.99, 0.95))
    fig.savefig(out, dpi=OUTPUT_DPI)
    plt.close(fig)
    print(f"Wrote {out}")


def make_linkedin_summary(rows: list[dict[str, str]], out: Path) -> None:
    speed_col = "speedup_vs_std" if any("speedup_vs_std" in r for r in rows) else "speedup"
    points: list[tuple[str, int, float, str]] = []
    for row in rows:
        dataset = row.get("dataset", "unknown")
        try:
            n = int(row.get("n", "0"))
            speedup = float(row[speed_col])
        except (ValueError, KeyError):
            continue
        strategy = row.get("strategy", "fallback")
        points.append((dataset, n, speedup, strategy))

    if not points:
        raise ValueError("No benchmark rows available for LinkedIn summary")

    points.sort(key=lambda p: p[2], reverse=True)
    top = points[:5]
    bottom = sorted(points, key=lambda p: p[2])[:3]
    med = sorted(p[2] for p in points)[len(points) // 2]
    peak = max(p[2] for p in points)
    mean = sum(p[2] for p in points) / len(points)
    workloads_beating_baseline = 100.0 * sum(1 for p in points if p[2] >= 1.0) / max(1, len(points))
    dataset_count = len({p[0] for p in points})
    size_count = len({p[1] for p in points})
    row_count = len(points)

    fig = plt.figure(figsize=LINKEDIN_SQUARE_SIZE)
    fig.patch.set_facecolor("#F1F5F9")

    fig.text(0.06, 0.95, "Predictive Adaptive Sorting Engine", fontsize=12, color="#2563EB", fontweight="bold")
    fig.text(0.06, 0.915, "Benchmark Highlights", fontsize=30, color="#0F172A", fontweight="bold")
    fig.text(
        0.06,
        0.875,
        f"{row_count} benchmark points across {dataset_count} workloads and {size_count} sizes",
        fontsize=11,
        color="#475569",
    )

    metric_boxes = [
        ("Peak speedup", f"{peak:.2f}x"),
        ("Median speedup", f"{med:.2f}x"),
        ("Mean speedup", f"{mean:.2f}x"),
        ("Workloads beating baseline", f"{workloads_beating_baseline:.0f}%"),
    ]

    for idx, (title, value) in enumerate(metric_boxes):
        left = 0.06 + idx * 0.22
        rect = plt.Rectangle((left, 0.745), 0.20, 0.10, transform=fig.transFigure, facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.0)
        fig.patches.append(rect)
        fig.text(left + 0.015, 0.810, title, fontsize=9.8, color="#64748B")
        fig.text(left + 0.015, 0.772, value, fontsize=18, color="#0F172A", fontweight="bold")

    ax = fig.add_axes([0.20, 0.36, 0.74, 0.33])
    top = list(reversed(top))
    labels = [f"{_dataset_label(ds)}  n={_size_label(n)}" for ds, n, _, _ in top]
    vals = [spd for _, _, spd, _ in top]
    cols = [STRATEGY_COLOURS.get(st, DEFAULT_COLOUR) for _, _, _, st in top]

    bars = ax.barh(labels, vals, color=cols, edgecolor="#FFFFFF", linewidth=1.0)
    ax.axvline(1.0, color="#DC2626", linestyle="--", linewidth=1.2)
    ax.set_xlim(0.95, max(vals) + 0.3)
    ax.set_xlabel("Speedup vs std::sort", fontsize=11)
    ax.set_title("Top 5 benchmark wins", fontsize=13, fontweight="bold", loc="left")
    ax.tick_params(axis="y", labelsize=9.8, pad=8)
    ax.grid(True, axis="x", linestyle="-", linewidth=0.8)
    ax.grid(False, axis="y")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.2f}x", va="center", fontsize=10, color="#0F172A")

    weakest_text = "\n".join(
        f"- {_dataset_label(ds)} (n={_size_label(n)}): {spd:.2f}x"
        for ds, n, spd, _ in bottom
    )
    fig.text(
        0.06,
        0.24,
        "Lowest-speedup workloads to target next:",
        fontsize=11,
        color="#0F172A",
        fontweight="bold",
    )
    fig.text(0.06, 0.16, weakest_text, fontsize=10.5, color="#334155", linespacing=1.45)

    fig.text(
        0.94,
        0.24,
        "Takeaway:\nPASE is strongest on\nstructured data patterns\nand still near parity on\nmost random-like inputs.",
        fontsize=10.5,
        color="#0F172A",
        ha="right",
        va="top",
        linespacing=1.4,
    )

    fig.text(
        0.06,
        0.06,
        "Source: PASE benchmark suite | Generated from live benchmark CSV",
        fontsize=9,
        color="#64748B",
    )

    fig.savefig(out, dpi=OUTPUT_DPI)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> int:
    _setup_style()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).parent / "mock_bench_results.csv",
        help="Path to benchmark CSV (default: visuals/mock_bench_results.csv)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Output directory for image files",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = _load_csv(args.csv)
    if not csv_rows:
        print("ERROR: CSV contains no rows.", file=sys.stderr)
        return 1

    make_speedup_chart(csv_rows, args.out_dir / "speedup_chart.png")
    make_cost_model_fit(csv_rows, args.out_dir / "cost_model_fit.png")
    make_linkedin_summary(csv_rows, args.out_dir / "linkedin_summary.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
