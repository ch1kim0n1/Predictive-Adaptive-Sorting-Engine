#!/usr/bin/env python3
"""
generate_visuals.py — Generate benchmark PNG charts for PASE.

Usage (mock data, no build required):
    python3 generate_visuals.py

Usage (live benchmark data):
    # 1. Build and run the benchmark suite:
    #    cmake -B build && cmake --build build --target bench_results
    #    ./build/bench/bench_results --out visuals/bench_results.csv
    # 2. Run this script pointing at the live CSV:
    python3 generate_visuals.py --csv bench_results.csv

Outputs
-------
  speedup_chart.png  — Bar chart: PASE speedup vs std::sort per workload/size.
  cost_model_fit.png — Line chart: PASE vs std::sort latency (ms) across array
                       sizes, grouped by dataset, illustrating where the cost
                       model predicts GPU/specialist wins.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("ERROR: matplotlib not found.  Run:  pip install matplotlib", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
GPU_DISPATCH_THRESHOLD = 100_000  # n at which PASE considers GPU dispatch

STRATEGY_COLOURS = {
    "insertion":  "#2196F3",   # blue
    "run_merge":  "#4CAF50",   # green
    "three_way":  "#FF9800",   # orange
    "introsort":  "#9E9E9E",   # grey
    "gpu":        "#E91E63",   # pink/red
    "fallback":   "#795548",   # brown
}
DEFAULT_COLOUR = "#607D8B"


def _load_csv(path: Path) -> list[dict]:
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


# ---------------------------------------------------------------------------
# Chart 1 — speedup_chart.png
# ---------------------------------------------------------------------------
def make_speedup_chart(rows: list[dict], out: Path) -> None:
    """Bar chart of speedup_vs_std grouped by dataset, one bar per (dataset, n)."""
    speed_col = "speedup_vs_std" if any("speedup_vs_std" in r for r in rows) else "speedup"

    # Aggregate: keep highest-n entry per dataset for a cleaner per-workload view
    # plus a GPU-tier view for n >= 100k.
    summary: dict[str, list[tuple[int, float, str]]] = {}
    for r in rows:
        ds = r.get("dataset", "?")
        try:
            n = int(r.get("n", 0))
            spd = float(r[speed_col])
        except (ValueError, KeyError):
            continue
        strategy = r.get("strategy", "fallback")
        summary.setdefault(ds, []).append((n, spd, strategy))

    # Build label/value lists, sorted by dataset name then n.
    labels, values, colours = [], [], []
    for ds in sorted(summary):
        for n, spd, strat in sorted(summary[ds]):
            size_tag = f"{n//1000}k" if n < 1_000_000 else f"{n//1_000_000}M"
            labels.append(f"{ds}\n(n={size_tag})")
            values.append(spd)
            colours.append(STRATEGY_COLOURS.get(strat, DEFAULT_COLOUR))

    fig, ax = plt.subplots(figsize=(max(14, len(labels) * 0.55), 6))
    bars = ax.bar(labels, values, color=colours, edgecolor="white", linewidth=0.5)

    # Baseline reference line
    ax.axhline(1.0, color="#F44336", linestyle="--", linewidth=1.2, label="std::sort baseline (1.0×)")

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{val:.2f}×",
            ha="center", va="bottom", fontsize=7.5, color="#212121",
        )

    ax.set_ylim(0, max(values) * 1.18)
    ax.set_ylabel("Speedup vs std::sort  (higher is better)", fontsize=11)
    ax.set_title("PASE v1.2 — Speedup vs std::sort by Workload & Array Size", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)

    # Legend for strategies
    legend_patches = [
        mpatches.Patch(color=c, label=s.replace("_", " ").title())
        for s, c in STRATEGY_COLOURS.items()
    ]
    legend_patches.append(
        plt.Line2D([0], [0], color="#F44336", linestyle="--", linewidth=1.2, label="std::sort baseline")
    )
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Chart 2 — cost_model_fit.png
# ---------------------------------------------------------------------------
def make_cost_model_fit(rows: list[dict], out: Path) -> None:
    """
    Line chart: PASE vs std::sort latency (ms) across array sizes per dataset.
    Illustrates where PASE's cost model dispatches to a specialist or GPU and
    how the latency curves diverge from the std::sort baseline.
    """
    # Group rows: dataset → {n: (pase_ms, std_ms, strategy)}
    grouped: dict[str, dict[int, tuple[float, float, str]]] = {}
    for r in rows:
        ds = r.get("dataset", "?")
        try:
            n = int(r.get("n", 0))
            pase_ms = float(r.get("pase_ms", 0))
            std_ms = float(r.get("std_ms", 0))
        except ValueError:
            continue
        strategy = r.get("strategy", "introsort")
        grouped.setdefault(ds, {})[n] = (pase_ms, std_ms, strategy)

    # Select datasets that show interesting divergence
    highlight = ["sorted", "heavy_dup", "long_runs", "random"]
    datasets = [d for d in highlight if d in grouped] + [
        d for d in sorted(grouped) if d not in highlight
    ]

    cmap = plt.get_cmap("tab10")
    ds_colours = {ds: cmap(i % 10) for i, ds in enumerate(datasets)}

    fig, ax = plt.subplots(figsize=(11, 6))

    for ds in datasets:
        entries = sorted(grouped[ds].items())  # sort by n
        ns = [e[0] for e in entries]
        pase_vals = [e[1][0] for e in entries]
        std_vals = [e[1][1] for e in entries]
        col = ds_colours[ds]
        label_ds = ds.replace("_", " ")
        ax.plot(ns, pase_vals, color=col, marker="o", linewidth=2,
                label=f"PASE — {label_ds}")
        ax.plot(ns, std_vals, color=col, marker="s", linewidth=1.2,
                linestyle="--", alpha=0.55, label=f"std::sort — {label_ds}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Array size  n  (log scale)", fontsize=11)
    ax.set_ylabel("Sort latency  (ms, log scale)", fontsize=11)
    ax.set_title(
        "PASE Cost-Model Fit — Latency vs Array Size\n"
        "(solid = PASE, dashed = std::sort; lower is better)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=7.5, loc="upper left", ncol=2, framealpha=0.85)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    # Annotation: GPU threshold
    ax.axvline(GPU_DISPATCH_THRESHOLD, color="#E91E63", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(GPU_DISPATCH_THRESHOLD * 1.05, ax.get_ylim()[0] * 1.15, "GPU threshold\n(n ≥ 100k)",
            color="#E91E63", fontsize=7.5, va="bottom")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).parent / "mock_bench_results.csv",
        help="Path to benchmark CSV (default: mock_bench_results.csv alongside this script)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Output directory for PNG files (default: same directory as this script)",
    )
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_csv(args.csv)
    if not rows:
        print("ERROR: CSV contains no data rows.", file=sys.stderr)
        return 1

    make_speedup_chart(rows, args.out_dir / "speedup_chart.png")
    make_cost_model_fit(rows, args.out_dir / "cost_model_fit.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
