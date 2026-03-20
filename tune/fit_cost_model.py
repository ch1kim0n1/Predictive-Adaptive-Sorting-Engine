#!/usr/bin/env python3
"""
Offline cost-model nudge from bench_results CSV.

Uses per-strategy median ratio actual_ms / pred_cpu_ms to suggest cost_fit
scales (heuristic, not full ML). Merge the printed JSON under "cost_fit" in
your PASE_CONFIG file.

Requires: pandas optional; stdlib-only fallback below.
"""
import argparse
import csv
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pase_csv import read_csv_text_skip_comments


def median(vals):
    if not vals:
        return 1.0
    s = sorted(vals)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="bench_results.csv from bench_results")
    ap.add_argument(
        "--min-rows",
        type=int,
        default=2,
        help="Minimum rows per strategy to emit a scale",
    )
    args = ap.parse_args()

    by_strategy = defaultdict(list)
    text = read_csv_text_skip_comments(args.csv)
    r = csv.DictReader(io.StringIO(text))
    need = {"strategy", "pase_ms", "pred_cpu_ms"}
    if not r.fieldnames or not need.issubset(set(r.fieldnames)):
        print("CSV must include strategy, pase_ms, pred_cpu_ms", file=sys.stderr)
        return 1
    for row in r:
        try:
            actual = float(row["pase_ms"])
            pred = float(row["pred_cpu_ms"])
        except (KeyError, ValueError):
            continue
        if pred <= 0 or actual <= 0 or math.isnan(pred) or math.isnan(actual):
            continue
        by_strategy[row["strategy"].strip()].append(actual / pred)

    if not by_strategy:
        print("No usable rows", file=sys.stderr)
        return 1

    mapping = {
        "INTROSORT": "introsort",
        "RUN_MERGE_OPT": "run_merge",
        "THREE_WAY_QS": "three_way",
        "INSERTION_OPT": "insertion",
    }

    cost_fit = {}
    for strat, ratios in sorted(by_strategy.items()):
        if len(ratios) < args.min_rows:
            continue
        key = mapping.get(strat)
        if not key:
            continue
        m = median(ratios)
        cost_fit[key] = round(max(0.2, min(5.0, m)), 4)

    if not cost_fit:
        print("Not enough rows per strategy; lower --min-rows", file=sys.stderr)
        return 1

    _lo, _hi = 0.2, 5.0
    print(
        f"# Clamp / bounds: each emitted scale is clamped to [{_lo}, {_hi}] "
        "(see fit_cost_model.py). If stderr ratios sit on a clamp, add data or "
        "inspect that strategy.",
        file=sys.stderr,
    )

    # Report: median |log ratio| as rough calibration error
    print("# Per-strategy median(actual/pred), n rows:", file=sys.stderr)
    for strat, ratios in sorted(by_strategy.items()):
        med = median(ratios)
        elog = median([abs(math.log(max(1e-9, r))) for r in ratios])
        print(f"#   {strat}: median_ratio={med:.4f} n={len(ratios)} mae_log={elog:.4f}", file=sys.stderr)

    out = {"cost_fit": cost_fit}
    print(json.dumps(out, indent=2))
    print(
        "\n# Paste under ~/.pase/optimized_thresholds.json (merge with existing keys).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
