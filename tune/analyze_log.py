#!/usr/bin/env python3
"""Summarize bench_results.csv or PASE feedback CSV (mean, std, percentiles)."""
import argparse
import csv
import io
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pase_csv import read_csv_text_skip_comments


def percentile(sorted_vals, p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--col", default="speedup_vs_std", help="Numeric column")
    args = ap.parse_args()

    vals = []
    text = read_csv_text_skip_comments(args.csv)
    f = io.StringIO(text)
    reader = csv.DictReader(f)
    cols = reader.fieldnames or []
    col = args.col
    if col not in cols and "speedup" in cols:
        col = "speedup"
    if col not in cols:
        print(f"Column {args.col!r} not in {cols}", file=sys.stderr)
        return 1
    for row in reader:
        try:
            vals.append(float(row[col]))
        except (KeyError, ValueError):
            continue

    if not vals:
        print("No numeric values", file=sys.stderr)
        return 1

    vals.sort()
    print(
        f"n={len(vals)}  mean={statistics.mean(vals):.4f}  "
        f"median={statistics.median(vals):.4f}  "
        f"p10={percentile(vals, 10):.4f}  p90={percentile(vals, 90):.4f}  "
        f"min={min(vals):.4f}  max={max(vals):.4f}"
    )
    if len(vals) > 1:
        print(f"stdev={statistics.stdev(vals):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
