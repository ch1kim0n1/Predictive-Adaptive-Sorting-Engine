#!/usr/bin/env python3
"""
Score a bench_results CSV: mean/median/min speedup_vs_std and worst slowdown.
Use after: PASE_CONFIG=... ./bench/bench_results --out results.csv
"""
import argparse
import csv
import io
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pase_csv import read_csv_text_skip_comments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument(
        "--col",
        default="speedup_vs_std",
        help="Column to aggregate (higher is better for PASE)",
    )
    args = ap.parse_args()

    vals = []
    text = read_csv_text_skip_comments(args.csv)
    r = csv.DictReader(io.StringIO(text))
    if args.col not in (r.fieldnames or []):
        print(f"Missing column {args.col!r}", file=sys.stderr)
        return 1
    for row in r:
        try:
            vals.append(float(row[args.col]))
        except (KeyError, ValueError):
            pass

    if not vals:
        print("No values", file=sys.stderr)
        return 1

    vals.sort()
    print(
        f"n={len(vals)}  mean={statistics.mean(vals):.4f}  "
        f"median={statistics.median(vals):.4f}  min={min(vals):.4f}  "
        f"max={max(vals):.4f}"
    )
    if len(vals) > 1:
        print(f"stdev={statistics.stdev(vals):.4f}")
    print(f"worst_vs_std (min speedup) = {min(vals):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
