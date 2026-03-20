#!/usr/bin/env python3
"""
Aggregate multiple bench_results CSVs for ML threshold training (Phase 5).

Usage:
  python3 collect_training_data.py run1.csv run2.csv ... --out training_rows.csv

Each input must use the standard bench_results schema (after comment strip).
Output: one merged CSV with an added column source_file for provenance.
"""
import argparse
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pase_csv import read_csv_text_skip_comments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path, help="bench_results CSV paths")
    ap.add_argument("--out", type=Path, required=True, help="merged CSV path")
    args = ap.parse_args()

    all_rows = []
    fieldnames = None
    for path in args.inputs:
        text = read_csv_text_skip_comments(path)
        r = csv.DictReader(io.StringIO(text))
        if not r.fieldnames:
            continue
        if fieldnames is None:
            fieldnames = list(r.fieldnames) + ["source_file"]
        for row in r:
            row = dict(row)
            row["source_file"] = str(path)
            all_rows.append(row)

    if not all_rows:
        print("No rows merged", file=sys.stderr)
        return 1

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
