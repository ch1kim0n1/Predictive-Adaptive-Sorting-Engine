#!/usr/bin/env python3
"""Plot bench_results.csv: speedup by dataset (uses speedup_vs_std if present)."""
import argparse
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pase_csv import read_csv_text_skip_comments

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Install: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, default=Path("bench_results.csv"), nargs="?")
    ap.add_argument("-o", "--out", type=Path, default=Path("bench_speedup.png"))
    args = ap.parse_args()

    text = read_csv_text_skip_comments(args.csv)
    rows = list(csv.DictReader(io.StringIO(text)))

    if not rows:
        print("No rows in CSV", file=sys.stderr)
        return 1

    speed_col = "speedup_vs_std" if rows and "speedup_vs_std" in rows[0] else "speedup"

    labels = []
    speedups = []
    for r in rows:
        key = f"{r.get('dataset','?')}"
        if "n" in r:
            key += f" n={r['n']}"
        labels.append(key)
        speedups.append(float(r[speed_col]))

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.35), 4))
    ax.bar(labels, speedups, color="steelblue")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel(f"speedup ({speed_col})")
    ax.set_title("PASE vs baseline")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
