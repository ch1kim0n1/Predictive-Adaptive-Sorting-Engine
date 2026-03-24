#!/usr/bin/env python3
"""Compare two bench_results CSV files and report speedup deltas.

Exit code is non-zero when configured regression limits are violated.
"""

from __future__ import annotations

import argparse
import csv
import io
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pase_csv import read_csv_text_skip_comments


Key = tuple[str, str]


def load_rows(path: Path) -> dict[Key, dict[str, str]]:
    text = read_csv_text_skip_comments(path)
    reader = csv.DictReader(io.StringIO(text))
    out: dict[Key, dict[str, str]] = {}
    for row in reader:
        k = (row.get("dataset", ""), row.get("n", ""))
        if not k[0] or not k[1]:
            continue
        out[k] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline", type=Path, help="Baseline CSV")
    ap.add_argument("candidate", type=Path, help="Candidate CSV")
    ap.add_argument(
        "--max-worst-drop",
        type=float,
        default=0.08,
        help="Max allowed drop in speedup for any dataset-size cell (default: 0.08)",
    )
    ap.add_argument(
        "--max-median-drop",
        type=float,
        default=0.03,
        help="Max allowed median speedup drop (default: 0.03)",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=8,
        help="Show top N regressions/improvements (default: 8)",
    )
    args = ap.parse_args()

    base = load_rows(args.baseline)
    cand = load_rows(args.candidate)

    common = sorted(set(base.keys()) & set(cand.keys()))
    if not common:
        print("No overlapping dataset-size rows between baseline and candidate.", file=sys.stderr)
        return 2

    deltas: list[tuple[Key, float, float, float, str]] = []
    base_vals: list[float] = []
    cand_vals: list[float] = []

    for key in common:
        b = base[key]
        c = cand[key]
        try:
            b_sp = float(b["speedup_vs_std"])
            c_sp = float(c["speedup_vs_std"])
        except (KeyError, ValueError):
            continue
        delta = c_sp - b_sp
        strat = c.get("strategy", "")
        deltas.append((key, b_sp, c_sp, delta, strat))
        base_vals.append(b_sp)
        cand_vals.append(c_sp)

    if not deltas:
        print("No numeric speedup rows to compare.", file=sys.stderr)
        return 2

    base_med = statistics.median(base_vals)
    cand_med = statistics.median(cand_vals)
    base_mean = statistics.mean(base_vals)
    cand_mean = statistics.mean(cand_vals)

    worst = min(deltas, key=lambda x: x[3])
    best = max(deltas, key=lambda x: x[3])

    print("Aggregate")
    print(f"- rows: {len(deltas)}")
    print(f"- baseline median: {base_med:.4f}")
    print(f"- candidate median: {cand_med:.4f} (delta {cand_med - base_med:+.4f})")
    print(f"- baseline mean: {base_mean:.4f}")
    print(f"- candidate mean: {cand_mean:.4f} (delta {cand_mean - base_mean:+.4f})")
    print(
        "- best cell: "
        f"{best[0][0]} n={best[0][1]} {best[1]:.3f}->{best[2]:.3f} ({best[3]:+.3f}) [{best[4]}]"
    )
    print(
        "- worst cell: "
        f"{worst[0][0]} n={worst[0][1]} {worst[1]:.3f}->{worst[2]:.3f} ({worst[3]:+.3f}) [{worst[4]}]"
    )

    print("\nTop regressions")
    for key, b_sp, c_sp, delta, strat in sorted(deltas, key=lambda x: x[3])[: args.top]:
        print(f"- {key[0]} n={key[1]} {b_sp:.3f}->{c_sp:.3f} ({delta:+.3f}) [{strat}]")

    print("\nTop improvements")
    for key, b_sp, c_sp, delta, strat in sorted(deltas, key=lambda x: x[3], reverse=True)[: args.top]:
        print(f"- {key[0]} n={key[1]} {b_sp:.3f}->{c_sp:.3f} ({delta:+.3f}) [{strat}]")

    failed = False
    if (cand_med - base_med) < -args.max_median_drop:
        failed = True
        print(
            "\nFAIL: candidate median drop exceeds limit "
            f"({cand_med - base_med:+.4f} < {-args.max_median_drop:+.4f})"
        )

    if worst[3] < -args.max_worst_drop:
        failed = True
        print(
            "FAIL: worst single-cell drop exceeds limit "
            f"({worst[3]:+.4f} < {-args.max_worst_drop:+.4f})"
        )

    if failed:
        return 1

    print("\nPASS: regression limits respected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
