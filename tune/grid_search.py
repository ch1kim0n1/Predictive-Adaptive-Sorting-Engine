#!/usr/bin/env python3
"""
Write a small grid of threshold JSON files for manual benchmarking.
Does not run the C++ binary; use after editing paths.
"""
import argparse
import json
from itertools import product
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "out_dir",
        type=Path,
        nargs="?",
        default=Path("grid_configs"),
        help="Directory for config_*.json",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sorted_vals = [0.88, 0.90, 0.92]
    win_vals = [0.80, 0.85, 0.90]
    i = 0
    for s, w in product(sorted_vals, win_vals):
        i += 1
        cfg = {
            "sorted": s,
            "gpu_win_factor": w,
            "run_merge": 32,
            "dup": 0.32,
            "min_gpu": 250000,
            "max_insertion_n": 384,
            "strategy_guardrail": 2.25,
            "gpu_rel_margin": 1.12,
        }
        p = args.out_dir / f"config_{i:03d}.json"
        p.write_text(json.dumps(cfg, indent=2))
        print(p)
    print(f"Wrote {i} files under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
