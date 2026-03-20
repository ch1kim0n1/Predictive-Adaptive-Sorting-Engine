#!/usr/bin/env python3
"""
Train a simple model to suggest dispatcher-friendly JSON (Phase 5).

Requires: pandas, scikit-learn (optional). Without sklearn, prints heuristic only.

Input: merged CSV from collect_training_data.py with columns including
sortedness, dup_ratio, entropy, avg_run_length, n, strategy, pase_ms, std_ms.

Output: JSON suitable as ~/.pase/ml_thresholds.json (partial keys).

Example:
  python3 train_ml_model.py training_rows.csv --out ml_thresholds.json
"""
import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pase_csv import read_csv_text_skip_comments

try:
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    np = None
    RandomForestRegressor = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, default=Path("ml_thresholds.json"))
    args = ap.parse_args()

    text = read_csv_text_skip_comments(args.csv)
    r = csv.DictReader(io.StringIO(text))
    rows = list(r)
    if len(rows) < 5:
        print("Need >= 5 rows for meaningful training", file=sys.stderr)
        return 1

    # Heuristic fallback: nudge dup toward median of dup_ratio when strategy is THREE_WAY_QS
    dup_samples = []
    for row in rows:
        try:
            if row.get("strategy", "").strip() != "THREE_WAY_QS":
                continue
            dup_samples.append(float(row["dup_ratio"]))
        except (KeyError, ValueError):
            continue
    dup_guess = sum(dup_samples) / max(1, len(dup_samples)) if dup_samples else 0.32

    out = {
        "_comment": "Phase 5 ML/heuristic export; merge manually with optimized_thresholds.json",
        "dup": round(max(0.12, min(0.55, dup_guess * 0.95)), 4),
    }

    if np is not None and RandomForestRegressor is not None:
        X = []
        y_speed = []
        for row in rows:
            try:
                feat = [
                    float(row["sortedness"]),
                    float(row["dup_ratio"]),
                    float(row["entropy"]),
                    float(row["avg_run_length"]),
                    math.log2(max(2, float(row["n"]))),
                ]
                speed = float(row["std_ms"]) / max(1e-9, float(row["pase_ms"]))
                X.append(feat)
                y_speed.append(speed)
            except (KeyError, ValueError):
                continue
        if len(X) >= 8:
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y_speed, dtype=np.float64)
            m = RandomForestRegressor(
                n_estimators=32, max_depth=6, random_state=0
            )
            m.fit(X, y)
            # Feature-importance guided nudge (interpretability over accuracy here)
            imp = m.feature_importances_
            if imp[1] > 0.25:  # dup_ratio important
                out["dup"] = round(float(out["dup"]) * 0.98, 4)
            print("# sklearn RandomForest fitted; importances:", imp, file=sys.stderr)

    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
