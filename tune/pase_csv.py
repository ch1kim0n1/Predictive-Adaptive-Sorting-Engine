"""Skip # comment lines (e.g. pase_bench_suite) before CSV parsing."""
import io
from pathlib import Path


def read_csv_text_skip_comments(path: Path) -> str:
    lines = [
        ln for ln in path.read_text().splitlines() if not ln.startswith("#")
    ]
    return "\n".join(lines)
