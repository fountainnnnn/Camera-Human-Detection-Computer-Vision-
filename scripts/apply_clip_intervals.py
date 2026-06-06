from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import common  # noqa: F401

from src.interval_tools import apply_intervals_to_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply clip-level human-present intervals to a frame manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--intervals", required=True)
    parser.add_argument("--output", help="Optional output path. Defaults to in-place update.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    intervals_path = Path(args.intervals)
    output_path = Path(args.output) if args.output else manifest_path

    manifest = pd.read_csv(manifest_path)
    intervals = pd.read_csv(intervals_path)
    labeled = apply_intervals_to_manifest(manifest, intervals)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(output_path, index=False)
    print(f"manifest={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
