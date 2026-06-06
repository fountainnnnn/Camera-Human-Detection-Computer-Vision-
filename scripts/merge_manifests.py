from __future__ import annotations

import argparse
from pathlib import Path

import common  # noqa: F401

from src.manifest_tools import merge_manifests


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge multiple frame manifests into a single evaluation manifest.")
    parser.add_argument("--manifest", action="append", required=True, help="Input manifest path. Repeat per file.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    merged = merge_manifests(args.manifest)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"manifest={output_path}")
    print(f"rows={len(merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
