from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import common  # noqa: F401


KAIST_DOWNLOADS = {
    "preview": {
        "gdrive_id": "11nhHpmuh2FUjrLNfGs51R2Mqqy1GTjY8",
        "description": "Official KAIST preview set, approximately 1.44 GB.",
    },
    "full": {
        "gdrive_id": "1sBcAmFqNJmNMBZdMtKmO2X4BRjKPyKMc",
        "description": "Official KAIST full set, approximately 36.32 GB.",
    },
}


def fetch_kaist_assets(output_dir: Path, variant: str) -> Path:
    if variant not in KAIST_DOWNLOADS:
        raise ValueError(f"unknown KAIST variant: {variant}")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"kaist_{variant}"
    command = [
        sys.executable,
        "-m",
        "gdown",
        KAIST_DOWNLOADS[variant]["gdrive_id"],
        "-O",
        str(target),
        "--fuzzy",
    ]
    subprocess.run(command, check=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch official KAIST multispectral pedestrian benchmark assets.")
    parser.add_argument("--output-dir", default="data/sources/kaist")
    parser.add_argument("--variant", choices=sorted(KAIST_DOWNLOADS), default="preview")
    parser.add_argument("--print-only", action="store_true", help="Print the official gdown command without downloading.")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    target = output_dir / f"kaist_{args.variant}"
    command = (
        f"{sys.executable} -m gdown {KAIST_DOWNLOADS[args.variant]['gdrive_id']} "
        f"-O {target} --fuzzy"
    )
    if args.print_only:
        print(command)
        return 0

    downloaded = fetch_kaist_assets(output_dir, args.variant)
    print(f"downloaded={downloaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
