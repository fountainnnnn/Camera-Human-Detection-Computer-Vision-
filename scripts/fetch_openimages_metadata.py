from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import common  # noqa: F401


DEFAULT_URLS = {
    "class_descriptions": "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv",
    "validation_boxes": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test_boxes": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
}


def fetch_metadata(output_dir: Path, include_train_boxes: bool = False) -> dict[str, Path]:
    urls = dict(DEFAULT_URLS)
    if include_train_boxes:
        urls["train_boxes"] = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Path] = {}
    for name, url in urls.items():
        target = output_dir / Path(url).name
        if not target.exists():
            urlretrieve(url, target)
        downloaded[name] = target
    return downloaded


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch official Open Images metadata CSVs used by this project.")
    parser.add_argument("--output-dir", default="data/sources/openimages")
    parser.add_argument(
        "--include-train-boxes",
        action="store_true",
        help="Also download the large training bounding-box CSV. Validation/test metadata are enough for a first proof run.",
    )
    args = parser.parse_args()

    downloaded = fetch_metadata(Path(args.output_dir), include_train_boxes=args.include_train_boxes)
    for name, path in downloaded.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
