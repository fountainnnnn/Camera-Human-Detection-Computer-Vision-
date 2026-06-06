from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import common  # noqa: F401


DEFAULT_URLS = {
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
}


def fetch_coco_assets(output_dir: Path, include_val_images: bool = True) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    urls = {"annotations": DEFAULT_URLS["annotations"]}
    if include_val_images:
        urls["val_images"] = DEFAULT_URLS["val_images"]

    downloaded: dict[str, Path] = {}
    for name, url in urls.items():
        archive = output_dir / Path(url).name
        if not archive.exists():
            urlretrieve(url, archive)
        downloaded[name] = archive
        marker = output_dir / f".{archive.stem}.extracted"
        if not marker.exists():
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(output_dir)
            marker.write_text("ok\n", encoding="utf-8")
    return downloaded


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch official COCO 2017 annotations and validation images.")
    parser.add_argument("--output-dir", default="data/sources/coco")
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Download only the annotations archive. Images are needed before preparing a trainable subset.",
    )
    args = parser.parse_args()

    downloaded = fetch_coco_assets(Path(args.output_dir), include_val_images=not args.annotations_only)
    for name, path in downloaded.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
