from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import common  # noqa: F401


OPEN_IMAGES_S3_BASE = "https://open-images-dataset.s3.amazonaws.com"


def download_images(
    image_ids_file: Path,
    output_root: Path,
    max_images: int | None = None,
    base_url: str = OPEN_IMAGES_S3_BASE,
) -> dict[str, int]:
    rows = [line.strip() for line in image_ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_images is not None:
        rows = rows[:max_images]

    downloaded = 0
    skipped = 0
    failed = 0
    for row in rows:
        split, image_id = row.split("/", 1)
        output_dir = output_root / split / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"{image_id}.jpg"
        if target.exists():
            skipped += 1
            continue
        url = f"{base_url.rstrip('/')}/{split}/{image_id}.jpg"
        try:
            urlretrieve(url, target)
            downloaded += 1
        except Exception:
            failed += 1
            if target.exists():
                target.unlink()
    return {"downloaded": downloaded, "skipped": skipped, "failed": failed}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Open Images files from an image-id manifest.")
    parser.add_argument("--image-ids", required=True)
    parser.add_argument("--output-root", default="data/openimages_person")
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args()

    counts = download_images(Path(args.image_ids), Path(args.output_root), max_images=args.max_images)
    print(f"downloaded={counts['downloaded']} skipped={counts['skipped']} failed={counts['failed']}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
