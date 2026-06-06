from __future__ import annotations

import argparse
from pathlib import Path

import common  # noqa: F401
import yaml
from ultralytics import YOLO

from src.config import load_config, resolve_model_device


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune a YOLO model for person detection.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--project", default="")
    parser.add_argument("--name", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_root = Path(args.dataset_root)
    dataset_yaml = dataset_root / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_root.resolve()),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": {0: "person"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    device = resolve_model_device(config.model.device)
    model_path = args.resume or args.model_path or config.model.path
    model = YOLO(model_path)
    train_args = {
        "data": str(dataset_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "device": device,
        "batch": args.batch,
        "workers": args.workers,
        "patience": args.patience,
        "hsv_h": 0.01,
        "hsv_s": 0.2,
        "hsv_v": 0.25,
        "degrees": 2.0,
        "translate": 0.08,
        "scale": 0.15,
        "shear": 0.0,
        "perspective": 0.0005,
        "fliplr": 0.0,
        "mosaic": 0.2,
        "mixup": 0.0,
        "erasing": 0.2,
    }
    if args.resume:
        train_args["resume"] = True
    if args.project:
        train_args["project"] = args.project
    if args.name:
        train_args["name"] = args.name
    model.train(**train_args)
    print(f"dataset_yaml={dataset_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
