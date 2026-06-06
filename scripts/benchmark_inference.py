from __future__ import annotations

import argparse
from pathlib import Path

import common  # noqa: F401

from src.config import load_config, resolve_model_device
from src.detector import YOLOPersonDetector
from src.inference_benchmark import (
    collect_image_paths,
    run_inference_benchmark,
    write_benchmark_json,
    write_benchmark_markdown,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark person-detector inference latency and resource use.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--images", required=True)
    parser.add_argument("--output-dir", default="reports/inference_benchmark")
    parser.add_argument("--model-path")
    parser.add_argument("--inference-size", type=int)
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.model_path:
        config.model.path = args.model_path
    if args.inference_size:
        config.model.inference_size = args.inference_size
    if args.confidence_threshold is not None:
        config.model.confidence_threshold = args.confidence_threshold

    images = collect_image_paths(Path(args.images), limit=args.limit)
    detector = YOLOPersonDetector(config)
    resolved_device = resolve_model_device(config.model.device)
    report = run_inference_benchmark(
        detector=detector,
        image_paths=images,
        model_path=config.model.path,
        inference_size=config.model.inference_size,
        confidence_threshold=config.model.confidence_threshold,
        device=resolved_device,
        repeats=args.repeats,
        warmup=args.warmup,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_benchmark_json(output_dir / "inference_benchmark.json", report)
    write_benchmark_markdown(output_dir / "inference_benchmark.md", report)
    print(f"summary={output_dir / 'inference_benchmark.md'}")
    print(f"json={output_dir / 'inference_benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
