from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import cv2
import psutil


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Detector(Protocol):
    def detect(self, frame) -> list[object]:
        ...


@dataclass
class BenchmarkFrameResult:
    image: str
    repeat: int
    elapsed_ms: float
    detection_count: int


@dataclass
class BenchmarkSummary:
    image_count: int
    total_inferences: int
    total_detections: int
    elapsed_seconds: float
    average_latency_ms: float
    p95_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    fps: float
    average_process_cpu_percent: float
    peak_rss_mb: float
    model_path: str
    inference_size: int
    confidence_threshold: float
    device: str
    processor: str
    logical_cpus: int


@dataclass
class BenchmarkReport:
    summary: BenchmarkSummary
    frames: list[BenchmarkFrameResult]


def collect_image_paths(images_dir: Path, limit: int | None = None) -> list[Path]:
    paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    return paths[:limit] if limit is not None else paths


def _percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=20, method="inclusive")[18]


def run_inference_benchmark(
    *,
    detector: Detector,
    image_paths: list[Path],
    model_path: str,
    inference_size: int,
    confidence_threshold: float,
    device: str,
    repeats: int = 1,
    warmup: int = 1,
) -> BenchmarkReport:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if not image_paths:
        raise ValueError("At least one image is required")

    frames = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss

    warmup_paths = image_paths[: max(0, min(warmup, len(image_paths)))]
    for image_path in warmup_paths:
        frame = cv2.imread(str(image_path))
        if frame is not None:
            detector.detect(frame)

    cpu_before = process.cpu_times()
    started = time.perf_counter()
    for repeat in range(repeats):
        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            inference_started = time.perf_counter()
            detections = detector.detect(frame)
            elapsed_ms = (time.perf_counter() - inference_started) * 1000
            frames.append(
                BenchmarkFrameResult(
                    image=image_path.name,
                    repeat=repeat + 1,
                    elapsed_ms=round(elapsed_ms, 3),
                    detection_count=len(detections),
                )
            )
            peak_rss = max(peak_rss, process.memory_info().rss)
    elapsed_seconds = time.perf_counter() - started
    cpu_after = process.cpu_times()

    latency_values = [frame.elapsed_ms for frame in frames]
    cpu_seconds = (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)
    logical_cpus = psutil.cpu_count(logical=True) or 1
    average_cpu_percent = (cpu_seconds / elapsed_seconds / logical_cpus * 100) if elapsed_seconds > 0 else 0.0
    total_inferences = len(frames)

    summary = BenchmarkSummary(
        image_count=len(image_paths),
        total_inferences=total_inferences,
        total_detections=sum(frame.detection_count for frame in frames),
        elapsed_seconds=round(elapsed_seconds, 3),
        average_latency_ms=round(statistics.fmean(latency_values), 3) if latency_values else 0.0,
        p95_latency_ms=round(_percentile_95(latency_values), 3),
        min_latency_ms=round(min(latency_values), 3) if latency_values else 0.0,
        max_latency_ms=round(max(latency_values), 3) if latency_values else 0.0,
        fps=round(total_inferences / elapsed_seconds, 3) if elapsed_seconds > 0 else 0.0,
        average_process_cpu_percent=round(average_cpu_percent, 2),
        peak_rss_mb=round(peak_rss / (1024 * 1024), 2),
        model_path=model_path,
        inference_size=inference_size,
        confidence_threshold=confidence_threshold,
        device=device,
        processor=platform.processor() or platform.machine(),
        logical_cpus=logical_cpus,
    )
    return BenchmarkReport(summary=summary, frames=frames)


def write_benchmark_json(path: str | Path, report: BenchmarkReport) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "summary": asdict(report.summary),
                "frames": [asdict(frame) for frame in report.frames],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def render_benchmark_markdown(report: BenchmarkReport) -> str:
    summary = report.summary
    lines = [
        "# Inference Benchmark",
        "",
        "## Summary",
        "",
        f"- model: `{summary.model_path}`",
        f"- device: `{summary.device}`",
        f"- inference size: `{summary.inference_size}`",
        f"- confidence threshold: `{summary.confidence_threshold}`",
        f"- images: `{summary.image_count}`",
        f"- total inferences: `{summary.total_inferences}`",
        f"- elapsed seconds: `{summary.elapsed_seconds}`",
        f"- FPS: `{summary.fps}`",
        f"- average latency ms: `{summary.average_latency_ms}`",
        f"- p95 latency ms: `{summary.p95_latency_ms}`",
        f"- average process CPU percent: `{summary.average_process_cpu_percent}`",
        f"- peak RSS MB: `{summary.peak_rss_mb}`",
        f"- processor: `{summary.processor}`",
        f"- logical CPUs: `{summary.logical_cpus}`",
        "",
    ]
    return "\n".join(lines)


def write_benchmark_markdown(path: str | Path, report: BenchmarkReport) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_benchmark_markdown(report), encoding="utf-8")
    return output_path
