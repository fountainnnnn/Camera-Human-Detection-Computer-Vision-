from pathlib import Path

import src.inference_benchmark as benchmark


class FakeDetector:
    def detect(self, frame):
        return [object(), object()]


def test_run_inference_benchmark_summarizes_latency_cpu_and_memory(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake")

    monkeypatch.setattr(benchmark.cv2, "imread", lambda _path: object())
    report = benchmark.run_inference_benchmark(
        detector=FakeDetector(),
        image_paths=[image_path],
        model_path="models/yolo11n.pt",
        inference_size=512,
        confidence_threshold=0.6,
        device="cpu",
        repeats=2,
        warmup=1,
    )

    assert report.summary.image_count == 1
    assert report.summary.total_inferences == 2
    assert report.summary.total_detections == 4
    assert report.summary.fps > 0
    assert report.summary.peak_rss_mb > 0
    assert "Inference Benchmark" in benchmark.render_benchmark_markdown(report)


def test_collect_image_paths_filters_supported_extensions(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"")
    (tmp_path / "b.png").write_bytes(b"")
    (tmp_path / "c.txt").write_text("no", encoding="utf-8")

    assert [path.name for path in benchmark.collect_image_paths(tmp_path)] == ["a.jpg", "b.png"]
