from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_top_level_project_artifacts_exist() -> None:
    expected = [
        "README.md",
        "DATASETS.md",
        "requirements.txt",
        "config.example.yaml",
        ".env.example",
        ".gitignore",
        "src",
        "scripts",
        "deploy",
    ]
    for name in expected:
        assert (PROJECT_ROOT / name).exists(), name


def test_readme_covers_setup_runtime_and_deployment() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "## Windows setup" in readme
    assert "## Configuration" in readme
    assert "## Common commands" in readme
    assert "## Deployment" in readme
    assert "## Troubleshooting" in readme
    assert "python .\\src\\main.py" in readme
    assert ".\\deploy\\install_task_scheduler.ps1" in readme
    assert "test_webcam.py" in readme
    assert "test_rtsp.py" in readme
    assert "test_telegram.py" in readme
    assert "benchmark_inference.py" in readme
    assert "monitor_runtime.py" in readme
    assert "runtime_monitor_summary.md" in readme
    assert "build_acceptance_report.py" in readme
    assert "acceptance_report.md" in readme
    assert "tune_alert_thresholds.py" in readme
    assert "alert_threshold_recommendations.md" in readme
    assert "analyze_alert_failures.py" in readme
    assert "alert_failure_analysis.md" in readme
    assert "review_validation_run.py" in readme
    assert "validation_review_index.md" in readme
    assert "smoke_validation_review.py" in readme
    assert "prepare_openimages_person.py" in readme
    assert "fetch_openimages_metadata.py" in readme
    assert "download_openimages_images.py" in readme
    assert "fetch_kaist_assets.py" in readme
    assert "prepare_kaist_person.py" in readme
    assert "dataset_acquisition_plan.md" in readme


def test_dataset_notes_cover_sources_licenses_and_preprocessing() -> None:
    datasets = (PROJECT_ROOT / "DATASETS.md").read_text(encoding="utf-8")

    assert "ExDark" in datasets
    assert "LLVIP" in datasets
    assert "COCO" in datasets
    assert "Open Images" in datasets
    assert "KAIST" in datasets
    assert "FLIR" in datasets
    assert "License note" in datasets
    assert "Preprocessing" in datasets
    assert "Do not scrape random copyrighted web videos or images" in datasets


def test_requirements_and_env_examples_cover_runtime_dependencies() -> None:
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    config_example = (PROJECT_ROOT / "config.example.yaml").read_text(encoding="utf-8")

    for dep in ["opencv-python", "ultralytics", "torch", "PyYAML", "requests", "python-dotenv", "psutil", "numpy"]:
        assert dep in requirements
    assert "TELEGRAM_BOT_TOKEN=" in env_example
    assert "TELEGRAM_CHAT_ID=" in env_example
    assert 'mode: "webcam"' in config_example
    assert "rtsp_detection_url:" in config_example
    assert "rtsp_snapshot_url:" in config_example
    assert "webcam_index:" in config_example


def test_source_and_script_inventory_exists() -> None:
    src_expected = [
        "__init__.py",
        "config.py",
        "camera.py",
        "detector.py",
        "alert_logic.py",
        "telegram_bot.py",
        "logger.py",
        "health.py",
        "metrics.py",
        "dataset_catalog.py",
        "openimages_tools.py",
        "coco_tools.py",
        "kaist_tools.py",
        "acceptance.py",
        "alert_tuning.py",
        "error_analysis.py",
        "inference_benchmark.py",
        "runtime_validation.py",
        "utils.py",
        "main.py",
    ]
    script_expected = [
        "download_datasets.py",
        "fetch_coco_assets.py",
        "fetch_kaist_assets.py",
        "download_openimages_images.py",
        "fetch_openimages_metadata.py",
        "prepare_coco_person.py",
        "prepare_kaist_person.py",
        "prepare_openimages_person.py",
        "prepare_dataset.py",
        "convert_annotations.py",
        "train.py",
        "evaluate.py",
        "test_image.py",
        "test_video.py",
        "test_webcam.py",
        "test_rtsp.py",
        "test_telegram.py",
        "capture_webcam_samples.py",
        "capture_rtsp_samples.py",
        "benchmark_inference.py",
        "monitor_runtime.py",
        "build_acceptance_report.py",
        "tune_alert_thresholds.py",
        "analyze_alert_failures.py",
        "review_validation_run.py",
        "smoke_validation_review.py",
    ]
    for name in src_expected:
        assert (PROJECT_ROOT / "src" / name).exists(), name
    for name in script_expected:
        assert (PROJECT_ROOT / "scripts" / name).exists(), name


def test_validation_runbook_covers_runtime_monitoring() -> None:
    validation = (PROJECT_ROOT / "VALIDATION.md").read_text(encoding="utf-8")

    assert "monitor_runtime.py" in validation
    assert "benchmark_inference.py" in validation
    assert "fetch_kaist_assets.py" in validation
    assert "prepare_kaist_person.py" in validation
    assert "runtime_monitor_summary.md" in validation
    assert "build_acceptance_report.py" in validation
    assert "acceptance_report.md" in validation
    assert "tune_alert_thresholds.py" in validation
    assert "alert_threshold_recommendations.md" in validation
    assert "analyze_alert_failures.py" in validation
    assert "alert_failure_analysis.md" in validation
    assert "review_validation_run.py" in validation
    assert "validation_review_index.md" in validation
    assert "smoke_validation_review.py" in validation
