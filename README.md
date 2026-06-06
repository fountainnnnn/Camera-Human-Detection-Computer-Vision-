# Security AI

Windows-first local human detector for webcam testing and RTSP night-vision deployment.

## What this project does

- Uses a single configurable detection pipeline for `webcam`, `rtsp`, `video_file`, and `image_folder`
- Runs person-only detection locally with YOLO
- Applies temporal filtering so single-frame hits do not trigger alerts
- Saves annotated alert snapshots and can send them to Telegram
- Writes rotating logs and a JSON health/status file
- Includes Windows deployment scripts for Task Scheduler and NSSM

## Project layout

```text
security-ai/
  README.md
  DATASETS.md
  requirements.txt
  config.example.yaml
  .env.example
  src/
  scripts/
  models/
  data/
  alerts/
  logs/
  reports/
  deploy/
  tests/
```

## Windows setup

```powershell
git clone https://github.com/fountainnnnn/Camera-Human-Detection-Computer-Vision-.git
cd Camera-Human-Detection-Computer-Vision-
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .\config.example.yaml .\config.yaml
Copy-Item .\.env.example .\.env
python .\scripts\test_webcam.py
python .\src\main.py
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Configuration

`config.yaml` controls the runtime. The shipped defaults start in webcam mode so the pipeline can be validated before RTSP setup. The example config points at the packaged trained model:

```yaml
model:
  path: "models/human_kaist_lwir_yolo11n.pt"
  confidence_threshold: 0.60
  inference_size: 640

detection:
  sample_fps: 2
  rolling_window_size: 5
  min_positive_frames: 2
  min_detection_duration_seconds: 0.5
```

For an i5-8500 / 16 GB RAM machine, start with `model.inference_size: 512` if CPU use is high. Use the low-resolution RTSP substream for detection and the 2K stream only for alert snapshots.

Switch to RTSP production mode by editing:

```yaml
input:
  mode: "rtsp"
```

Then set:

- `input.rtsp_detection_url`
- `input.rtsp_snapshot_url`
- `.env` values for Telegram if alerts are enabled

Keep RTSP reachable only on the local LAN or a private VPN. Do not port-forward or expose the RTSP endpoint publicly.

## Common commands

Webcam pipeline:

```powershell
python .\scripts\test_webcam.py
python .\scripts\test_webcam.py --frames 10 --save-snapshots --output-dir .\reports\webcam_checks
python .\src\main.py
```

RTSP validation:

```powershell
python .\scripts\test_rtsp.py
python .\src\main.py
```

Production RTSP example:

```yaml
input:
  mode: "rtsp"
  rtsp_detection_url: "rtsp://username:password@CAMERA_IP:554/stream2"
  rtsp_snapshot_url: "rtsp://username:password@CAMERA_IP:554/stream1"

runtime:
  debug_view: false
```

Then run:

```powershell
python .\scripts\test_rtsp.py --config .\config.yaml
python .\src\main.py .\config.yaml
```

Image and video spot checks:

```powershell
python .\scripts\test_image.py --image path\to\frame.jpg
python .\scripts\test_video.py --video path\to\clip.mp4
```

Telegram test:

```powershell
python .\scripts\test_telegram.py
```

Evaluation:

```powershell
python .\scripts\init_validation_workspace.py --raw-root .\data\raw --intervals-dir .\data\intervals --output-root .\data\validation_setup
python .\scripts\download_datasets.py --plan-output .\reports\dataset_acquisition_plan.md
python .\scripts\fetch_openimages_metadata.py --output-dir .\data\sources\openimages
python .\scripts\prepare_openimages_person.py --annotations-csv .\data\sources\openimages\validation-annotations-bbox.csv --class-descriptions-csv .\data\sources\openimages\oidv7-class-descriptions-boxable.csv --output-root .\data\openimages_person --split validation --max-images 1000
python .\scripts\download_openimages_images.py --image-ids .\data\openimages_person\validation\image_ids.txt --output-root .\data\openimages_person --max-images 1000
python .\scripts\fetch_coco_assets.py --output-dir .\data\sources\coco
python .\scripts\prepare_coco_person.py --annotations-json .\data\sources\coco\annotations\instances_val2017.json --images-dir .\data\sources\coco\val2017 --output-root .\data\coco_person --split val2017
python .\scripts\fetch_kaist_assets.py --output-dir .\data\sources\kaist --variant preview --print-only
python .\scripts\prepare_kaist_person.py --annotations-dir .\data\sources\kaist\annotations --images-dir .\data\sources\kaist\images --output-root .\data\kaist_visible_person --split train --modality visible
python .\scripts\prepare_kaist_person.py --annotations-dir .\data\sources\kaist\annotations --images-dir .\data\sources\kaist\images --output-root .\data\kaist_lwir_person --split train --modality lwir
python .\scripts\benchmark_inference.py --config .\config.yaml --images .\data\online_coco_person_2693\test\images --output-dir .\reports\inference_benchmark_yolo11n_cpu --model-path .\models\yolo11n.pt --inference-size 512 --confidence-threshold 0.6 --limit 100
python .\scripts\evaluate.py --images .\data\test\images --labels .\data\test\labels
python .\scripts\evaluate.py --images .\data\test\images --labels .\data\test\labels --manifest .\data\test\manifest.csv
python .\scripts\extract_video_frames.py --video .\data\raw\night_hallway.mp4 --output-images .\data\test\images --output-manifest .\data\test\manifest.csv --clip-id night-hallway-001 --subset night
python .\scripts\build_frame_manifest.py --images .\data\test\images --output .\data\test\manifest.csv --subset night --clip-id clip-001
python .\scripts\apply_clip_intervals.py --manifest .\data\test\manifest.csv --intervals .\data\test\intervals.csv
python .\scripts\record_clip.py --config .\config.yaml --mode webcam --duration-seconds 20 --prefix day_walkthrough
python .\scripts\record_clip.py --config .\config.yaml --mode rtsp --duration-seconds 20 --prefix night_ir_walkthrough
python .\scripts\extract_clip_batch.py --raw-root .\data\raw --output-images .\data\test\images --output-manifests-dir .\data\test\manifests --merged-manifest .\data\test\merged_manifest.csv
python .\scripts\merge_manifests.py --manifest .\data\test\day_manifest.csv --manifest .\data\test\night_manifest.csv --output .\data\test\merged_manifest.csv
python .\scripts\run_validation_pipeline.py --config .\config.yaml --raw-root .\data\raw --intervals-dir .\data\intervals --output-root .\data\validation_runs\latest --create-missing-interval-templates
python .\scripts\evaluate_alerts.py --images .\data\test\images --manifest .\data\test\manifest.csv
python .\scripts\analyze_alert_failures.py --clip-results .\reports\alert_clip_results.csv --frame-predictions .\reports\alert_frame_predictions.csv --output-dir .\reports
python .\scripts\tune_alert_thresholds.py --frame-predictions .\reports\alert_frame_predictions.csv --output-dir .\reports
python .\scripts\monitor_runtime.py --status-path .\logs\status.json --output-dir .\reports\runtime_monitor --duration-seconds 1800 --max-average-cpu-percent 80 --max-peak-cpu-percent 95 --max-ram-percent 85 --min-average-fps 1.0 --max-disconnects 0 --max-stale-samples 2
python .\scripts\build_acceptance_report.py --clip-results .\reports\alert_clip_results.csv --runtime-summary .\reports\runtime_monitor\runtime_monitor_summary.json --output-dir .\reports
python .\scripts\review_validation_run.py --reports-dir .\reports --runtime-summary .\reports\runtime_monitor\runtime_monitor_summary.json
python .\scripts\smoke_validation_review.py --output-dir .\reports\demo_validation_review
```

Evaluation outputs:

- `reports/dataset_catalog.json`
- `reports/dataset_acquisition_plan.md`
- `reports/evaluation_summary.md`
- `reports/threshold_sweep.csv`
- `reports/per_image_results.csv`
- `reports/confidence_distribution.csv`
- `reports/subset_metrics.csv` when a manifest provides subset labels such as `day`, `night`, or `ir`
- `reports/false_positives/`
- `reports/false_negatives/`
- `reports/alert_evaluation_summary.md`
- `reports/alert_clip_results.csv`
- `reports/alert_subset_metrics.csv`
- `reports/alert_frame_predictions.csv`
- `reports/alert_failure_analysis.md`
- `reports/false_positive_frames.csv`
- `reports/missed_positive_clips.csv`
- `reports/alert_threshold_sweep.csv`
- `reports/alert_threshold_recommendations.md`
- `reports/runtime_monitor/runtime_monitor_samples.csv`
- `reports/runtime_monitor/runtime_monitor_summary.json`
- `reports/runtime_monitor/runtime_monitor_summary.md`
- `reports/acceptance_report.json`
- `reports/acceptance_report.md`
- `reports/validation_review_index.md`

Manifest columns used by alert evaluation:

- `image`
- `clip_id`
- `subset` such as `day`, `low_light`, `ir`, or `night_vision`
- `timestamp_seconds`
- `has_human_gt`

Suggested subset values:

- `day`
- `low_light`
- `ir`
- `night_vision`

Recommended raw clip layout for batch extraction:

```text
data/
  raw/
    day/
      hallway_walkthrough.mp4
    low_light/
      room_entry.mp4
    ir/
      hallway_ir.mp4
    night_vision/
      porch_nightvision.mp4
```

Recommended interval layout for `run_validation_pipeline.py`:

```text
data/
  intervals/
    hallway_walkthrough.csv
    room_entry.csv
    hallway_ir.csv
    porch_nightvision.csv
```

Each interval CSV is named after the inferred `clip_id` and uses:

- `clip_id`
- `start_seconds`
- `end_seconds`
- optional `subset`
- optional `has_human_gt`

`run_validation_pipeline.py` also writes:

- `clip_index.csv`
- `label_coverage.csv`
- `label_coverage_summary.md`
- starter interval CSVs for missing labels when `--create-missing-interval-templates` is set

`init_validation_workspace.py` writes:

- subset folders under `data/raw/` for `day`, `low_light`, `ir`, and `night_vision`
- `data/validation_setup/validation_collection_checklist.csv`
- `data/validation_setup/validation_collection_guide.md`

If no labeled clips are available for evaluation and `--skip-eval` is not set, `run_validation_pipeline.py` exits nonzero instead of pretending evaluation succeeded.

For the full operator flow, see [VALIDATION.md](/D:/Camera%20Detection/VALIDATION.md).

Interval CSV columns for `apply_clip_intervals.py`:

- `clip_id`
- `start_seconds`
- `end_seconds`
- optional `subset`
- optional `has_human_gt`

## Runtime notes

- Detection is sampled at low FPS by design.
- The detector does not need a smooth live GUI. `runtime.debug_view` can be disabled for 24/7 operation.
- `runtime.save_debug_frames: true` writes annotated validation frames under `alerts\test_snapshots\` and is intended for short local test runs rather than 24/7 operation.
- RTSP mode uses the detection stream for inference and attempts a higher-quality snapshot stream for alerts when configured.
- `model.device: auto` prefers a supported accelerator when the current `torch` runtime exposes one, otherwise it falls back to CPU.
- Cooldown, rolling-window confirmation, detection duration, box-size filtering, and optional zones are all configurable.
- `scripts/monitor_runtime.py` samples `logs/status.json` from a running instance and produces CPU, RAM, FPS, disconnect, and stale-sample evidence for runtime acceptance checks.
- `scripts/build_acceptance_report.py` combines alert-level validation results and runtime-monitor outputs into a single pass/fail acceptance report plus tuning guidance.
- `scripts/tune_alert_thresholds.py` replays saved frame predictions across threshold combinations so tuning can happen without rerunning YOLO for every config guess.
- `scripts/analyze_alert_failures.py` turns clip results and frame predictions into a concrete false-positive and missed-positive review surface before threshold tuning or model fine-tuning.
- `scripts/review_validation_run.py` runs the post-validation review chain over existing reports and writes a single index for the resulting evidence bundle.
- `scripts/smoke_validation_review.py` generates a synthetic report set and runs the review bundle end to end, which is useful for verifying the tooling on a fresh machine before using real footage.
- `scripts/download_datasets.py` records the researched online dataset sources and writes the acquisition plan for direct and form-gated datasets.
- `scripts/fetch_openimages_metadata.py` fetches official Open Images class-description and validation/test bounding-box CSVs, with the large train-box CSV available behind `--include-train-boxes`.
- `scripts/prepare_openimages_person.py` converts official Open Images bounding-box CSV rows into YOLO person labels plus an image-id manifest for the official Open Images downloader.
- `scripts/download_openimages_images.py` downloads Open Images files from the generated image-id manifest into the matching YOLO image folder.
- `scripts/fetch_coco_assets.py` fetches official COCO annotations and validation images.
- `scripts/prepare_coco_person.py` converts official COCO `person` boxes into single-class YOLO labels.
- `scripts/fetch_kaist_assets.py` prints or runs the official KAIST `gdown` download command for preview/full assets.
- `scripts/prepare_kaist_person.py` converts KAIST visible or LWIR person boxes into single-class YOLO labels.
- `scripts/benchmark_inference.py` measures actual detector latency, FPS, process CPU, and memory on an image folder and writes `inference_benchmark.md` / `inference_benchmark.json`.

## Health and logs

Files written at runtime:

- `logs/app.log`
- `logs/errors.log`
- `logs/detections.log`
- `logs/runtime.log`
- `logs/status.json`

Optional local health endpoint:

- `http://127.0.0.1:8765/health`

## Deployment

The Windows deployment paths use `deploy/run_security_ai.ps1` as the entrypoint so the runtime writes to `logs/runtime.log` and restarts locally after nonzero exits.

Task Scheduler:

```powershell
.\deploy\install_task_scheduler.ps1
```

NSSM:

- Follow [deploy/nssm_setup.md](/D:/Camera%20Detection/deploy/nssm_setup.md)

## Troubleshooting

- If the model file is missing, `ultralytics` may download it on first run.
- If webcam open fails, confirm the correct `webcam_index` and close other apps using the camera.
- If RTSP fails, validate the URL with `scripts/test_rtsp.py` and confirm camera credentials on the local LAN.
- If Telegram send fails, confirm `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`.
- If CPU usage is too high, lower `detection.sample_fps` or `model.inference_size`.
