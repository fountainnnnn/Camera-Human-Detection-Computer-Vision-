# Validation Workflow

Use this when collecting real footage to prove or falsify alert quality across `day`, `low_light`, `ir`, and `night_vision`.

## Bootstrap

```powershell
python .\scripts\init_validation_workspace.py --raw-root .\data\raw --intervals-dir .\data\intervals --output-root .\data\validation_setup
```

This creates:

- `data/raw/day/`
- `data/raw/low_light/`
- `data/raw/ir/`
- `data/raw/night_vision/`
- `data/intervals/`
- `data/validation_setup/validation_collection_checklist.csv`
- `data/validation_setup/validation_collection_guide.md`

## Acquire public training data

Start with researched public sources before relying only on local footage:

```powershell
python .\scripts\download_datasets.py --plan-output .\reports\dataset_acquisition_plan.md
```

Review:

- `reports/dataset_catalog.json`
- `reports/dataset_acquisition_plan.md`

Direct sources currently encoded by the catalog:

- Open Images V7 person subset for general daytime positives and hard negatives
- KAIST multispectral pedestrian benchmark for visible/thermal day-night pedestrian support

Form-gated/manual sources:

- LLVIP for visible/infrared low-light pedestrian data
- Teledyne FLIR ADAS for thermal person data
- ExDark for low-light visible images after license review

For Open Images, download the official class-description and bounding-box CSV files, then generate YOLO labels and the official downloader image-id list:

```powershell
python .\scripts\fetch_openimages_metadata.py --output-dir .\data\sources\openimages
python .\scripts\prepare_openimages_person.py --annotations-csv .\data\sources\openimages\validation-annotations-bbox.csv --class-descriptions-csv .\data\sources\openimages\oidv7-class-descriptions-boxable.csv --output-root .\data\openimages_person --split validation --max-images 1000
python .\scripts\download_openimages_images.py --image-ids .\data\openimages_person\validation\image_ids.txt --output-root .\data\openimages_person --max-images 1000
```

For COCO, download the official 2017 annotations and validation images, then convert only the `person` category into one-class YOLO labels:

```powershell
python .\scripts\fetch_coco_assets.py --output-dir .\data\sources\coco
python .\scripts\prepare_coco_person.py --annotations-json .\data\sources\coco\annotations\instances_val2017.json --images-dir .\data\sources\coco\val2017 --output-root .\data\coco_person --split val2017
```

For KAIST, start with the official preview download command, extract the downloaded archive, then prepare either visible or LWIR single-modality YOLO subsets:

```powershell
python .\scripts\fetch_kaist_assets.py --output-dir .\data\sources\kaist --variant preview --print-only
python .\scripts\prepare_kaist_person.py --annotations-dir .\data\sources\kaist\annotations --images-dir .\data\sources\kaist\images --output-root .\data\kaist_visible_person --split train --modality visible
python .\scripts\prepare_kaist_person.py --annotations-dir .\data\sources\kaist\annotations --images-dir .\data\sources\kaist\images --output-root .\data\kaist_lwir_person --split train --modality lwir
```

Benchmark CPU inference on the current deployable model before claiming mini-PC suitability:

```powershell
python .\scripts\benchmark_inference.py --config .\config.yaml --images .\data\online_coco_person_2693\test\images --output-dir .\reports\inference_benchmark_yolo11n_cpu --model-path .\models\yolo11n.pt --inference-size 512 --confidence-threshold 0.6 --limit 100
```

Review:

- `reports/inference_benchmark_yolo11n_cpu/inference_benchmark.md`
- `reports/inference_benchmark_yolo11n_cpu/inference_benchmark.json`

The target stop condition is 90%+ person-alert precision on held-out validation/test clips, or a clear failure report if the public-source training and tuning path cannot reach it.

## Record clips

- Put each clip under the matching subset folder.
- Keep filenames stable. The system derives `clip_id` from the filename, and the interval CSV must use the same `clip_id`.
- Include ordinary negative footage as well as human-present footage.

Example:

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

## Label intervals

Create one CSV per clip in `data/intervals/` named `<clip_id>.csv`.

Required columns:

- `clip_id`
- `start_seconds`
- `end_seconds`

Optional columns:

- `subset`
- `has_human_gt`
- `notes`

## Run the batch pipeline

```powershell
python .\scripts\run_validation_pipeline.py --config .\config.yaml --raw-root .\data\raw --intervals-dir .\data\intervals --output-root .\data\validation_runs\latest --create-missing-interval-templates
```

Review:

- `clip_index.csv`
- `label_coverage.csv`
- `label_coverage_summary.md`
- `reports/alert_evaluation_summary.md`
- `reports/alert_subset_metrics.csv`

If no labeled clips are available, the pipeline exits nonzero.

## Monitor runtime evidence

While `python .\src\main.py` is running on the target machine, collect runtime evidence in a second shell:

```powershell
python .\scripts\monitor_runtime.py --status-path .\logs\status.json --output-dir .\reports\runtime_monitor --duration-seconds 1800 --max-average-cpu-percent 80 --max-peak-cpu-percent 95 --max-ram-percent 85 --min-average-fps 1.0 --max-disconnects 0 --max-stale-samples 2
```

Review:

- `reports/runtime_monitor/runtime_monitor_samples.csv`
- `reports/runtime_monitor/runtime_monitor_summary.json`
- `reports/runtime_monitor/runtime_monitor_summary.md`

Use this for the remaining CPU-usage and long-runtime stability evidence after the clip-quality pipeline is in place.

## Build acceptance report

After alert evaluation and runtime monitoring are both available, combine them into a single decision report:

```powershell
python .\scripts\build_acceptance_report.py --clip-results .\reports\alert_clip_results.csv --runtime-summary .\reports\runtime_monitor\runtime_monitor_summary.json --output-dir .\reports
```

Review:

- `reports/acceptance_report.json`
- `reports/acceptance_report.md`

This report makes the pending precision, recall, false-positive, CPU, and runtime-stability checks explicit and includes tuning guidance when targets are missed.

## Sweep alert thresholds

After `evaluate_alerts.py` writes frame-level predictions, sweep threshold combinations without rerunning detector inference:

```powershell
python .\scripts\tune_alert_thresholds.py --frame-predictions .\reports\alert_frame_predictions.csv --output-dir .\reports
```

Review:

- `reports/alert_threshold_sweep.csv`
- `reports/alert_threshold_recommendations.md`

Use this when the acceptance report shows the baseline misses precision or recall targets and you need a repeatable threshold-tuning pass before considering model fine-tuning.

## Analyze false positives and missed clips

After `evaluate_alerts.py` writes clip and frame prediction outputs, generate a focused FP/FN review:

```powershell
python .\scripts\analyze_alert_failures.py --clip-results .\reports\alert_clip_results.csv --frame-predictions .\reports\alert_frame_predictions.csv --output-dir .\reports
```

Review:

- `reports/alert_failure_analysis.md`
- `reports/false_positive_frames.csv`
- `reports/missed_positive_clips.csv`

Use this before threshold tuning to see which subsets and clips are driving false alerts or missed detections.

## Run the full review bundle

Once `alert_clip_results.csv`, `alert_frame_predictions.csv`, and `runtime_monitor_summary.json` exist, you can run the review chain in one command:

```powershell
python .\scripts\review_validation_run.py --reports-dir .\reports --runtime-summary .\reports\runtime_monitor\runtime_monitor_summary.json
```

Review:

- `reports/validation_review_index.md`

This wraps failure analysis, threshold tuning, and acceptance reporting over the existing report set.

## Synthetic smoke run

If you want to verify the review tooling without real footage first:

```powershell
python .\scripts\smoke_validation_review.py --output-dir .\reports\demo_validation_review
```

This creates a synthetic evidence set and runs the post-validation review chain end to end.
