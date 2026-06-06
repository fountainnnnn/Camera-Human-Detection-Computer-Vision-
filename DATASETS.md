# Dataset Notes

This project is designed to use legally obtainable datasets for low-light, CCTV, infrared, and general person detection.

## Recommended sources

### Open Images V7 person subset

- Source: <https://storage.googleapis.com/openimages/web/download_v7.html>
- Description: <https://storage.googleapis.com/openimages/web/factsfigures_v7.html>
- Use for: large-scale daytime/general positives, hard negatives, and baseline person detector stability
- Access: direct, using official Open Images annotation files and image download tooling
- License note: the official Open Images pages describe downloadable annotation files and warn that image licenses must be verified per image; keep attribution metadata for downloaded images
- Preprocessing:
  - download class descriptions and bounding-box annotation CSVs from the official V7 download page
  - use `scripts/fetch_openimages_metadata.py` to fetch the default validation/test metadata
  - run `scripts/prepare_openimages_person.py` to produce YOLO labels and an image-id list
  - run `scripts/download_openimages_images.py` to download only the selected image IDs from the Open Images public dataset objects
  - keep `Person` by default; add `Man`, `Woman`, `Boy`, or `Girl` only if the experiment defines them as person-class positives

### KAIST multispectral pedestrian benchmark

- Source: <https://soonminhwang.github.io/rgbt-ped-detection/>
- Use for: visible/thermal day-night pedestrian training and validation support
- Access: direct `gdown` downloads published by the benchmark maintainers
- License note: the official benchmark page links CC BY-NC-SA 4.0 license details
- Preprocessing:
  - download the preview set first and verify checksums
  - use `scripts/fetch_kaist_assets.py --variant preview` before attempting the full 36 GB dataset
  - convert person-like labels to YOLO class `0` with `scripts/prepare_kaist_person.py`
  - keep visible and thermal frames as separate subsets unless a multispectral model is introduced

### ExDark

- Source: <https://github.com/cs-chan/Exclusively-Dark-Image-Dataset>
- Use for: low-light general object detection, then retain only `person` labels
- License note: the official repository states the project is under BSD-3-Clause, and also says commercial use should be cleared with the authors
- Preprocessing:
  - convert annotations to YOLO with `scripts/convert_annotations.py`
  - keep `person` labels only
  - add local negative samples from your own camera environment

### LLVIP

- Source: <https://bupt-ai-cz.github.io/LLVIP/>
- Terms: <https://github.com/bupt-ai-cz/LLVIP/blob/main/Term%20of%20Use%20and%20License.md>
- Use for: aligned visible/infrared pedestrian detection in low light
- Access: form-gated; submit the official LLVIP form and keep the terms record outside git
- License note: official terms allow academic and non-academic use for non-commercial purposes; derivative commercial use is not allowed by default
- Preprocessing:
  - retain pedestrian labels
  - keep visible and infrared subsets organized separately if you fine-tune only one modality
  - use a held-out local validation split because LLVIP scenes differ from home-security camera geometry

### Teledyne FLIR ADAS thermal dataset

- Source: <https://oem.flir.com/en-gb/solutions/automotive/adas-dataset-form/>
- Use for: thermal person training and thermal false-positive review
- Access: form-gated; submit the official Teledyne FLIR dataset form
- License note: review the current terms during form submission before any redistribution or commercial use
- Preprocessing:
  - retain `Person` labels as YOLO class `0`
  - keep thermal and visible images as separate subsets
  - treat automotive/traffic geometry as domain support, not final proof for a fixed home-security camera

### COCO person class

- Source: <https://cocodataset.org/>
- Use for: general person-detector baseline warm start
- License note: the official COCO site exposes a Terms of Use page. Use COCO mainly as a general baseline and re-check the current terms before redistributing mirrors or derived image bundles
- Preprocessing:
  - retain only category `person`
  - fetch official 2017 annotations and validation images with `scripts/fetch_coco_assets.py`
  - convert person boxes with `scripts/prepare_coco_person.py`
  - use it to stabilize daytime and normal-light person detection
  - do not treat COCO metrics as proof of night-vision performance

## Selection guidance

- Start baseline evaluation with the pretrained YOLO weights and a held-out public validation subset.
- Pull direct public sources first: Open Images for general person examples and KAIST for visible/thermal day-night pedestrians.
- Add form-gated LLVIP and FLIR if their terms fit the intended use and the baseline misses the low-light/IR target.
- Add ExDark for low-light visible images after reviewing its commercial-use note.
- Add your own local negative captures early. For this security use case, those negatives matter more than generic benchmark breadth.
- Stop condition is not generic accuracy. The target is 90%+ person-alert precision on held-out validation/test clips, or a clear failure report if public-source fine-tuning cannot reach that target.

## Local structure

```text
data/
  raw/
  processed/
  train/
    images/
    labels/
  val/
    images/
    labels/
  test/
    images/
    labels/
  negatives/
  local_test/
    webcam_captures/
    rtsp_captures/
```

## Notes

- Do not scrape random copyrighted web videos or images for training data unless you have explicit permission and a documented license path for that material.
- `scripts/download_datasets.py` writes a curated local catalog plus `reports/dataset_acquisition_plan.md`; it does not auto-scrape benchmark images.
- `scripts/prepare_openimages_person.py` turns official Open Images CSVs into YOLO labels and image-id manifests for the official Open Images download flow.
- `scripts/fetch_coco_assets.py` downloads official COCO archives.
- `scripts/prepare_coco_person.py` turns official COCO `person` annotations into one-class YOLO labels.
- `scripts/fetch_kaist_assets.py` downloads official KAIST preview/full assets through the benchmark's published Google Drive IDs.
- `scripts/prepare_kaist_person.py` turns KAIST visible or LWIR `person` annotations into one-class YOLO labels.
- Keep negative samples with no humans.
- Retain a separate manually reviewed validation/test subset for local camera footage.
- Use `scripts/convert_annotations.py` to convert Pascal VOC style annotations into YOLO person-only labels when a dedicated importer is not available.
- Use `scripts/prepare_dataset.py` to split data into train/val/test folders, including optional negative-image directories.
- Use `scripts/extract_video_frames.py` plus `scripts/apply_clip_intervals.py` to build frame manifests from labeled clips for alert-level evaluation.
- Run `python .\scripts\download_datasets.py` to write the local dataset catalog JSON used by this project.
