from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


AccessMode = Literal["direct", "form_gated", "manual_review"]


@dataclass(frozen=True)
class DatasetSource:
    dataset_id: str
    title: str
    source_url: str
    license_note: str
    access_mode: AccessMode
    conditions: tuple[str, ...]
    use_for: tuple[str, ...]
    acquisition: tuple[str, ...]
    preprocessing: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["manual_download"] = self.access_mode != "direct"
        return payload


DATASET_SOURCES: dict[str, DatasetSource] = {
    "openimages": DatasetSource(
        dataset_id="openimages",
        title="Open Images V7 person subset",
        source_url="https://storage.googleapis.com/openimages/web/download_v7.html",
        license_note=(
            "Official pages describe annotation files as downloadable CSVs and note that image licenses "
            "must be verified per image. Keep attribution metadata with downloaded images."
        ),
        access_mode="direct",
        conditions=("day", "general", "negatives"),
        use_for=("daytime baseline positives", "hard-negative mining", "general person detector stability"),
        acquisition=(
            "Download V7 class descriptions and bounding box annotation CSVs from the official Open Images download page.",
            "Run scripts/prepare_openimages_person.py to build YOLO labels and an image-id manifest.",
            "Use the official Open Images downloader/CVDF flow to fetch only the emitted image IDs.",
        ),
        preprocessing=(
            "Keep Person by default and optionally add Man, Woman, Boy, and Girl only if the labeling policy treats them as person.",
            "Drop group-of boxes by default because security alerts need individual human boxes.",
            "Use verified negative labels for hard-negative evaluation where available.",
        ),
    ),
    "kaist": DatasetSource(
        dataset_id="kaist",
        title="KAIST Multispectral Pedestrian Detection Benchmark",
        source_url="https://soonminhwang.github.io/rgbt-ped-detection/",
        license_note="Official benchmark page links CC BY-NC-SA 4.0 license details.",
        access_mode="direct",
        conditions=("day", "night", "thermal", "traffic"),
        use_for=("thermal/night pedestrian validation", "cross-condition visible/thermal training"),
        acquisition=(
            "Install gdown.",
            "Run scripts/fetch_kaist_assets.py --variant preview first, then --variant full if disk space allows.",
            "Verify the MD5 checksum files published by the benchmark maintainers.",
        ),
        preprocessing=(
            "Run scripts/prepare_kaist_person.py to convert person-like labels to YOLO class 0.",
            "Keep visible and thermal frames as separate subsets unless a multispectral model is introduced.",
            "Use this as domain support, not as final proof for a fixed home-security camera.",
        ),
    ),
    "exdark": DatasetSource(
        dataset_id="exdark",
        title="Exclusively Dark (ExDark)",
        source_url="https://github.com/cs-chan/Exclusively-Dark-Image-Dataset",
        license_note=(
            "Repository lists BSD-3-Clause and asks users to contact the authors for commercial-purpose usage."
        ),
        access_mode="manual_review",
        conditions=("low_light", "night", "twilight"),
        use_for=("low-light person fine-tuning", "low-light false-positive analysis"),
        acquisition=(
            "Download from the official repository or linked official distribution route.",
            "Review the current license and commercial-use note before use outside personal/research experiments.",
        ),
        preprocessing=(
            "Convert annotations to YOLO with scripts/convert_annotations.py.",
            "Retain only person labels.",
            "Keep condition metadata where available for subset metrics.",
        ),
    ),
    "llvip": DatasetSource(
        dataset_id="llvip",
        title="LLVIP visible-infrared paired low-light pedestrian dataset",
        source_url="https://bupt-ai-cz.github.io/LLVIP/",
        license_note=(
            "Official terms allow academic and non-academic use for non-commercial purposes; commercial use is not granted by default."
        ),
        access_mode="form_gated",
        conditions=("low_light", "infrared", "thermal"),
        use_for=("low-light pedestrian fine-tuning", "infrared validation", "visible/IR comparison"),
        acquisition=(
            "Submit the official LLVIP access form and agree to the terms.",
            "Keep the received archive and terms record outside git.",
        ),
        preprocessing=(
            "Retain pedestrian bounding boxes as YOLO class 0.",
            "Keep visible and infrared subsets separate in manifests.",
            "Hold out clips/images for validation before tuning thresholds.",
        ),
    ),
    "flir_adas": DatasetSource(
        dataset_id="flir_adas",
        title="Teledyne FLIR ADAS thermal dataset",
        source_url="https://oem.flir.com/en-gb/solutions/automotive/adas-dataset-form/",
        license_note="Official form describes a free thermal/visible ADAS dataset; review current terms during form submission.",
        access_mode="form_gated",
        conditions=("thermal", "day", "night", "traffic"),
        use_for=("thermal person training", "thermal false-positive review"),
        acquisition=(
            "Submit the official Teledyne FLIR dataset form.",
            "Retain the suggested train/validation split when importing.",
        ),
        preprocessing=(
            "Convert Person labels from the provided annotation format to YOLO class 0.",
            "Use thermal stills/videos as thermal support, not final home-camera proof.",
        ),
    ),
    "coco": DatasetSource(
        dataset_id="coco",
        title="COCO person class",
        source_url="https://cocodataset.org/",
        license_note="Use the official COCO terms and image licenses before redistribution.",
        access_mode="manual_review",
        conditions=("day", "general"),
        use_for=("general person warm start", "sanity baseline"),
        acquisition=(
            "Download images and detection annotations from the official COCO site.",
            "Use scripts/convert_annotations.py to keep category person only.",
        ),
        preprocessing=(
            "Retain only category person.",
            "Do not use COCO metrics as night/IR proof.",
        ),
    ),
}


def selected_sources(dataset_ids: list[str] | None = None) -> dict[str, DatasetSource]:
    ids = dataset_ids or sorted(DATASET_SOURCES)
    return {dataset_id: DATASET_SOURCES[dataset_id] for dataset_id in ids}


def catalog_payload(dataset_ids: list[str] | None = None) -> dict[str, dict[str, object]]:
    return {dataset_id: source.to_dict() for dataset_id, source in selected_sources(dataset_ids).items()}


def acquisition_plan_markdown(dataset_ids: list[str] | None = None) -> str:
    lines = [
        "# Dataset Acquisition Plan",
        "",
        "Target metric: 90%+ person-alert precision on held-out validation/test clips or a clear failure report.",
        "",
    ]
    for dataset_id, source in selected_sources(dataset_ids).items():
        lines.extend(
            [
                f"## {source.title}",
                "",
                f"- ID: `{dataset_id}`",
                f"- Source: {source.source_url}",
                f"- Access: `{source.access_mode}`",
                f"- Conditions: {', '.join(source.conditions)}",
                f"- License note: {source.license_note}",
                "- Acquisition:",
            ]
        )
        lines.extend(f"  - {item}" for item in source.acquisition)
        lines.append("- Preprocessing:")
        lines.extend(f"  - {item}" for item in source.preprocessing)
        lines.append("")
    return "\n".join(lines)
