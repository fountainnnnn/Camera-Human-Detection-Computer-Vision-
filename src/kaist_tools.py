from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


DEFAULT_HUMAN_LABELS = ("person",)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class KaistBox:
    label: str
    x: float
    y: float
    width: float
    height: float

    def yolo_line(self, image_width: int, image_height: int) -> str:
        x_center = (self.x + self.width / 2) / image_width
        y_center = (self.y + self.height / 2) / image_height
        return f"0 {x_center:.6f} {y_center:.6f} {self.width / image_width:.6f} {self.height / image_height:.6f}"


def parse_kaist_label_lines(lines: list[str], human_labels: tuple[str, ...] = DEFAULT_HUMAN_LABELS) -> list[KaistBox]:
    wanted = {label.lower() for label in human_labels}
    boxes: list[KaistBox] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        label = parts[0].lower()
        if label not in wanted:
            continue
        try:
            x, y, width, height = (float(parts[index]) for index in range(1, 5))
        except ValueError:
            continue
        if width <= 0 or height <= 0:
            continue
        boxes.append(KaistBox(label=label, x=x, y=y, width=width, height=height))
    return boxes


def parse_kaist_xml(path: Path, human_labels: tuple[str, ...] = DEFAULT_HUMAN_LABELS) -> list[KaistBox]:
    wanted = {label.lower() for label in human_labels}
    root = ET.parse(path).getroot()
    boxes: list[KaistBox] = []
    for obj in root.findall("object"):
        label = (obj.findtext("name") or "").strip().lower()
        if label not in wanted:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            x = float(box.findtext("x", "0"))
            y = float(box.findtext("y", "0"))
            width = float(box.findtext("w", "0"))
            height = float(box.findtext("h", "0"))
        except ValueError:
            continue
        if width <= 0 or height <= 0:
            continue
        boxes.append(KaistBox(label=label, x=x, y=y, width=width, height=height))
    return boxes


def find_kaist_label_files(annotations_dir: Path, modality: str) -> list[Path]:
    return sorted(
        path
        for path in [*annotations_dir.rglob("*.txt"), *annotations_dir.rglob("*.xml")]
        if not path.name.startswith("._")
        and (path.suffix.lower() == ".xml" or path.parent.name.lower() == modality.lower())
    )


def _image_path_for_label(label_file: Path, annotations_dir: Path, images_dir: Path, modality: str) -> Path | None:
    relative = label_file.relative_to(annotations_dir).with_suffix("")
    if label_file.suffix.lower() == ".xml":
        relative = relative.parent / modality / relative.name
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / relative.with_suffix(extension)
        if candidate.exists():
            return candidate
    return None


def image_path_for_kaist_xml(label_file: Path, annotations_dir: Path, images_dir: Path, modality: str) -> Path | None:
    return _image_path_for_label(label_file, annotations_dir, images_dir, modality)


def frame_index_from_kaist_stem(stem: str) -> int:
    if not stem.startswith("I"):
        raise ValueError(f"KAIST frame stem must start with 'I': {stem}")
    return int(stem[1:])


def prepare_kaist_alert_sequence(
    *,
    annotations_dir: Path,
    images_dir: Path,
    output_root: Path,
    modality: str,
    subset: str,
    source_fps: float,
    sample_fps: float,
    window_seconds: float,
    max_frames: int | None = None,
    human_labels: tuple[str, ...] = DEFAULT_HUMAN_LABELS,
) -> dict[str, int | float]:
    if source_fps <= 0:
        raise ValueError("source_fps must be greater than 0")
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than 0")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be greater than 0")

    output_images = output_root / "images"
    output_images.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.csv"
    sample_stride = max(1, round(source_fps / sample_fps))

    copied_frames = 0
    positive_frames = 0
    negative_frames = 0
    missing_images = 0
    clips: set[str] = set()

    label_files = [
        path
        for path in find_kaist_label_files(annotations_dir, modality=modality)
        if path.suffix.lower() == ".xml"
    ]
    label_files.sort(key=lambda path: (path.parent.as_posix(), frame_index_from_kaist_stem(path.stem)))

    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest:
        manifest.write("image,clip_id,subset,timestamp_seconds,has_human_gt,source_dataset,modality,source_image,source_label\n")
        for label_file in label_files:
            frame_index = frame_index_from_kaist_stem(label_file.stem)
            if frame_index % sample_stride != 0:
                continue
            if max_frames is not None and copied_frames >= max_frames:
                break

            image_path = image_path_for_kaist_xml(
                label_file,
                annotations_dir=annotations_dir,
                images_dir=images_dir,
                modality=modality,
            )
            if image_path is None:
                missing_images += 1
                continue

            boxes = parse_kaist_xml(label_file, human_labels=human_labels)
            has_human_gt = bool(boxes)
            timestamp_seconds = frame_index / source_fps
            window_index = int(timestamp_seconds // window_seconds)
            sequence_id = "_".join(label_file.parent.relative_to(annotations_dir).parts)
            clip_id = f"{sequence_id}_w{window_index:04d}"
            target_name = f"{sequence_id}_{modality}_{label_file.stem}.jpg"
            shutil.copy2(image_path, output_images / target_name)
            clips.add(clip_id)
            copied_frames += 1
            if has_human_gt:
                positive_frames += 1
            else:
                negative_frames += 1
            manifest.write(
                f"{target_name},{clip_id},{subset},{timestamp_seconds:.3f},{str(has_human_gt).lower()},kaist,{modality},"
                f"{image_path},{label_file}\n"
            )

    return {
        "frames": copied_frames,
        "positive_frames": positive_frames,
        "negative_frames": negative_frames,
        "clips": len(clips),
        "missing_images": missing_images,
        "source_fps": source_fps,
        "sample_fps": sample_fps,
        "window_seconds": window_seconds,
    }


def prepare_kaist_person_subset(
    *,
    annotations_dir: Path,
    images_dir: Path,
    output_root: Path,
    split: str,
    modality: str,
    max_images: int | None = None,
    human_labels: tuple[str, ...] = DEFAULT_HUMAN_LABELS,
) -> dict[str, int]:
    output_images = output_root / split / "images"
    output_labels = output_root / split / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / split / "manifest.csv"

    copied_images = 0
    copied_boxes = 0
    missing_images = 0
    empty_labels = 0

    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest:
        manifest.write("image,label,split,subset,has_human_gt,source_dataset,modality,source_image,source_label\n")
        for label_file in find_kaist_label_files(annotations_dir, modality=modality):
            if max_images is not None and copied_images >= max_images:
                break
            if label_file.suffix.lower() == ".xml":
                boxes = parse_kaist_xml(label_file, human_labels=human_labels)
            else:
                boxes = parse_kaist_label_lines(
                    label_file.read_text(encoding="utf-8").splitlines(),
                    human_labels=human_labels,
                )
            if not boxes:
                empty_labels += 1
                continue
            image_path = _image_path_for_label(
                label_file,
                annotations_dir=annotations_dir,
                images_dir=images_dir,
                modality=modality,
            )
            if image_path is None:
                missing_images += 1
                continue

            import cv2

            frame = cv2.imread(str(image_path))
            if frame is None:
                missing_images += 1
                continue
            image_height, image_width = frame.shape[:2]
            target_name = "_".join(label_file.relative_to(annotations_dir).with_suffix("").parts) + image_path.suffix.lower()
            target_image = output_images / target_name
            target_label = output_labels / f"{Path(target_name).stem}.txt"
            shutil.copy2(image_path, target_image)
            target_label.write_text(
                "\n".join(box.yolo_line(image_width, image_height) for box in boxes),
                encoding="utf-8",
            )
            copied_images += 1
            copied_boxes += len(boxes)
            manifest.write(
                f"{target_image.name},{target_label.name},{split},kaist_{modality},true,kaist,{modality},"
                f"{image_path},{label_file}\n"
            )

    return {
        "images": copied_images,
        "boxes": copied_boxes,
        "missing_images": missing_images,
        "empty_labels": empty_labels,
    }
