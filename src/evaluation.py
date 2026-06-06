from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0


@dataclass
class DetectionMatch:
    tp: int
    fp: int
    fn: int
    matched_gt_indices: set[int]
    false_positive_indices: list[int]
    false_negative_indices: list[int]
    matched_confidences: list[float]
    false_positive_confidences: list[float]


def read_yolo_labels(
    lines: list[str],
    width: int,
    height: int,
    person_class_id: int = 0,
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id, cx, cy, bw, bh = map(float, parts)
        if int(class_id) != person_class_id:
            continue
        box_width = bw * width
        box_height = bh * height
        center_x = cx * width
        center_y = cy * height
        x1 = int(center_x - box_width / 2)
        y1 = int(center_y - box_height / 2)
        x2 = int(center_x + box_width / 2)
        y2 = int(center_y + box_height / 2)
        boxes.append((x1, y1, x2, y2))
    return boxes


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter_area
    return inter_area / denom if denom else 0.0


def match_detections(
    gt_boxes: list[tuple[int, int, int, int]],
    detections: list[object],
    iou_threshold: float = 0.5,
) -> DetectionMatch:
    matched_gt_indices: set[int] = set()
    false_positive_indices: list[int] = []
    matched_confidences: list[float] = []
    false_positive_confidences: list[float] = []

    for detection_index, detection in enumerate(detections):
        best_gt_index: int | None = None
        best_iou = 0.0
        for gt_index, gt_box in enumerate(gt_boxes):
            if gt_index in matched_gt_indices:
                continue
            score = iou(detection.box, gt_box)
            if score > best_iou:
                best_iou = score
                best_gt_index = gt_index
        if best_gt_index is not None and best_iou >= iou_threshold:
            matched_gt_indices.add(best_gt_index)
            matched_confidences.append(float(detection.confidence))
        else:
            false_positive_indices.append(detection_index)
            false_positive_confidences.append(float(detection.confidence))

    false_negative_indices = [
        gt_index for gt_index in range(len(gt_boxes)) if gt_index not in matched_gt_indices
    ]
    return DetectionMatch(
        tp=len(matched_gt_indices),
        fp=len(false_positive_indices),
        fn=len(false_negative_indices),
        matched_gt_indices=matched_gt_indices,
        false_positive_indices=false_positive_indices,
        false_negative_indices=false_negative_indices,
        matched_confidences=matched_confidences,
        false_positive_confidences=false_positive_confidences,
    )


def precision_recall_f1(counts: Counts) -> tuple[float, float, float]:
    precision = counts.tp / max(counts.tp + counts.fp, 1)
    recall = counts.tp / max(counts.tp + counts.fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1
