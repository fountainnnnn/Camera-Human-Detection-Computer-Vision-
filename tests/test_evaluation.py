from src.evaluation import Counts, match_detections, precision_recall_f1, read_yolo_labels


class DummyDetection:
    def __init__(self, box: tuple[int, int, int, int], confidence: float) -> None:
        self.box = box
        self.confidence = confidence


def test_read_yolo_labels_filters_person_class() -> None:
    lines = [
        "0 0.5 0.5 0.2 0.4",
        "1 0.1 0.1 0.1 0.1",
    ]
    boxes = read_yolo_labels(lines, width=100, height=100)
    assert boxes == [(40, 30, 60, 70)]


def test_match_detections_counts_tp_fp_fn() -> None:
    gt_boxes = [(10, 10, 50, 50), (60, 60, 90, 90)]
    detections = [
        DummyDetection((12, 12, 48, 48), 0.9),
        DummyDetection((0, 0, 5, 5), 0.8),
    ]
    result = match_detections(gt_boxes, detections)
    assert result.tp == 1
    assert result.fp == 1
    assert result.fn == 1
    assert result.false_positive_indices == [1]
    assert result.false_negative_indices == [1]


def test_precision_recall_f1_handles_zero_division() -> None:
    precision, recall, f1 = precision_recall_f1(Counts())
    assert precision == 0.0
    assert recall == 0.0
    assert f1 == 0.0
