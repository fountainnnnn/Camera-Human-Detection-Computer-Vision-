from pathlib import Path

from src.dataset_tools import collect_examples, split_examples


def test_collect_examples_includes_negatives(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    negatives_dir = tmp_path / "negatives"
    images_dir.mkdir()
    labels_dir.mkdir()
    negatives_dir.mkdir()

    positive_image = images_dir / "pos.jpg"
    positive_label = labels_dir / "pos.txt"
    negative_image = negatives_dir / "neg.jpg"
    positive_image.write_bytes(b"img")
    positive_label.write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")
    negative_image.write_bytes(b"img")

    examples = collect_examples(images_dir, labels_dir, negatives_dir)
    assert len(examples) == 2
    assert sum(example.has_human_gt for example in examples) == 1
    assert sum(not example.has_human_gt for example in examples) == 1


def test_split_examples_preserves_count(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    for index in range(10):
        (images_dir / f"img_{index}.jpg").write_bytes(b"img")
        (labels_dir / f"img_{index}.txt").write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")

    examples = collect_examples(images_dir, labels_dir)
    splits = split_examples(examples, train_ratio=0.6, val_ratio=0.2, seed=42)
    assert sum(len(items) for items in splits.values()) == 10
    assert len(splits["train"]) == 6
    assert len(splits["val"]) == 2
    assert len(splits["test"]) == 2
