from pathlib import Path

from src.openimages_tools import prepare_openimages_person_subset


def test_prepare_openimages_person_subset_writes_labels_manifest_and_image_ids(tmp_path: Path) -> None:
    classes = tmp_path / "classes.csv"
    annotations = tmp_path / "boxes.csv"
    output_root = tmp_path / "out"
    classes.write_text(
        "/m/person,Person\n/m/man,Man\n/m/woman,Woman\n/m/boy,Boy\n/m/girl,Girl\n/m/hand,Human hand\n/m/dog,Dog\n",
        encoding="utf-8",
    )
    annotations.write_text(
        "\n".join(
            [
                "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside",
                "img1,xclick,/m/person,1,0.10,0.50,0.20,0.80,0,0,0,0,0",
                "img1,xclick,/m/man,1,0.20,0.60,0.10,0.50,0,0,0,0,0",
                "img1,xclick,/m/hand,1,0.00,0.10,0.00,0.10,0,0,0,0,0",
                "img1,xclick,/m/dog,1,0.00,0.10,0.00,0.10,0,0,0,0,0",
                "img2,xclick,/m/person,1,0.20,0.40,0.25,0.75,0,0,1,0,0",
                "img3,xclick,/m/person,1,0.00,1.00,0.00,1.00,0,0,0,0,0",
            ]
        ),
        encoding="utf-8",
    )

    counts = prepare_openimages_person_subset(
        annotations_csv=annotations,
        class_descriptions_csv=classes,
        output_root=output_root,
        split="train",
        max_images=2,
    )

    assert counts == {"images": 2, "boxes": 3}
    assert (output_root / "train" / "labels" / "img1.txt").read_text(encoding="utf-8").splitlines() == [
        "0 0.300000 0.500000 0.400000 0.600000",
        "0 0.400000 0.300000 0.400000 0.400000",
    ]
    assert not (output_root / "train" / "labels" / "img2.txt").exists()
    assert (output_root / "train" / "image_ids.txt").read_text(encoding="utf-8").splitlines() == [
        "train/img1",
        "train/img3",
    ]
    manifest = (output_root / "train" / "manifest.csv").read_text(encoding="utf-8")
    assert "openimages_person" in manifest
    assert "img3.jpg" in manifest
