from pathlib import Path

import numpy as np

from src.kaist_tools import parse_kaist_label_lines, parse_kaist_xml, prepare_kaist_alert_sequence, prepare_kaist_person_subset


def test_parse_kaist_label_lines_keeps_human_boxes() -> None:
    boxes = parse_kaist_label_lines(
        [
            "% bbGt version=3",
            "person 10 20 30 40 0 0 0 0 0",
            "people 1 2 3 4 0 0 0 0 0",
            "cyclist 5 6 7 8 0 0 0 0 0",
            "person bad 20 30 40",
        ],
        human_labels=("person", "people"),
    )

    assert len(boxes) == 2
    assert boxes[0].yolo_line(100, 200) == "0 0.250000 0.200000 0.300000 0.200000"
    assert boxes[1].label == "people"


def test_prepare_kaist_person_subset_writes_single_modality_yolo_labels(tmp_path: Path) -> None:
    import cv2

    annotations_dir = tmp_path / "annotations"
    images_dir = tmp_path / "images"
    label_dir = annotations_dir / "set00" / "V000" / "visible"
    image_dir = images_dir / "set00" / "V000" / "visible"
    lwir_label_dir = annotations_dir / "set00" / "V000" / "lwir"
    lwir_image_dir = images_dir / "set00" / "V000" / "lwir"
    label_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    lwir_label_dir.mkdir(parents=True)
    lwir_image_dir.mkdir(parents=True)

    (label_dir / "I00001.txt").write_text("% bbGt version=3\nperson 10 20 30 40 0 0 0 0 0\n", encoding="utf-8")
    (lwir_label_dir / "I00001.txt").write_text("% bbGt version=3\nperson 1 2 3 4 0 0 0 0 0\n", encoding="utf-8")
    cv2.imwrite(str(image_dir / "I00001.jpg"), np.zeros((200, 100, 3), dtype=np.uint8))
    cv2.imwrite(str(lwir_image_dir / "I00001.jpg"), np.zeros((50, 50, 3), dtype=np.uint8))

    counts = prepare_kaist_person_subset(
        annotations_dir=annotations_dir,
        images_dir=images_dir,
        output_root=tmp_path / "out",
        split="train",
        modality="visible",
    )

    assert counts == {"images": 1, "boxes": 1, "missing_images": 0, "empty_labels": 0}
    label = (tmp_path / "out" / "train" / "labels" / "set00_V000_visible_I00001.txt").read_text(encoding="utf-8")
    assert label == "0 0.250000 0.200000 0.300000 0.200000"
    manifest = (tmp_path / "out" / "train" / "manifest.csv").read_text(encoding="utf-8")
    assert "kaist_visible" in manifest


def test_prepare_kaist_person_subset_supports_preview_xml_layout(tmp_path: Path) -> None:
    import cv2

    annotations_dir = tmp_path / "annotations-xml-new-sanitized"
    images_dir = tmp_path / "images"
    annotation_dir = annotations_dir / "set04" / "V001"
    visible_dir = images_dir / "set04" / "V001" / "visible"
    lwir_dir = images_dir / "set04" / "V001" / "lwir"
    annotation_dir.mkdir(parents=True)
    visible_dir.mkdir(parents=True)
    lwir_dir.mkdir(parents=True)
    xml_path = annotation_dir / "I00071.xml"
    xml_path.write_text(
        """
<annotation>
  <size><width>640</width><height>512</height></size>
  <object>
    <name>person</name>
    <bndbox><x>467</x><y>199</y><w>20</w><h>39</h></bndbox>
  </object>
</annotation>
""".strip(),
        encoding="utf-8",
    )
    cv2.imwrite(str(visible_dir / "I00071.jpg"), np.zeros((512, 640, 3), dtype=np.uint8))
    cv2.imwrite(str(lwir_dir / "I00071.jpg"), np.zeros((512, 640, 3), dtype=np.uint8))

    assert parse_kaist_xml(xml_path)[0].label == "person"
    counts = prepare_kaist_person_subset(
        annotations_dir=annotations_dir,
        images_dir=images_dir,
        output_root=tmp_path / "out_xml",
        split="night",
        modality="lwir",
    )

    assert counts == {"images": 1, "boxes": 1, "missing_images": 0, "empty_labels": 0}
    label = (tmp_path / "out_xml" / "night" / "labels" / "set04_V001_I00071.txt").read_text(encoding="utf-8")
    assert label == "0 0.745313 0.426758 0.031250 0.076172"


def test_prepare_kaist_alert_sequence_includes_negative_frames(tmp_path: Path) -> None:
    import cv2

    annotations_dir = tmp_path / "annotations-xml-new-sanitized"
    images_dir = tmp_path / "images"
    annotation_dir = annotations_dir / "set04" / "V001"
    lwir_dir = images_dir / "set04" / "V001" / "lwir"
    annotation_dir.mkdir(parents=True)
    lwir_dir.mkdir(parents=True)
    for index, has_person in [(0, False), (10, True), (20, False)]:
        object_xml = (
            "<object><name>person</name><bndbox><x>10</x><y>20</y><w>30</w><h>40</h></bndbox></object>"
            if has_person
            else ""
        )
        (annotation_dir / f"I{index:05d}.xml").write_text(f"<annotation>{object_xml}</annotation>", encoding="utf-8")
        cv2.imwrite(str(lwir_dir / f"I{index:05d}.jpg"), np.zeros((512, 640, 3), dtype=np.uint8))

    counts = prepare_kaist_alert_sequence(
        annotations_dir=annotations_dir,
        images_dir=images_dir,
        output_root=tmp_path / "alert",
        modality="lwir",
        subset="kaist_lwir_sequence",
        source_fps=20,
        sample_fps=2,
        window_seconds=1,
    )

    assert counts["frames"] == 3
    assert counts["positive_frames"] == 1
    assert counts["negative_frames"] == 2
    manifest = (tmp_path / "alert" / "manifest.csv").read_text(encoding="utf-8")
    assert "set04_V001_w0000" in manifest
    assert ",true,kaist,lwir," in manifest
    assert ",false,kaist,lwir," in manifest
