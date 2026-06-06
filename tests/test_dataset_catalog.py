from src.dataset_catalog import DATASET_SOURCES, acquisition_plan_markdown, catalog_payload


def test_dataset_catalog_distinguishes_direct_and_gated_sources() -> None:
    payload = catalog_payload()

    assert payload["openimages"]["access_mode"] == "direct"
    assert payload["kaist"]["access_mode"] == "direct"
    assert payload["llvip"]["access_mode"] == "form_gated"
    assert payload["flir_adas"]["access_mode"] == "form_gated"
    assert payload["exdark"]["access_mode"] == "manual_review"
    assert payload["openimages"]["manual_download"] is False


def test_acquisition_plan_names_metric_and_every_source() -> None:
    plan = acquisition_plan_markdown()

    assert "90%+ person-alert precision" in plan
    for dataset_id, source in DATASET_SOURCES.items():
        assert dataset_id in plan
        assert source.source_url in plan
