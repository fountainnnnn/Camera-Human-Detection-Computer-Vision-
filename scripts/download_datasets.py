from __future__ import annotations

import argparse
import json
from pathlib import Path

import common  # noqa: F401

from src.dataset_catalog import DATASET_SOURCES, acquisition_plan_markdown, catalog_payload


DATASET_CATALOG = catalog_payload()


def main() -> int:
    parser = argparse.ArgumentParser(description="Write curated online dataset source records and acquisition plan.")
    parser.add_argument("--dataset", choices=sorted(DATASET_SOURCES), nargs="*")
    parser.add_argument("--output", default="reports/dataset_catalog.json")
    parser.add_argument("--plan-output", default="reports/dataset_acquisition_plan.md")
    args = parser.parse_args()

    dataset_ids = args.dataset or sorted(DATASET_SOURCES)
    payload = catalog_payload(dataset_ids)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    plan_path = Path(args.plan_output)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(acquisition_plan_markdown(dataset_ids), encoding="utf-8")

    print(f"catalog_written={output_path}")
    print(f"plan_written={plan_path}")
    for dataset_id in dataset_ids:
        meta = payload[dataset_id]
        print(f"{dataset_id}: source={meta['source_url']} access={meta['access_mode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
