from __future__ import annotations

"""Utility to export sorted class names from a dataset folder or FINAL_REPORT.json."""

import argparse
import json
from pathlib import Path


def export_from_dataset(dataset_dir: Path) -> list[str]:
    class_names = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
    if not class_names:
        raise ValueError(f"No sub-directories (classes) found under {dataset_dir}")
    return sorted(class_names)


def export_from_report(report_path: Path) -> list[str]:
    with report_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "class_names" not in data:
        raise KeyError("'class_names' key not found in FINAL_REPORT.json")
    class_names = data["class_names"]
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("'class_names' must be a non-empty list")
    return class_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Export class_names.json for inference")
    parser.add_argument("--dataset", type=Path, help="Path to the dataset root (folders per label)", required=False)
    parser.add_argument("--report", type=Path, help="Path to FINAL_REPORT.json", required=False)
    parser.add_argument("--output", type=Path, default=Path("class_names.json"), help="Where to save the JSON output")
    args = parser.parse_args()

    if args.report and args.report.exists():
        class_names = export_from_report(args.report)
    elif args.dataset and args.dataset.exists():
        class_names = export_from_dataset(args.dataset)
    else:
        raise ValueError("Provide either --report or --dataset with valid paths")

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(class_names)} class names to {args.output}")


if __name__ == "__main__":
    main()
