from __future__ import annotations

"""Apply a validated auxiliary vision-head policy to a Qwen result.

This is deliberately a separate candidate step. It preserves the official
result and emits an audit trail for every automatic label change.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from track4_agent.data import OUTPUT_KEYS, load_assets, write_json


def scene_key(filename: str) -> str:
    if re.search(r"DJI_|^S\d|航拍", filename, re.IGNORECASE):
        return "aerial"
    if re.search(r"桥面|铺装|道路", filename):
        return "deck"
    if re.search(r"支座|墩", filename):
        return "support"
    if re.search(r"梁底|跨中|横隔板", filename):
        return "bottom"
    return "generic"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "result" / "result.json")
    parser.add_argument("--vision", type=Path, default=ROOT / "logs" / "vision_test_predictions.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()

    _, test_manifest, lexicon = load_assets(ROOT)
    results = json.loads(args.input.read_text(encoding="utf-8"))
    vision = json.loads(args.vision.read_text(encoding="utf-8"))["records"]
    by_filename: dict[str, list[dict]] = {}
    for value in vision:
        by_filename.setdefault(value["filename"], []).append(value)
    manifest_by_filename: dict[str, list[dict]] = {}
    for value in test_manifest:
        manifest_by_filename.setdefault(value["filename"], []).append(value)
    duplicate_names = [name for name, values in manifest_by_filename.items() if len(values) > 1]
    if duplicate_names:
        raise ValueError(f"Cannot map basename-only result safely; duplicate test names: {duplicate_names[:5]}")
    candidate = []
    audit = []
    for row in results:
        item = dict(row)
        matches = by_filename.get(row["filename"], [])
        vision_row = matches[0] if len(matches) == 1 else None
        use_vision = (
            row["questionCategory"] == "桥梁"
            and row["defectType"] == "完好"
            and vision_row is not None
            and vision_row["defectType"] != "完好"
            and float(vision_row.get("confidence", 0.0)) >= args.min_confidence
        )
        if use_vision:
            label = vision_row["defectType"]
            spec = lexicon["categories"]["桥梁"]["label_specs"].get(label, {})
            item["defectType"] = label
            item["defectDescription"] = f"图像复核显示{label}特征"
            item["ratingScale(1-5)"] = str(spec.get("default_rating", ""))
            audit.append({
                "filename": row["filename"],
                "scene": scene_key(row["filename"]),
                "from": row["defectType"],
                "to": label,
                "vision_confidence": vision_row.get("confidence", 0.0),
                "original_description": row.get("defectDescription", ""),
                "new_description": item["defectDescription"],
                "new_rating": item["ratingScale(1-5)"],
            })
        candidate.append({key: item[key] for key in OUTPUT_KEYS})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, candidate)
    write_json(args.audit, {
        "policy": "bridge healthy Qwen prediction -> nonhealthy SigLIP2 candidate",
        "min_confidence": args.min_confidence,
        "changed": len(audit),
        "by_scene": dict(Counter(row["scene"] for row in audit)),
        "by_label": dict(Counter(row["to"] for row in audit)),
        "records": audit,
    })
    print(json.dumps({"output": str(args.output), "changed": len(audit),
                      "by_scene": dict(Counter(row["scene"] for row in audit)),
                      "by_label": dict(Counter(row["to"] for row in audit))},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
