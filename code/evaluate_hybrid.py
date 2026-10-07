from __future__ import annotations

"""Offline ablations for combining the generative and frozen-vision heads.

This script only reads saved holdout predictions. It never reads labels for
test images and does not modify result.json.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from track4_agent.model import scene_key as item_scene_key
from track4_agent.vision_review import probability


def scene_key(filename: str) -> str:
    return item_scene_key({"questionCategory": "桥梁", "filename": filename})


def final_confidence(record: dict[str, Any]) -> float:
    final = record["prediction"]
    for raw in reversed(record.get("raw", [])):
        prediction = raw.get("prediction", {})
        if (prediction.get("defectType") == final["defectType"]
                and prediction.get("defectDescription") == final["defectDescription"]
                and prediction.get("ratingScale") == final["ratingScale(1-5)"]):
            return probability(prediction.get("confidence"))
    raise ValueError(f"No matching internal confidence: {record['truth']['image']}")


def load_rows(qwen_path: Path, vision_path: Path, variant: str = "coarse_anchor") -> list[dict[str, Any]]:
    qwen = json.loads(qwen_path.read_text(encoding="utf-8"))
    vision = json.loads(vision_path.read_text(encoding="utf-8"))
    group = qwen["split"]["bridge_group_held_out"]
    if group not in vision["groups"]:
        raise ValueError(f"Vision report has no matching holdout group: {group}")
    qwen_rows = qwen["results"][variant]["records"]
    vision_rows = vision["groups"][group]["results"]["桥梁/multiclass"]["records"]
    vision_by_image = {row["truth"]["image"]: row for row in vision_rows}
    expected = {row["truth"]["image"] for row in qwen_rows if row["truth"]["questionCategory"] == "桥梁"}
    if len(vision_by_image) != len(vision_rows) or set(vision_by_image) != expected:
        raise ValueError("Qwen and vision bridge holdout images differ or contain duplicates")
    rows = []
    for row in qwen_rows:
        if row["truth"]["questionCategory"] != "桥梁":
            continue
        other = vision_by_image[row["truth"]["image"]]
        if row["truth"] != other["truth"]:
            raise ValueError(f"Holdout truth metadata differs: {row['truth']['image']}")
        scores = other["scores"]
        rows.append({
            "truth": row["truth"]["defectType"],
            "qwen": row["prediction"]["defectType"],
            "qwen_confidence": final_confidence(row),
            "vision": other["prediction"]["defectType"],
            "vision_confidence": max(map(float, scores.values())),
            "filename": row["truth"]["filename"],
        })
    return rows


def accuracy(rows: list[dict[str, Any]], chooser) -> float:
    return sum(row["truth"] == chooser(row) for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory direct-label ablation, not the Qwen review policy")
    parser.add_argument("--qwen", type=Path, default=ROOT / "logs" / "holdout_evaluation.json")
    parser.add_argument("--vision", type=Path, default=ROOT / "logs" / "vision_heads_holdout.json")
    parser.add_argument("--variant", default="coarse_anchor")
    args = parser.parse_args()
    rows = load_rows(args.qwen, args.vision, args.variant)
    print("Exploratory direct-label substitution; not a generated submission or review-policy evaluation")
    print(f"n={len(rows)} qwen={accuracy(rows, lambda row: row['qwen']):.4f} "
          f"vision={accuracy(rows, lambda row: row['vision']):.4f}")
    for threshold in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        for policy in ("all", "healthy", "low_qwen", "nonhealthy_qwen"):
            def choose(row: dict[str, Any], threshold=threshold, policy=policy) -> str:
                use_vision = row["vision_confidence"] >= threshold
                if policy == "healthy":
                    use_vision = use_vision and row["qwen"] == "完好"
                elif policy == "low_qwen":
                    use_vision = use_vision and row["qwen_confidence"] < 0.55
                elif policy == "nonhealthy_qwen":
                    use_vision = use_vision and row["qwen"] != "完好"
                return row["vision"] if use_vision else row["qwen"]

            print(f"threshold={threshold:.2f} policy={policy:16s} "
                  f"correct={sum(row['truth'] == choose(row) for row in rows)} "
                  f"accuracy={accuracy(rows, choose):.4f}")

    for scene in ("aerial", "deck", "support", "bottom", "generic"):
        subset = [row for row in rows if scene_key(row["filename"]) == scene]
        if subset:
            healthy_switch = lambda row: row["vision"] if row["qwen"] == "完好" else row["qwen"]
            print(f"scene={scene:8s} n={len(subset)} "
                  f"qwen={accuracy(subset, lambda row: row['qwen']):.4f} "
                  f"vision={accuracy(subset, lambda row: row['vision']):.4f} "
                  f"healthy_switch={accuracy(subset, healthy_switch):.4f}")


if __name__ == "__main__":
    main()
