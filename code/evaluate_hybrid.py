from __future__ import annotations

"""Offline ablations for combining the generative and frozen-vision heads.

This script only reads saved holdout predictions. It never reads labels for
test images and does not modify result.json.
"""

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


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


def load_rows() -> list[dict[str, Any]]:
    qwen = json.loads((ROOT / "logs" / "holdout_evaluation.json").read_text(encoding="utf-8"))
    vision = json.loads((ROOT / "logs" / "vision_heads_holdout.json").read_text(encoding="utf-8"))
    qwen_rows = qwen["results"]["coarse_anchor"]["records"]
    vision_rows = vision["groups"]["范家坪1号大桥"]["results"]["桥梁/multiclass"]["records"]
    vision_by_image = {row["truth"]["image"]: row for row in vision_rows}
    rows = []
    for row in qwen_rows:
        if row["truth"]["questionCategory"] != "桥梁":
            continue
        other = vision_by_image[row["truth"]["image"]]
        scores = other["scores"]
        rows.append({
            "truth": row["truth"]["defectType"],
            "qwen": row["prediction"]["defectType"],
            "qwen_confidence": float(row["prediction"].get("confidence", 0.0)),
            "vision": other["prediction"]["defectType"],
            "vision_confidence": max(map(float, scores.values())),
            "filename": row["truth"]["filename"],
        })
    return rows


def accuracy(rows: list[dict[str, Any]], chooser) -> float:
    return sum(row["truth"] == chooser(row) for row in rows) / len(rows)


def main() -> None:
    rows = load_rows()
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
