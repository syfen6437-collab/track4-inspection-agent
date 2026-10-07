from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from track4_agent.data import _build_lexicon, load_assets, write_json
from track4_agent.inference import _macro_atomic_f1, infer_item
from track4_agent.model import ModelClient


def _bridge_group(row: dict[str, Any]) -> str:
    name = str(row.get("bridgeName", "")).strip()
    return re.sub(r"[（(](左幅|右幅)[）)]$", "", name).strip()


def _round_robin_sample(
    rows: list[dict[str, Any]], size: int, rng: random.Random
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["defectType"])].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    labels = list(buckets)
    rng.shuffle(labels)
    selected: list[dict[str, Any]] = []
    while len(selected) < min(size, len(rows)) and labels:
        for label in list(labels):
            if buckets[label]:
                selected.append(buckets[label].pop())
            if not buckets[label]:
                labels.remove(label)
            if len(selected) >= size:
                break
    return selected


def _make_holdout(
    train: list[dict[str, Any]], size: int, seed: int, bridge_group: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    bridge_pool = [
        row for row in train
        if row["questionCategory"] == "桥梁" and _bridge_group(row) == bridge_group
    ]
    track_pool = [row for row in train if row["questionCategory"] == "轨道"]
    bridge_size = min(len(bridge_pool), round(size * 0.7))
    track_size = min(len(track_pool), size - bridge_size)
    sample = _round_robin_sample(bridge_pool, bridge_size, rng)
    sample.extend(_round_robin_sample(track_pool, track_size, rng))
    selected_images = {row["image"] for row in sample}
    heldout_bridge_images = {row["image"] for row in bridge_pool}
    training = [
        row for row in train
        if row["image"] not in selected_images and row["image"] not in heldout_bridge_images
    ]
    if len(sample) < size:
        raise ValueError(
            f"Only selected {len(sample)} of {size} requested samples; "
            f"check holdout bridge {bridge_group!r} and category counts"
        )
    return sample, training, {
        "bridge_group_held_out": bridge_group,
        "bridge_group_image_count": len(bridge_pool),
        "sample_size": len(sample),
        "sample_category_counts": dict(Counter(row["questionCategory"] for row in sample)),
        "sample_label_counts": dict(Counter(
            f"{row['questionCategory']}|{row['defectType']}" for row in sample
        )),
    }


def _holdout_lexicon(
    training: list[dict[str, Any]], full_lexicon: dict[str, Any]
) -> dict[str, Any]:
    lexicon = _build_lexicon(training)
    for category, full_data in full_lexicon["categories"].items():
        category_data = lexicon["categories"].setdefault(category, {
            "sample_count": 0,
            "allowed_labels": [],
            "label_specs": {},
            "scene_label_counts": {},
        })
        category_data["allowed_labels"] = list(full_data["allowed_labels"])
        for label in full_data["allowed_labels"]:
            category_data["label_specs"].setdefault(label, {
                "count": 0,
                "descriptions": [],
                "rating_distribution": {},
                "default_rating": "",
            })
    return lexicon


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    exact = sum(row["truth"]["defectType"] == row["prediction"]["defectType"] for row in records)
    rating = sum(
        row["truth"]["ratingScale(1-5)"] == row["prediction"]["ratingScale(1-5)"]
        for row in records
    )
    pairs = [(row["truth"]["defectType"], row["prediction"]["defectType"]) for row in records]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_category[row["truth"]["questionCategory"]].append(row)
        by_label[row["truth"]["defectType"]].append(row)
    nonhealthy = [row for row in records if row["truth"]["defectType"] != "完好"]
    nonhealthy_recall = sum(row["prediction"]["defectType"] != "完好" for row in nonhealthy)
    return {
        "sample_size": len(records),
        "defect_type_exact_accuracy": round(exact / len(records), 6) if records else 0.0,
        "atomic_macro_f1": round(_macro_atomic_f1(pairs), 6),
        "rating_accuracy": round(rating / len(records), 6) if records else 0.0,
        "nonhealthy_recall": round(nonhealthy_recall / len(nonhealthy), 6) if nonhealthy else None,
        "category_accuracy": {
            category: round(
                sum(r["truth"]["defectType"] == r["prediction"]["defectType"] for r in rows)
                / len(rows),
                6,
            )
            for category, rows in by_category.items()
        },
        "per_label": {
            label: {
                "support": len(rows),
                "exact_accuracy": round(
                    sum(r["truth"]["defectType"] == r["prediction"]["defectType"] for r in rows)
                    / len(rows),
                    6,
                ),
                "predicted_labels": dict(Counter(r["prediction"]["defectType"] for r in rows)),
            }
            for label, rows in by_label.items()
        },
        "predicted_label_counts": dict(Counter(row["prediction"]["defectType"] for row in records)),
    }


def _variants(names: list[str], base: dict[str, Any]) -> dict[str, dict[str, Any]]:
    variants = {
        "coarse_anchor": {
            "presence_context_policy": "coarse_as_hint",
            "calibration_review": False,
        },
        "independent": {
            "presence_context_policy": "none",
            "calibration_review": False,
        },
        "multiview_checklist": {
            "presence_context_policy": "none",
            "bridge_checklist": True,
            "bridge_aerial_crops": True,
            "bridge_aerial_review_crops": True,
            "review_aerial_healthy": True,
            "bridge_review_enabled": True,
            "calibration_review": True,
        },
    }
    return {name: {**base, **variants[name]} for name in names}


def main() -> None:
    parser = argparse.ArgumentParser(description="按整桥留出的4B提示对照评测")
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--bridge-group", default="范家坪1号大桥")
    parser.add_argument("--variants", default="coarse_anchor,independent,multiview_checklist")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "logs" / "holdout_evaluation.json")
    args = parser.parse_args()

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["MODELSCOPE_OFFLINE"] = "1"

    train, _, full_lexicon = load_assets(WORKSPACE)
    config_path = CODE_DIR / "config" / "qwen3-vl-4b-review.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sample, training, split_report = _make_holdout(
        train, args.sample_size, int(config["seed"]), args.bridge_group
    )
    lexicon = _holdout_lexicon(training, full_lexicon)
    names = [name.strip() for name in args.variants.split(",") if name.strip()]
    variants = _variants(names, config)

    client = ModelClient(
        WORKSPACE / config["model_dir"],
        max_new_tokens=int(config["max_new_tokens"]),
        global_max_side=int(config["global_max_side"]),
        crop_size=int(config["crop_size"]),
    )
    results: dict[str, Any] = {}
    for name, variant_config in variants.items():
        records: list[dict[str, Any]] = []
        started = time.time()
        for index, row in enumerate(sample, start=1):
            item = {
                "id": row["image"],
                "image": row["image"],
                "questionCategory": row["questionCategory"],
                "bridgeName": row["bridgeName"],
                "defectLocation": row["defectLocation"],
                "filename": row["filename"],
            }
            prediction, raw = infer_item(
                client,
                item,
                lexicon,
                variant_config,
                WORKSPACE / "data" / "raw" / row["image"],
                raw_dir=WORKSPACE / "data" / "raw",
                allow_review=(
                    row["questionCategory"] == "轨道"
                    or bool(variant_config.get("calibration_review", False))
                ),
            )
            records.append({"truth": row, "prediction": prediction, "raw": raw})
            if index % 10 == 0 or index == len(sample):
                print(f"[{name}] {index}/{len(sample)}", flush=True)
        results[name] = {
            "metrics": _metrics(records),
            "elapsed_seconds": round(time.time() - started, 3),
            "records": records,
        }
        write_json(args.output, {
            "model": config["model_id"],
            "split": split_report,
            "variant_order": list(variants),
            "results": results,
        })

    print(json.dumps({
        "split": split_report,
        "metrics": {name: value["metrics"] for name, value in results.items()},
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
