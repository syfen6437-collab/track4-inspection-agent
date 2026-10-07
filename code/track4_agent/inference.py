from __future__ import annotations

import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from .data import OUTPUT_KEYS, read_jsonl, write_json, write_jsonl
from .model import (
    ModelClient,
    build_prompt,
    build_checklist_prompt,
    build_presence_prompt,
    crop_montage,
    needs_review,
    quadrant_crops,
    resize_global,
    sanitize_prediction,
    scene_key,
    checklist_prediction,
)


def _open_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _first_pass_images(path: Path, item: dict[str, Any], config: dict[str, Any]) -> list[Image.Image]:
    image = _open_image(path)
    max_side = int(config["global_max_side"])
    if item["questionCategory"] == "桥梁":
        max_side = int(config.get("bridge_max_side", max_side))
    global_image = resize_global(image, max_side)
    if item["questionCategory"] == "轨道" and config.get("track_use_crops", True):
        return [global_image, crop_montage(image, int(config["crop_size"]))]
    if (
        item["questionCategory"] == "桥梁"
        and scene_key(item) == "aerial"
        and config.get("bridge_aerial_crops", False)
    ):
        return [global_image, *quadrant_crops(image, int(config.get("aerial_crop_size", 448)))]
    if item["questionCategory"] == "桥梁" and config.get("bridge_use_crops", True):
        # Close views (supports, piers, girder bottoms and bridge decks) lose
        # small defects when reduced to a single 384px full-frame image.
        if scene_key(item) != "aerial":
            return [global_image, crop_montage(image, int(config["crop_size"]))]
    return [global_image]


def _review_images(
    path: Path,
    item: dict[str, Any],
    config: dict[str, Any],
    neighbors: dict[str, tuple[str | None, str | None]],
    raw_dir: Path,
) -> list[Image.Image]:
    image = _open_image(path)
    if item["questionCategory"] == "桥梁" and item["filename"].upper().startswith("DJI_"):
        images: list[Image.Image] = []
        previous_id, next_id = neighbors.get(item["id"], (None, None))
        for neighbor_id in (previous_id, item["id"], next_id):
            if not neighbor_id:
                continue
            if neighbor_id == item["id"]:
                neighbor_image = image
            else:
                neighbor_path = raw_dir / "初赛测试集" / Path(neighbor_id)
                neighbor_image = _open_image(neighbor_path)
            images.append(resize_global(neighbor_image, 768))
        if config.get("bridge_aerial_review_crops", False):
            images.extend(quadrant_crops(image, int(config.get("aerial_crop_size", 448))))
        return images
    if item["questionCategory"] == "桥梁" and config.get("bridge_review_crops_only", False):
        return [crop_montage(image, int(config["crop_size"]))]
    return [
        resize_global(image, int(config["global_max_side"])),
        crop_montage(image, int(config["crop_size"])),
    ]


def _neighbor_index(manifest: list[dict[str, Any]]) -> dict[str, tuple[str | None, str | None]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for item in manifest:
        parent = str(Path(item["id"]).parent).replace("\\", "/")
        groups[parent].append(item["id"])
    result: dict[str, tuple[str | None, str | None]] = {}
    for ids in groups.values():
        ids.sort(key=str.lower)
        for index, item_id in enumerate(ids):
            result[item_id] = (
                ids[index - 1] if index > 0 else None,
                ids[index + 1] if index + 1 < len(ids) else None,
            )
    return result


def infer_item(
    client: ModelClient,
    item: dict[str, Any],
    lexicon: dict[str, Any],
    config: dict[str, Any],
    image_path: Path,
    *,
    neighbors: dict[str, tuple[str | None, str | None]] | None = None,
    raw_dir: Path | None = None,
    allow_review: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_records: list[dict[str, Any]] = []
    first_images = _first_pass_images(image_path, item, config)
    use_checklist = bool(config.get("bridge_checklist", False) and item["questionCategory"] == "桥梁")
    first_prompt = build_checklist_prompt(item, lexicon) if use_checklist else build_prompt(item, lexicon)
    if config.get("two_stage_bridge", False) and item["questionCategory"] == "桥梁":
        presence_raw = client.generate(first_images, build_presence_prompt(item))
        presence = _parse_presence(presence_raw)
        raw_records.append({"round": 0, "raw": presence_raw, "presence": presence})
        # Preserve the current submission behavior for older configs; ablations
        # can explicitly set this to "none" and compare on a held-out bridge.
        if config.get("presence_context_policy", "coarse_as_hint") == "coarse_as_hint":
            coarse = str(presence.get("coarse_type", ""))
            if coarse:
                first_prompt += f"\n初筛模型观察到的粗类别（仅作辅助，必须重新核对图像）：{coarse}"
    first_raw = client.generate(first_images, first_prompt)
    if use_checklist:
        first_prediction, first_valid = checklist_prediction(
            first_raw, lexicon, item["questionCategory"], scene_key(item)
        )
    else:
        first_prediction, first_valid = sanitize_prediction(
            first_raw, lexicon, item["questionCategory"], scene_key(item)
        )
    raw_records.append({"round": 1, "raw": first_raw, "prediction": first_prediction, "valid": first_valid})
    chosen = first_prediction

    # Illegal label spellings are normalized against the training-derived
    # lexicon. Reserve the expensive visual review for genuinely uncertain or
    # contradictory predictions so the full test set stays within deadline.
    should_review = allow_review and (
        needs_review(first_prediction, float(config["confidence_threshold"]))
        or (
            config.get("bridge_review_all", False)
            and item["questionCategory"] == "桥梁"
            and scene_key(item) != "aerial"
        )
        or (
            config.get("review_aerial_healthy", False)
            and item["questionCategory"] == "桥梁"
            and scene_key(item) == "aerial"
            and first_prediction["defectType"] == "完好"
        )
    )
    if should_review and raw_dir is not None:
        review_images = _review_images(image_path, item, config, neighbors or {}, raw_dir)
        review_prompt = (
            build_checklist_prompt(item, lexicon, review=True)
            if use_checklist
            else build_prompt(item, lexicon, review=True, prior_prediction=first_prediction)
        )
        review_raw = client.generate(review_images, review_prompt)
        if use_checklist:
            review_prediction, review_valid = checklist_prediction(
                review_raw, lexicon, item["questionCategory"], scene_key(item)
            )
        else:
            review_prediction, review_valid = sanitize_prediction(
                review_raw, lexicon, item["questionCategory"], scene_key(item)
            )
        raw_records.append(
            {"round": 2, "raw": review_raw, "prediction": review_prediction, "valid": review_valid}
        )
        if (
            (first_prediction["defectType"] == "完好" and review_prediction["defectType"] != "完好")
            or review_valid
            or review_prediction["confidence"] >= first_prediction["confidence"]
        ):
            chosen = review_prediction

    final = {
        "questionCategory": item["questionCategory"],
        "bridgeName": item.get("bridgeName", ""),
        "defectLocation": item.get("defectLocation", ""),
        "filename": item["filename"],
        "defectType": chosen["defectType"],
        "defectDescription": chosen["defectDescription"],
        "ratingScale(1-5)": chosen["ratingScale"],
    }
    return final, raw_records


def _parse_presence(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    first, last = cleaned.find("{"), cleaned.rfind("}")
    if 0 <= first < last:
        try:
            value = json.loads(cleaned[first : last + 1])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return {}


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["id"]: row for row in read_jsonl(path)}


def run_inference(
    workspace: Path,
    client: ModelClient,
    config: dict[str, Any],
    manifest: list[dict[str, Any]],
    lexicon: dict[str, Any],
    *,
    resume: bool = False,
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    raw_dir = workspace / "data" / "raw"
    result_dir = workspace / "result"
    logs_dir = workspace / "logs"
    checkpoint_path = result_dir / "checkpoint.jsonl"
    raw_log_path = logs_dir / "raw_responses.jsonl"
    results_path = result_dir / "result.json"
    result_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = _load_checkpoint(checkpoint_path) if resume else {}
    raw_mode = "a" if resume and raw_log_path.exists() else "w"
    neighbors = _neighbor_index(manifest)
    sliced = manifest[offset:]
    target = sliced[:limit] if limit else sliced
    started = time.time()
    processed_this_run = 0
    fast_mode = False

    with raw_log_path.open(raw_mode, encoding="utf-8", newline="\n") as raw_handle:
        for index, item in enumerate(target, start=1):
            if item["id"] in checkpoint:
                continue
            image_path = raw_dir / item["image"]
            allow_review = True
            if fast_mode and item["questionCategory"] == "桥梁":
                allow_review = False
            if item["questionCategory"] == "桥梁" and not config.get("bridge_review_enabled", True):
                allow_review = False
            final, raw_records = infer_item(
                client,
                item,
                lexicon,
                config,
                image_path,
                neighbors=neighbors,
                raw_dir=raw_dir,
                allow_review=allow_review,
            )
            checkpoint[item["id"]] = {"id": item["id"], "result": final}
            raw_handle.write(
                json.dumps({"id": item["id"], "records": raw_records}, ensure_ascii=False) + "\n"
            )
            raw_handle.flush()
            processed_this_run += 1

            if processed_this_run % int(config["checkpoint_every"]) == 0:
                ordered_checkpoint = [checkpoint[key] for key in sorted(checkpoint, key=str.lower)]
                write_jsonl(checkpoint_path, ordered_checkpoint)
                elapsed = time.time() - started
                rate = processed_this_run / elapsed if elapsed else 0.0
                remaining = max(0, len(target) - len(checkpoint))
                eta_hours = remaining / rate / 3600 if rate else float("inf")
                print(
                    f"[infer] {len(checkpoint)}/{len(target)} complete; "
                    f"{rate:.3f} img/s; ETA {eta_hours:.2f} h",
                    flush=True,
                )
                if processed_this_run >= 50 and eta_hours > float(config["deadline_hours"]):
                    fast_mode = True
                    print("[infer] deadline guard enabled: bridge review disabled", flush=True)

    ordered_checkpoint = [checkpoint[item["id"]] for item in target if item["id"] in checkpoint]
    write_jsonl(checkpoint_path, ordered_checkpoint)
    final_results = [entry["result"] for entry in ordered_checkpoint]
    write_json(results_path, final_results)
    elapsed = time.time() - started
    summary = {
        "requested": len(target),
        "offset": offset,
        "completed": len(final_results),
        "processed_this_run": processed_this_run,
        "elapsed_seconds": round(elapsed, 3),
        "images_per_second": round(processed_this_run / elapsed, 5) if elapsed and processed_this_run else 0.0,
        "fast_mode": fast_mode,
        "result_path": str(results_path),
    }
    write_json(logs_dir / "inference_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _atomic_labels(value: str) -> set[str]:
    return {part.strip() for part in re.split(r"[、,+，]", value) if part.strip()}


def _macro_atomic_f1(pairs: list[tuple[str, str]]) -> float:
    labels = sorted({label for pair in pairs for value in pair for label in _atomic_labels(value)})
    if not labels:
        return 0.0
    scores: list[float] = []
    for label in labels:
        tp = fp = fn = 0
        for truth, predicted in pairs:
            truth_set, predicted_set = _atomic_labels(truth), _atomic_labels(predicted)
            tp += int(label in truth_set and label in predicted_set)
            fp += int(label not in truth_set and label in predicted_set)
            fn += int(label in truth_set and label not in predicted_set)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def _calibration_sample(train: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train:
        by_category[row["questionCategory"]].append(row)
    bridge_target = min(len(by_category["桥梁"]), round(size * 0.7))
    track_target = min(len(by_category["轨道"]), size - bridge_target)

    def weighted_pick(rows: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[row["defectType"]].append(row)
        labels = list(buckets)
        rng.shuffle(labels)
        selected: list[dict[str, Any]] = []
        while len(selected) < target and labels:
            progressed = False
            for label in list(labels):
                if buckets[label]:
                    selected.append(rng.choice(buckets[label]))
                    buckets[label].remove(selected[-1])
                    progressed = True
                    if len(selected) >= target:
                        break
                if not buckets[label]:
                    labels.remove(label)
            if not progressed:
                break
        return selected

    return weighted_pick(by_category["桥梁"], bridge_target) + weighted_pick(by_category["轨道"], track_target)


def run_calibration(
    workspace: Path,
    client: ModelClient,
    config: dict[str, Any],
    train: list[dict[str, Any]],
    lexicon: dict[str, Any],
    sample_size: int,
) -> dict[str, Any]:
    raw_dir = workspace / "data" / "raw"
    sample = _calibration_sample(train, sample_size, int(config["seed"]))
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
        predicted, raw_records = infer_item(
            client,
            item,
            lexicon,
            config,
            raw_dir / row["image"],
            raw_dir=raw_dir,
            allow_review=bool(config.get("calibration_review", False)),
        )
        records.append({"truth": row, "prediction": predicted, "raw": raw_records})
        print(f"[calibrate] {index}/{len(sample)}", flush=True)

    exact = sum(r["truth"]["defectType"] == r["prediction"]["defectType"] for r in records)
    rating = sum(
        r["truth"]["ratingScale(1-5)"] == r["prediction"]["ratingScale(1-5)"] for r in records
    )
    pairs = [(r["truth"]["defectType"], r["prediction"]["defectType"]) for r in records]
    metrics = {
        "sample_size": len(records),
        "defect_type_exact_accuracy": round(exact / len(records), 6) if records else 0.0,
        "atomic_macro_f1": round(_macro_atomic_f1(pairs), 6),
        "rating_accuracy": round(rating / len(records), 6) if records else 0.0,
        "elapsed_seconds": round(time.time() - started, 3),
        "category_counts": dict(Counter(r["truth"]["questionCategory"] for r in records)),
    }
    write_json(workspace / "logs" / "calibration_metrics.json", metrics)
    write_json(workspace / "logs" / "calibration_records.json", records)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics
