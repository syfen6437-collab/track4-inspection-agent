"""Shared, auditable selection and VLM review for offline candidates."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .data import OUTPUT_KEYS
from .inference import _review_images
from .model import _extract_json, build_prompt, sanitize_prediction, scene_key


REVIEW_VERSION = 2
DEFAULT_SCENES = ("support", "bottom")
INTERNAL_KEYS = ("defectType", "defectDescription", "ratingScale", "confidence", "evidence")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Confidence must be a number")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Confidence must be finite and within [0, 1]")
    return float(value)


def index_unique(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    indexed = {}
    for row in rows:
        identity = row[key]
        if not isinstance(identity, str) or not identity or identity in indexed:
            raise ValueError(f"Missing or duplicate {key}: {identity!r}")
        indexed[identity] = row
    return indexed


def select_candidates(
    manifest: list[dict[str, Any]], results: list[dict[str, Any]],
    vision_records: list[dict[str, Any]], lexicon: dict[str, Any], *, min_confidence: float = 0.55,
    scenes: tuple[str, ...] = DEFAULT_SCENES,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    probability(min_confidence)
    by_name = index_unique(manifest, "filename")
    base_by_name = index_unique(results, "filename")
    vision_by_id = index_unique(vision_records, "id")
    if set(base_by_name) != set(by_name):
        raise ValueError("Base result filenames differ from manifest")
    if set(vision_by_id) != {row["id"] for row in manifest}:
        raise ValueError("Vision prediction IDs differ from manifest")
    selected = []
    for item in manifest:
        base, vision = base_by_name[item["filename"]], vision_by_id[item["id"]]
        if set(base) != set(OUTPUT_KEYS) or any(not isinstance(base[key], str) for key in OUTPUT_KEYS):
            raise ValueError(f"Base result schema mismatch: {item['id']}")
        for key in ("questionCategory", "bridgeName", "defectLocation", "filename"):
            if base.get(key) != item.get(key, ""):
                raise ValueError(f"Base metadata mismatch for {item['id']}: {key}")
        for key in ("questionCategory", "filename"):
            if vision.get(key) != item.get(key):
                raise ValueError(f"Vision metadata mismatch for {item['id']}: {key}")
        allowed = lexicon["categories"][item["questionCategory"]]["allowed_labels"]
        if base["defectType"] not in allowed or base["ratingScale(1-5)"] not in {"", "1", "2", "3", "4", "5"}:
            raise ValueError(f"Illegal base label or rating: {item['id']}")
        if vision.get("defectType") not in allowed:
            raise ValueError(f"Illegal vision label for {item['id']}: {vision.get('defectType')!r}")
        confidence = probability(vision.get("confidence"))
        if (item["questionCategory"] == "桥梁" and scene_key(item) in scenes
                and base["defectType"] == "完好" and vision["defectType"] != "完好"
                and confidence >= min_confidence):
            selected.append((item, base, vision))
    return sorted(selected, key=lambda value: value[0]["id"].lower())


def strict_prediction(raw: str, lexicon: dict[str, Any], item: dict[str, Any]) -> dict[str, Any] | None:
    parsed = _extract_json(raw)
    if parsed is None or any(key not in parsed for key in INTERNAL_KEYS):
        return None
    if parsed["defectType"] not in lexicon["categories"][item["questionCategory"]]["allowed_labels"]:
        return None
    if not all(isinstance(parsed[key], str) for key in INTERNAL_KEYS if key != "confidence"):
        return None
    if not parsed["defectDescription"].strip() or not parsed["evidence"].strip():
        return None
    if parsed["ratingScale"] not in {"", "1", "2", "3", "4", "5"}:
        return None
    try:
        probability(parsed["confidence"])
    except ValueError:
        return None
    prediction, valid = sanitize_prediction(raw, lexicon, item["questionCategory"], scene_key(item))
    return {key: prediction[key] for key in INTERNAL_KEYS} if valid else None


def review_image(client, item, base, vision, lexicon, config, image_path, raw_dir,
                 neighbors=None, *, retries: int = 2) -> dict[str, Any]:
    # Keep the global view; cropped texture alone is not enough to identify
    # whether a stain belongs to a bearing, girder or healthy deck.
    review_config = {**config, "bridge_review_crops_only": False}
    images = _review_images(image_path, item, review_config, neighbors or {}, raw_dir)
    prompt = build_prompt(item, lexicon, review=True)
    prompt += (f"\n独立视觉分类器的待核对候选是“{vision['defectType']}”。"
               "该候选可能有误，请独立判断，允许输出完好或其他合法标签。")
    attempts = []
    prediction = None
    for attempt in range(retries + 1):
        raw = client.generate(images, prompt)
        prediction = strict_prediction(raw, lexicon, item)
        attempts.append(raw)
        if prediction is not None:
            break
        prompt += "\n上一轮输出不完整或字段非法。请重新观察图片并完整输出所有五个 JSON 字段。"
    return {
        "id": item["id"], "filename": item["filename"],
        "vision_candidate": vision["defectType"], "vision_confidence": vision["confidence"],
        "valid": prediction is not None, "prediction": prediction,
        "raw": attempts[-1], "attempts": attempts,
    }


def review_context(workspace, config, manifest, results, vision_records, lexicon,
                   *, min_confidence, scenes) -> dict[str, Any]:
    model_dir = Path(config["model_dir"])
    if not model_dir.is_absolute():
        model_dir = workspace / model_dir
    sources = [Path(__file__), Path(__file__).with_name("model.py"),
               Path(__file__).with_name("inference.py")]
    weights = sorted(model_dir.glob("*.safetensors"))
    if not weights:
        raise ValueError(f"No model weights in {model_dir}")
    return {
        "version": REVIEW_VERSION, "config": config,
        "manifest": digest_json(manifest), "base_result": digest_json(results),
        "vision_records": digest_json(vision_records), "lexicon": digest_json(lexicon),
        "min_confidence": min_confidence, "scenes": list(scenes),
        "sources": {path.name: sha256_file(path) for path in sources},
        "weights": {path.name: sha256_file(path) for path in weights},
        "images": {row["id"]: sha256_file(workspace / "data" / "raw" / row["image"])
                   for row in manifest},
    }


def merge_reviews(manifest, results, vision_records, records, lexicon,
                  *, min_confidence=0.55, review_min_confidence=0.55,
                  scenes=DEFAULT_SCENES):
    probability(review_min_confidence)
    candidates = select_candidates(manifest, results, vision_records, lexicon,
                                   min_confidence=min_confidence, scenes=scenes)
    review_by_id = index_unique(records, "id")
    expected = {item["id"] for item, _, _ in candidates}
    if expected != set(review_by_id):
        raise ValueError("Review set is incomplete or contains stale/unselected images")
    audit, replacements = [], {}
    for item, base, vision in candidates:
        review = review_by_id[item["id"]]
        if review["filename"] != item["filename"] or review["vision_candidate"] != vision["defectType"]:
            raise ValueError(f"Review metadata mismatch: {item['id']}")
        if probability(review["vision_confidence"]) != probability(vision["confidence"]):
            raise ValueError(f"Review vision confidence differs: {item['id']}")
        if "attempts" in review and (not review["attempts"] or review["attempts"][-1] != review["raw"]):
            raise ValueError(f"Review final attempt differs from raw text: {item['id']}")
        prediction = strict_prediction(review["raw"], lexicon, item)
        if review.get("valid") != (prediction is not None) or review.get("prediction") != prediction:
            raise ValueError(f"Review raw text and stored prediction disagree: {item['id']}")
        accepted = bool(prediction and prediction["defectType"] != "完好"
                        and prediction["confidence"] >= review_min_confidence)
        final = dict(base)
        if accepted:
            final.update(defectType=prediction["defectType"],
                         defectDescription=prediction["defectDescription"])
            final["ratingScale(1-5)"] = prediction["ratingScale"]
        replacements[item["filename"]] = {key: final[key] for key in OUTPUT_KEYS}
        audit.append({"id": item["id"], "scene": scene_key(item), "accepted": accepted,
                      "original": base, "review": review, "final": final})
    merged = [replacements.get(row["filename"], dict(row)) for row in results]
    return merged, audit


def candidate_path(workspace: Path, path: Path) -> Path:
    resolved = path.resolve()
    if (workspace / "runs").resolve() not in resolved.parents:
        raise ValueError("Candidate output must be inside workspace/runs; official artifacts are protected")
    return resolved
