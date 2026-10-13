"""Shared conservative policy for track adapter deployment and held-out evaluation."""
from __future__ import annotations

from .data import OUTPUT_KEYS
from .vision_review import index_unique, probability, strict_prediction


def adapter_decision(base, first, second, threshold=0.55):
    """Select a defect only when both adapters agree; protect healthy boundaries."""
    probability(threshold)
    scores = [probability(first["confidence"]), probability(second["confidence"])]
    labels = [first["label"], second["label"]]
    selected = (base["defectType"] != "完好" and labels[0] != "完好"
                and labels[0] == labels[1] and min(scores) >= threshold)
    label = labels[0] if selected else base["defectType"]
    return {
        "filename": base["filename"], "questionCategory": base["questionCategory"],
        "original_label": base["defectType"], "final_label": label,
        "accepted": label != base["defectType"],
        "reason": ("two adapters agree above threshold; baseline healthy protected"
                   if selected else "healthy boundary protected or adapters disagree/below threshold"),
        "adapter_labels": labels, "adapter_confidences": scores,
    }


def checked_review(review, item, candidate, lexicon):
    if review.get("id") != item["id"] or review.get("filename") != item["filename"]:
        raise ValueError("Review image identity differs")
    if review.get("vision_candidate") != candidate["defectType"]:
        raise ValueError("Review adapter label differs")
    probability(review.get("vision_confidence"))
    attempts = review.get("attempts")
    if not isinstance(attempts, list) or not attempts or attempts[-1] != review.get("raw"):
        raise ValueError("Review final attempt differs from raw response")
    if any(not isinstance(raw, str) for raw in attempts):
        raise ValueError("Review attempts must be raw text")
    prediction = strict_prediction(review["raw"], lexicon, item)
    if review.get("prediction") != prediction or review.get("valid") != (prediction is not None):
        raise ValueError("Review raw response and prediction disagree")
    return prediction


def merge_track_reviews(base, candidate, manifest, reviews, lexicon, min_confidence=0.55):
    """Never use edited cached predictions or unsupported/healthy replacements."""
    probability(min_confidence)
    by_name = index_unique(manifest, "filename")
    candidates = index_unique(candidate, "filename")
    bases = index_unique(base, "filename")
    if set(bases) != set(candidates) or set(bases) != set(by_name):
        raise ValueError("Merge filenames differ")
    indexed = index_unique(reviews, "filename")
    final, audit = [], []
    for before in base:
        name = before["filename"]
        after, item, review = candidates[name], by_name[name], indexed.get(name)
        row = dict(before)
        if item["questionCategory"] == "桥梁" and after != before:
            raise ValueError("Bridge rows cannot change")
        selected = (item["questionCategory"] == "轨道" and before["defectType"] != "完好"
                    and after["defectType"] not in (before["defectType"], "完好"))
        if review is not None and not selected:
            raise ValueError("Review contains an unselected image")
        reason = "not selected for review"
        if review is not None:
            prediction = checked_review(review, item, after, lexicon)
            if prediction is None:
                reason = "invalid or contradictory Qwen output"
            elif prediction["confidence"] <= 0 or prediction["confidence"] < min_confidence:
                reason = "Qwen confidence below threshold"
            elif prediction["defectType"] != after["defectType"]:
                reason = "Qwen disagrees with adapter consensus"
            else:
                row.update(defectType=prediction["defectType"],
                           defectDescription=prediction["defectDescription"])
                row["ratingScale(1-5)"] = prediction["ratingScale"]
                reason = "independent Qwen agrees above threshold with both adapters"
        final.append({key: row[key] for key in OUTPUT_KEYS})
        audit.append({"filename": name, "questionCategory": item["questionCategory"],
                      "adapter_label": after["defectType"], "qwen_review": review,
                      "accepted": row != before, "reason": reason})
    if not set(indexed) <= set(by_name):
        raise ValueError("Review contains an unknown image")
    return final, audit
