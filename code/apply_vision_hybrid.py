from __future__ import annotations

"""Merge validated local-Qwen reviews into an isolated candidate result."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from track4_agent.data import load_assets, write_json
from track4_agent.vision_review import candidate_path, merge_reviews, review_context
from run_pipeline import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CODE / "config" / "qwen3-vl-4b-review.json")
    parser.add_argument("--input", type=Path, default=ROOT / "result" / "result.json")
    parser.add_argument("--vision", type=Path, default=ROOT / "logs" / "vision_test_predictions.json")
    parser.add_argument("--reviews", type=Path, required=True,
                        help="Qwen review outputs from describe_vision_candidates.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--review-min-confidence", type=float, default=0.55)
    args = parser.parse_args()

    output_path = candidate_path(ROOT, args.output)
    audit_path = candidate_path(ROOT, args.audit)
    config = load_config(args.config.resolve())
    _, manifest, lexicon = load_assets(ROOT)
    results = json.loads(args.input.read_text(encoding="utf-8"))
    vision_records = json.loads(args.vision.read_text(encoding="utf-8"))["records"]
    review_payload = json.loads(args.reviews.read_text(encoding="utf-8"))
    context = review_context(ROOT, config, manifest, results, vision_records, lexicon,
                             min_confidence=args.min_confidence, scenes=("support", "bottom"))
    if review_payload.get("context") != context:
        raise ValueError("Review context differs from the current inputs/configuration")
    merged, audit = merge_reviews(
        manifest, results, vision_records, review_payload.get("records", []), lexicon,
        min_confidence=args.min_confidence,
        review_min_confidence=args.review_min_confidence,
        scenes=("support", "bottom"),
    )
    write_json(output_path, merged)
    write_json(audit_path, {
        "policy": "Qwen healthy prediction + nonhealthy SigLIP2 trigger -> local Qwen review",
        "min_confidence": args.min_confidence,
        "review_min_confidence": args.review_min_confidence,
        "flagged": len(audit),
        "accepted": sum(row["accepted"] for row in audit),
        "changed": sum(row["accepted"] and row["original"]["defectType"] != row["final"]["defectType"] for row in audit),
        "by_scene": dict(Counter(row["scene"] for row in audit if row["accepted"])),
        "by_label": dict(Counter(row["final"]["defectType"] for row in audit if row["accepted"])),
        "records": audit,
    })
    print(json.dumps({
        "output": str(output_path), "flagged": len(audit),
        "accepted": sum(row["accepted"] for row in audit),
        "changed": sum(row["accepted"] and row["original"]["defectType"] != row["final"]["defectType"] for row in audit),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
