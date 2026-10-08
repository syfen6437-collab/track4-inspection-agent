from __future__ import annotations

"""Run the exact paired-holdout review policy in an isolated candidate directory."""

import argparse
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from run_pipeline import load_client, load_config, set_reproducible
from track4_agent.data import load_assets, write_json
from track4_agent.vision_review import (
    DEFAULT_SCENES, cached_reviews, candidate_path, evidence_contradicts,
    merge_reviews, probability, review_context, review_image, select_candidates,
    strict_prediction,
)

# Retain these imports for read-only audits of historical candidate responses.
_candidate_is_contradicted = evidence_contradicts


def _strict_review(raw: str, allowed: set[str], fallback: str):
    lexicon = {"categories": {"audit": {"allowed_labels": sorted(allowed),
               "label_specs": {label: {"count": 1, "default_rating": "",
                               "rating_distribution": {}} for label in allowed}}}}
    return strict_prediction(raw, lexicon, {"questionCategory": "audit", "filename": ""})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CODE / "config" / "qwen3-vl-4b-review.json")
    parser.add_argument("--base", type=Path, default=ROOT / "result" / "result.json")
    parser.add_argument("--vision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--descriptions", type=Path, required=True)
    parser.add_argument("--bridge-scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--track", action="store_true", help="已停用：轨道候选尚无配对验证")
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--review-min-confidence", type=float, default=0.55)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--revalidate-only", action="store_true", help="完整缓存复核，绝不调用模型")
    args = parser.parse_args()
    if args.track:
        parser.error("Track candidates are disabled until the same end-to-end holdout policy is validated")
    scenes = tuple(sorted({value.strip() for value in args.bridge_scenes.split(",") if value.strip()}))
    if not scenes or not set(scenes) <= set(DEFAULT_SCENES):
        parser.error("Only support,bottom scenes are supported by the paired evaluator")
    probability(args.min_confidence)
    probability(args.review_min_confidence)
    output = candidate_path(ROOT, args.output)
    audit_path = candidate_path(ROOT, args.audit)
    descriptions = candidate_path(ROOT, args.descriptions)
    if len({output, audit_path, descriptions}) != 3:
        parser.error("Output, audit and descriptions paths must be distinct")
    if {output, audit_path, descriptions} & {args.base.resolve(), args.vision.resolve()}:
        parser.error("Candidate outputs must not overwrite input artifacts")
    if descriptions.exists() and not (args.resume or args.revalidate_only):
        parser.error("Descriptions already exist; use --resume or a new path")
    if args.revalidate_only and not descriptions.exists():
        parser.error("Revalidation requires a complete existing cache")
    base = json.loads(args.base.read_text(encoding="utf-8"))
    vision = json.loads(args.vision.read_text(encoding="utf-8"))["records"]
    _, manifest, lexicon = load_assets(ROOT)
    config = load_config(args.config.resolve())
    set_reproducible(int(config["seed"]))
    selected = select_candidates(manifest, base, vision, lexicon,
                                 min_confidence=args.min_confidence, scenes=scenes)
    context = review_context(ROOT, config, manifest, base, vision, lexicon,
                             min_confidence=args.min_confidence, scenes=scenes)
    context["review_min_confidence"] = args.review_min_confidence
    saved = []
    if descriptions.exists():
        document = json.loads(descriptions.read_text(encoding="utf-8"))
        if document.get("context") != context:
            raise ValueError("Review context differs; legacy or stale caches cannot be reused")
        saved = document.get("records", [])
    reviews = cached_reviews(selected, saved, require_complete=args.revalidate_only)
    # Validate every cached raw response before loading a model or writing files.
    if reviews:
        cached_ids = set(reviews)
        cached_selected = [item for item, _, _ in selected if item["id"] in cached_ids]
        names = {item["filename"] for item in cached_selected}
        merge_reviews(cached_selected, [row for row in base if row["filename"] in names],
                      [row for row in vision if row["id"] in cached_ids], list(reviews.values()),
                      lexicon, min_confidence=args.min_confidence,
                      review_min_confidence=args.review_min_confidence, scenes=scenes)
    client = None
    raw_dir = ROOT / "data" / "raw"
    for index, (item, original, visual) in enumerate(selected, start=1):
        if item["id"] in reviews:
            continue
        if args.revalidate_only:
            raise RuntimeError("Revalidation must never call a model")
        if client is None:
            client = load_client(config)
        reviews[item["id"]] = review_image(client, item, original, visual, lexicon, config,
                                            raw_dir / item["image"], raw_dir)
        write_json(descriptions, {"context": context,
                                  "records": sorted(reviews.values(), key=lambda row: row["id"])})
        print(f"[review] {index}/{len(selected)} {item['filename']}", flush=True)
    if not descriptions.exists():
        write_json(descriptions, {"context": context, "records": []})
    merged, audit = merge_reviews(manifest, base, vision, list(reviews.values()), lexicon,
                                  min_confidence=args.min_confidence,
                                  review_min_confidence=args.review_min_confidence, scenes=scenes)
    write_json(output, merged)
    write_json(audit_path, {"context": context, "candidate_count": len(selected),
                            "accepted_count": sum(row["accepted"] for row in audit), "records": audit})
    print(json.dumps({"candidate_count": len(selected), "accepted_count": sum(row["accepted"] for row in audit),
                      "output": str(output), "official_result_modified": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
