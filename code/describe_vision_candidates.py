from __future__ import annotations

"""Run resumable local-Qwen reviews for frozen-vision bridge candidates."""

import argparse
import json
import os
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from run_pipeline import load_client, load_config
from track4_agent.data import load_assets, write_json
from track4_agent.inference import _neighbor_index
from track4_agent.vision_review import (
    candidate_path,
    review_context,
    review_image,
    select_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CODE / "config" / "qwen3-vl-4b-review.json")
    parser.add_argument("--input", type=Path, default=ROOT / "result" / "result.json")
    parser.add_argument("--vision", type=Path, default=ROOT / "logs" / "vision_test_predictions.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    output_path = candidate_path(ROOT, args.output)

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["MODELSCOPE_OFFLINE"] = "1"
    config = load_config(args.config.resolve())
    _, manifest, lexicon = load_assets(ROOT)
    results = json.loads(args.input.read_text(encoding="utf-8"))
    vision_records = json.loads(args.vision.read_text(encoding="utf-8"))["records"]
    candidates = select_candidates(manifest, results, vision_records,
                                   min_confidence=args.min_confidence)
    if args.limit is not None:
        candidates = candidates[:args.limit]
    context = review_context(ROOT, config, manifest, results, vision_records, lexicon,
                             min_confidence=args.min_confidence, scenes=("support", "bottom"))

    saved_records: dict[str, dict] = {}
    if output_path.exists():
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        if saved.get("context") != context:
            raise ValueError("Existing review file context differs; choose a new --output path")
        saved_records = {row["id"]: row for row in saved.get("records", [])}
    neighbors = _neighbor_index(manifest)
    client = load_client(config)
    raw_dir = ROOT / "data" / "raw"
    for index, (item, base, vision) in enumerate(candidates, start=1):
        if item["id"] in saved_records:
            continue
        saved_records[item["id"]] = review_image(
            client, item, base, vision, lexicon, config,
            raw_dir / item["image"], raw_dir, neighbors,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(output_path, {
            "context": context,
            "model_id": config["model_id"],
            "policy": "SigLIP2 trigger followed by Qwen local review",
            "count": len(saved_records),
            "records": sorted(saved_records.values(), key=lambda row: row["id"].lower()),
        })
        print(f"[review] {index}/{len(candidates)} {item['filename']}", flush=True)
    write_json(output_path, {
        "context": context,
        "model_id": config["model_id"],
        "policy": "SigLIP2 trigger followed by Qwen local review",
        "count": len(saved_records),
        "records": sorted(saved_records.values(), key=lambda row: row["id"].lower()),
    })
    print(json.dumps({"flagged": len(candidates), "reviewed": len(saved_records),
                      "valid": sum(row["valid"] for row in saved_records.values())},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
