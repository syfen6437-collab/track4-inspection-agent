"""Resumable label-blind track prompt experiment; never touches official artifacts."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from evaluate_holdout import _holdout_lexicon, _variants, run_holdout
from run_pipeline import load_config
from track4_agent.data import load_assets
from track4_agent.vision_review import index_unique, sha256_file

ROOT = Path(__file__).resolve().parent.parent


def select_track_sample(train, size, seed, excluded=()):
    index_unique(train, "image")
    pool = sorted((row for row in train if row["questionCategory"] == "轨道" and row["image"] not in set(excluded)),
                  key=lambda row: row["image"])
    if not 0 < size < len(pool):
        raise ValueError("Track sample must leave training images")
    random.Random(seed).shuffle(pool)
    return pool[:size]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "code" / "config" / "qwen3-vl-4b-offload.json")
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--exclude-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train, _, full_lexicon = load_assets(ROOT)
    excluded = set()
    exclusion_hash = None
    if args.exclude_report:
        prior = json.loads(args.exclude_report.read_text(encoding="utf-8"))
        if prior.get("status") != "complete":
            raise ValueError("Exclusion report must be complete")
        excluded = {row["truth"]["image"] for row in prior["baseline"]}
        exclusion_hash = sha256_file(args.exclude_report)
    sample = select_track_sample(train, args.sample_size, args.seed, excluded)
    query_images = {row["image"] for row in sample} | excluded
    fitting = [row for row in train if row["image"] not in query_images]
    lexicon = _holdout_lexicon(fitting, full_lexicon)
    lexicon.pop("review_guidance", None)
    config = load_config(args.config.resolve())
    # Keep the sole experimental factor explicit even if the caller changes the base config.
    config.update(track_label_definitions=False, track_atomic_checklist=False, training_description_guidance=False)
    variants = _variants(["baseline", "track_definitions"], config)
    split = {"kind": "label-blind random track-only prompt diagnostic", "seed": args.seed,
             "sample_size": len(sample), "excluded_count": len(excluded), "excluded_report_sha256": exclusion_hash,
             "runner_sha256": sha256_file(Path(__file__))}
    run_holdout(ROOT, config, sample, fitting, lexicon, split, variants, args.output.resolve(), resume=args.resume)


if __name__ == "__main__":
    main()
