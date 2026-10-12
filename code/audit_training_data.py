"""Read-only training data audit. No test images or predictions are inspected."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from evaluate_holdout import _bridge_group, _make_holdout, _write_report
from track4_agent.data import read_jsonl
from track4_agent.model import scene_key
from track4_agent.vision_review import candidate_path, digest_json, index_unique, sha256_file


def _image_record(raw_dir, row):
    path = (raw_dir / row["image"]).resolve()
    if raw_dir.resolve() not in path.parents:
        raise ValueError(f"Training image is outside data/raw: {row['image']}")
    record = {"image": row["image"], "questionCategory": row["questionCategory"],
              "bridgeName": row.get("bridgeName", ""), "defectType": row["defectType"],
              "ratingScale(1-5)": row.get("ratingScale(1-5)", ""),
              "defectDescription": row.get("defectDescription", "")}
    try:
        record["file_sha256"] = sha256_file(path)
        with Image.open(path) as source:
            with ImageOps.exif_transpose(source).convert("RGB") as image:
                digest = hashlib.sha256(f"RGB:{image.width}:{image.height}:".encode("ascii"))
                digest.update(image.tobytes())
                record.update(pixel_sha256=digest.hexdigest(), dimensions=list(image.size))
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def duplicate_groups(records, hash_key):
    groups = defaultdict(list)
    for record in records:
        if record.get(hash_key):
            groups[record[hash_key]].append(record)
    return [{"sha256": digest, "image_count": len(rows),
             "conflicting_labels": len({(row["questionCategory"], row["defectType"]) for row in rows}) > 1,
             "conflicting_ratings": len({row["ratingScale(1-5)"] for row in rows}) > 1,
             "conflicting_descriptions": len({row["defectDescription"] for row in rows}) > 1,
             "images": rows}
            for digest, rows in groups.items() if len(rows) > 1]


def fold_content_overlap(sample, fitting, records):
    indexed = index_unique(records, "image")
    query_by_hash, fitting_by_hash = defaultdict(list), defaultdict(list)
    for rows, target in ((sample, query_by_hash), (fitting, fitting_by_hash)):
        for row in rows:
            digest = indexed[row["image"]].get("pixel_sha256")
            if digest:
                target[digest].append(row["image"])
    return [{"pixel_sha256": digest, "heldout_images": query_by_hash[digest],
             "fitting_images": fitting_by_hash[digest]}
            for digest in sorted(query_by_hash.keys() & fitting_by_hash.keys())]


def label_summary(train):
    buckets = defaultdict(list)
    for row in train:
        buckets[(row["questionCategory"], row["defectType"])].append(row)
    return [{"questionCategory": category, "defectType": label, "count": len(rows),
             "bridge_groups": dict(Counter(_bridge_group(row) for row in rows)) if category == "桥梁" else {},
             "scenes": dict(Counter(scene_key(row) for row in rows)),
             "rating_distribution": dict(Counter(row.get("ratingScale(1-5)", "") for row in rows)),
             "blank_descriptions": sum(not row.get("defectDescription", "").strip() for row in rows)}
            for (category, label), rows in buckets.items()]


def audit_training(workspace, train, groups, *, sample_size=120, seed=20261005):
    index_unique(train, "image")
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("Provide unique bridge groups")
    folds = {}
    for group in groups:
        sample, fitting, split = _make_holdout(train, sample_size, seed, group)
        folds[group] = (sample, fitting, split)
    records = []
    for index, row in enumerate(train, 1):
        records.append(_image_record(workspace / "data" / "raw", row))
        if index % 100 == 0 or index == len(train):
            print(f"[training audit] {index}/{len(train)}", flush=True)
    file_duplicates = duplicate_groups(records, "file_sha256")
    pixel_duplicates = duplicate_groups(records, "pixel_sha256")
    errors = [row for row in records if "error" in row]
    report = {"kind": "training-only read-only data quality audit", "training_count": len(train),
              "training_manifest_sha256": digest_json(train), "seed": seed,
              "implementation_sha256": sha256_file(Path(__file__)),
              "image_errors": errors, "file_duplicate_groups": file_duplicates,
              "pixel_duplicate_groups": pixel_duplicates, "labels": label_summary(train),
              "image_records": records, "folds": {}}
    for group, (sample, fitting, split) in folds.items():
        fitting_ids = {row["image"] for row in fitting}
        excluded = [row for row in train if row["image"] not in fitting_ids]
        report["folds"][group] = {
            "split": split, "fitting_manifest_sha256": digest_json(fitting),
            "sample_manifest_sha256": digest_json(sample),
            "evaluated_content_overlap": fold_content_overlap(sample, fitting, records),
            "all_excluded_content_overlap": fold_content_overlap(excluded, fitting, records),
        }
    report["summary"] = {
        "readable_training_images": len(train) - len(errors), "image_error_count": len(errors),
        "file_duplicate_group_count": len(file_duplicates),
        "pixel_duplicate_group_count": len(pixel_duplicates),
        "pixel_conflicting_label_groups": sum(row["conflicting_labels"] for row in pixel_duplicates),
        "pixel_conflicting_rating_groups": sum(row["conflicting_ratings"] for row in pixel_duplicates),
        "folds_with_evaluated_content_overlap": sum(bool(fold["evaluated_content_overlap"])
                                                   for fold in report["folds"].values()),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", default="范家坪1号大桥,青树湾1号大桥")
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "training_audit" / "report.json")
    args = parser.parse_args()
    output = candidate_path(ROOT, args.output)
    if output.exists():
        parser.error("Output exists; select a new --output to preserve the previous audit")
    train = read_jsonl(ROOT / "data" / "processed" / "train_manifest.jsonl")
    groups = [value.strip() for value in args.groups.split(",") if value.strip()]
    report = audit_training(ROOT, train, groups, sample_size=args.sample_size, seed=args.seed)
    _write_report(output, report)
    print(json.dumps({"summary": report["summary"], "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
