"""Training-content deduplication ablation for the existing frozen visual heads."""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from evaluate_holdout import _make_holdout, _metrics, _write_report
from track4_agent.data import load_assets, read_json
from track4_agent.vision_classifier import feature_bank, fit_head, fit_knn, predict_head, predict_knn
from track4_agent.vision_review import candidate_path, digest_json, index_unique, sha256_file


def deduplicate_fitting(rows, image_records, *, exclude_conflicts):
    indexed = index_unique(image_records, "image")
    buckets = defaultdict(list)
    for row in rows:
        record = indexed[row["image"]]
        if "error" in record or not record.get("pixel_sha256"):
            raise ValueError(f"Unreadable audited training image: {row['image']}")
        buckets[(row["questionCategory"], record["pixel_sha256"])].append(row)
    selected = []
    excluded = []
    for bucket in buckets.values():
        if len({row["defectType"] for row in bucket}) > 1:
            if exclude_conflicts:
                excluded.extend(row["image"] for row in bucket)
            else:
                selected.extend(bucket)
        else:
            selected.append(min(bucket, key=lambda row: row["image"]))
    return selected, {"original_count": len(rows), "fitting_count": len(selected),
                      "excluded_conflicting_images": excluded,
                      "removed_duplicate_count": len(rows) - len(selected) - len(excluded)}


def run_experiment(workspace, train, full_lexicon, audit, features, groups, sample_size):
    if audit.get("training_manifest_sha256") != digest_json(train):
        raise ValueError("Training audit does not match current manifest")
    if audit.get("summary", {}).get("image_error_count") != 0:
        raise ValueError("Resolve audited image errors before fitting")
    indexed = index_unique(audit["image_records"], "image")
    positions = {row["image"]: index for index, row in enumerate(train)}
    if len(features) != len(train):
        raise ValueError("Feature count differs from training manifest")
    report = {"kind": "training-only visual-head deduplication ablation, not a submission",
              "training_manifest_sha256": digest_json(train), "sample_size": sample_size,
              "groups": {}, "scope": "All selected labels counted; no ratings/descriptions generated"}
    for group in groups:
        sample, fitting, split = _make_holdout(train, sample_size, 20261005, group)
        fitting_hashes = {indexed[row["image"]]["pixel_sha256"] for row in fitting}
        if any(indexed[row["image"]]["pixel_sha256"] in fitting_hashes for row in sample):
            raise ValueError(f"Held-out content overlaps fitting data: {group}")
        variants = {"original": (fitting, {"fitting_count": len(fitting)})}
        variants["same_label_dedup"] = deduplicate_fitting(fitting, audit["image_records"], exclude_conflicts=False)
        variants["dedup_exclude_conflicts"] = deduplicate_fitting(fitting, audit["image_records"], exclude_conflicts=True)
        group_report = {"split": split, "results": {}}
        for variant, (variant_rows, policy) in variants.items():
            result = {"training_policy": policy, "categories": {}}
            for category in ("桥梁", "轨道"):
                fit_rows = [row for row in variant_rows if row["questionCategory"] == category]
                queries = [row for row in sample if row["questionCategory"] == category]
                x = features[[positions[row["image"]] for row in fit_rows]]
                q = features[[positions[row["image"]] for row in queries]]
                labels = [row["defectType"] for row in fit_rows]
                if category == "桥梁":
                    head = fit_knn(x, labels, k=5)
                    predictions, _ = predict_knn(head, q)
                else:
                    head = fit_head(x, labels, multilabel=True)
                    predictions, _ = predict_head(head, q, full_lexicon["categories"][category]["allowed_labels"])
                records = [{"truth": row, "prediction": {"questionCategory": category, "defectType": label,
                                                          "ratingScale(1-5)": "", "defectDescription": ""}}
                           for row, label in zip(queries, predictions, strict=True)]
                metrics = _metrics(records)
                metrics.pop("rating_accuracy", None)
                metrics["category_metrics"][category].pop("rating_correct", None)
                result["categories"][category] = {"training_count": len(fit_rows),
                                                  "training_label_counts": dict(Counter(labels)),
                                                  "metrics": metrics, "records": records}
                print(f"[{group}/{variant}/{category}] exact={metrics['defect_type_exact_accuracy']:.4f} "
                      f"atomic_f1={metrics['atomic_macro_f1']:.4f} fit={len(fit_rows)}", flush=True)
            group_report["results"][variant] = result
        report["groups"][group] = group_report
    report["status"] = "complete"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=ROOT / "runs" / "training_audit_20261012" / "report.json")
    parser.add_argument("--groups", default="范家坪1号大桥,青树湾1号大桥")
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "dedup_heads" / "report.json")
    args = parser.parse_args()
    output = candidate_path(ROOT, args.output)
    if output.exists():
        parser.error("Output exists; select a new output path")
    groups = [value.strip() for value in args.groups.split(",") if value.strip()]
    if not groups or len(set(groups)) != len(groups):
        parser.error("Provide unique bridge groups")
    train, _, lexicon = load_assets(ROOT)
    audit = read_json(args.audit)
    if audit.get("training_manifest_sha256") != digest_json(train):
        parser.error("Training audit does not match current manifest")
    indexed = index_unique(audit["image_records"], "image")
    for row in train:
        if sha256_file(ROOT / "data" / "raw" / row["image"]) != indexed[row["image"]].get("file_sha256"):
            raise ValueError(f"Training image changed since audit: {row['image']}")
    cache_path = ROOT / "models" / "siglip2_train_features_cpu.npz"
    features = feature_bank(ROOT, train, ROOT / "models" / "SigLIP2-base-patch16-224",
                            cache_path=cache_path, device="cpu", threads=4)
    report = run_experiment(ROOT, train, lexicon, audit, features, groups, args.sample_size)
    report["audit_sha256"] = sha256_file(args.audit)
    report["feature_cache_sha256"] = sha256_file(cache_path)
    report["source_sha256"] = {name: sha256_file(CODE / name) for name in (
        "evaluate_dedup_heads.py", "evaluate_holdout.py", "track4_agent/vision_classifier.py")}
    _write_report(output, report)
    print(f"[saved] {output}", flush=True)


if __name__ == "__main__":
    main()
