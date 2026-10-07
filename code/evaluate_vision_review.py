from __future__ import annotations

"""Paired, resumable evaluation of the same review policy used on test images."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from evaluate_holdout import _holdout_lexicon, _make_holdout, _metrics
from evaluate_hybrid import final_confidence
from run_pipeline import load_client, load_config
from track4_agent.data import OUTPUT_KEYS, load_assets, write_json
from track4_agent.inference import infer_item
from track4_agent.model import sanitize_prediction, scene_key
from track4_agent.vision_classifier import (
    feature_bank, fit_head, fit_knn, model_identity, predict_head, predict_knn,
)
from track4_agent.vision_review import (
    DEFAULT_SCENES, candidate_path, digest_json, index_unique, merge_reviews,
    probability, review_context, review_image, select_candidates, sha256_file,
)


def input_item(row: dict[str, Any]) -> dict[str, Any]:
    # Training targets must not enter model input metadata or prompts.
    return {"id": row["image"], **{key: row.get(key, "") for key in (
        "image", "questionCategory", "bridgeName", "defectLocation", "filename")}}


def validate_baseline(record: dict[str, Any], truth: dict[str, Any], lexicon: dict[str, Any]) -> None:
    if record["truth"] != truth or set(record["prediction"]) != set(OUTPUT_KEYS):
        raise ValueError("Saved baseline truth or output schema differs")
    item = input_item(truth)
    for key in ("questionCategory", "bridgeName", "defectLocation", "filename"):
        if record["prediction"][key] != item[key]:
            raise ValueError(f"Baseline metadata mismatch: {item['id']}")
    for raw in record["raw"]:
        if raw["round"] not in (1, 2):
            continue
        prediction, valid = sanitize_prediction(raw["raw"], lexicon, item["questionCategory"], scene_key(item))
        if prediction != raw.get("prediction") or valid != raw.get("valid"):
            raise ValueError(f"Baseline raw text differs from stored prediction: {item['id']}")
    final_confidence(record)


def fold_predictions(sample, training, lexicon, features, positions, *, bridge_method="linear", knn_k=5):
    records = []
    for category in ("桥梁", "轨道"):
        fitting = [row for row in training if row["questionCategory"] == category]
        queries = [row for row in sample if row["questionCategory"] == category]
        x_fit = features[[positions[row["image"]] for row in fitting]]
        x_query = features[[positions[row["image"]] for row in queries]]
        if category == "桥梁" and bridge_method == "knn":
            head = fit_knn(x_fit, [row["defectType"] for row in fitting], k=knn_k)
            labels, scores = predict_knn(head, x_query)
        else:
            head = fit_head(x_fit, [row["defectType"] for row in fitting], multilabel=category == "轨道")
            labels, scores = predict_head(head, x_query,
                                          lexicon["categories"][category]["allowed_labels"])
        for row, label, values in zip(queries, labels, scores):
            records.append({"id": row["image"], "questionCategory": category,
                            "filename": row["filename"], "defectType": label,
                            "confidence": float(max(values)),
                            "scores": dict(zip(head["labels"], map(float, values)))})
    return records


def policy_records(baselines, vision_records, reviews, lexicon, *, min_confidence, review_min_confidence):
    vision_by_id = index_unique(vision_records, "id")
    review_by_id = index_unique(reviews, "id")
    expected = set()
    evaluated, audit = [], []
    for record in baselines:
        item = input_item(record["truth"])
        base, vision = record["prediction"], vision_by_id[item["id"]]
        selected = select_candidates([item], [base], [vision], lexicon,
                                     min_confidence=min_confidence, scenes=DEFAULT_SCENES)
        final = base
        if selected:
            expected.add(item["id"])
            if item["id"] not in review_by_id:
                raise ValueError(f"Missing paired review: {item['id']}")
            merged, changes = merge_reviews([item], [base], [vision], [review_by_id[item["id"]]], lexicon,
                                             min_confidence=min_confidence,
                                             review_min_confidence=review_min_confidence,
                                             scenes=DEFAULT_SCENES)
            final = merged[0]
            audit.extend(changes)
        evaluated.append({"truth": record["truth"], "prediction": final})
    if set(review_by_id) != expected:
        raise ValueError("Saved reviews contain stale or unselected images")
    gained = lost = 0
    for base, final in zip(baselines, evaluated):
        truth = base["truth"]["defectType"]
        was_correct = base["prediction"]["defectType"] == truth
        is_correct = final["prediction"]["defectType"] == truth
        gained += int(is_correct and not was_correct)
        lost += int(was_correct and not is_correct)
    return {"metrics": _metrics(evaluated), "gained_correct": gained, "lost_correct": lost,
            "flagged": len(audit), "accepted": sum(row["accepted"] for row in audit),
            "audit": audit, "records": evaluated}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", default="范家坪1号大桥,青树湾1号大桥")
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--config", type=Path, default=CODE / "config" / "qwen3-vl-4b-review.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "review_validation_v3" / "paired_review.json")
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--review-min-confidence", type=float, default=0.55)
    parser.add_argument("--bridge-method", choices=("linear", "knn"), default="linear")
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--prepare-only", action="store_true", help="Prepare CPU folds without loading Qwen")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    probability(args.min_confidence)
    probability(args.review_min_confidence)
    output = candidate_path(ROOT, args.output)
    if output.exists() and not args.resume:
        parser.error("Output already exists; use --resume or a new --output")
    groups = [group.strip() for group in args.groups.split(",") if group.strip()]
    if not groups or len(set(groups)) != len(groups):
        parser.error("Provide nonempty unique --groups")
    train, _, full_lexicon = load_assets(ROOT)
    config = load_config(args.config.resolve())
    features_path = ROOT / "models" / "siglip2_train_features_cpu.npz"
    features = feature_bank(ROOT, train, ROOT / "models" / "SigLIP2-base-patch16-224",
                            cache_path=features_path, threads=args.threads)
    positions = {row["image"]: index for index, row in enumerate(train)}
    folds = {}
    for group in groups:
        sample, training, split = _make_holdout(train, args.sample_size, int(config["seed"]), group)
        lexicon = _holdout_lexicon(training, full_lexicon)
        vision = fold_predictions(sample, training, lexicon, features, positions,
                                  bridge_method=args.bridge_method, knn_k=args.knn_k)
        folds[group] = {"sample": sample, "lexicon": lexicon, "split": split, "vision": vision,
                        "training_sha256": digest_json(training)}

    context = review_context(
        ROOT, config, [input_item(row) for fold in folds.values() for row in fold["sample"]],
        [], [row for fold in folds.values() for row in fold["vision"]], full_lexicon,
        min_confidence=args.min_confidence, scenes=DEFAULT_SCENES)
    context.update({
        "kind": "paired whole-bridge holdout", "groups": groups, "sample_size": args.sample_size,
        "review_min_confidence": args.review_min_confidence,
        "bridge_method": args.bridge_method, "knn_k": args.knn_k,
        "training_manifest_sha256": digest_json(train),
        "feature_cache_sha256": sha256_file(features_path),
        "vision_model_identity": model_identity(ROOT / "models" / "SigLIP2-base-patch16-224"),
        "head_implementation_sha256": sha256_file(CODE / "track4_agent" / "vision_classifier.py"),
        "evaluation_implementation_sha256": {
            name: sha256_file(CODE / name) for name in (
                "evaluate_vision_review.py", "evaluate_holdout.py", "evaluate_hybrid.py")},
        "folds": {group: {"training_sha256": fold["training_sha256"],
                          "lexicon_sha256": digest_json(fold["lexicon"])} for group, fold in folds.items()},
    })
    report = {"context": context, "status": "prepared", "groups": {}}
    if output.exists():
        report = json.loads(output.read_text(encoding="utf-8"))
        if report.get("context") != context:
            raise ValueError("Resume context differs; use a new output path")
        if set(report["groups"]) != set(groups):
            raise ValueError("Resume fold coverage differs")
    for group, fold in folds.items():
        stored = report["groups"].setdefault(group, {
            "split": fold["split"], "vision_records": fold["vision"],
            "baseline_records": [], "review_records": []})
        if stored["split"] != fold["split"] or stored["vision_records"] != fold["vision"]:
            raise ValueError("Saved split or fold classifier predictions differ")
        baselines = index_unique([{**record, "id": record["truth"]["image"]}
                                  for record in stored["baseline_records"]], "id")
        sample_by_id = {row["image"]: row for row in fold["sample"]}
        if not set(baselines) <= set(sample_by_id):
            raise ValueError("Baseline contains stale images")
        for identity, record in baselines.items():
            validate_baseline(record, sample_by_id[identity], fold["lexicon"])
        index_unique(stored["review_records"], "id")
    write_json(output, report)
    print(f"[prepared] groups={len(folds)} samples={sum(len(f['sample']) for f in folds.values())}", flush=True)
    if args.prepare_only:
        return

    client = None
    raw_dir = ROOT / "data" / "raw"
    for group, fold in folds.items():
        stored = report["groups"][group]
        baselines = {row["truth"]["image"]: row for row in stored["baseline_records"]}
        for index, truth in enumerate(fold["sample"], start=1):
            if truth["image"] in baselines:
                continue
            if client is None:
                client = load_client(config)
            item = input_item(truth)
            prediction, raw = infer_item(client, item, fold["lexicon"], config, raw_dir / item["image"],
                                         raw_dir=raw_dir, allow_review=item["questionCategory"] == "轨道")
            record = {"truth": truth, "prediction": prediction, "raw": raw}
            validate_baseline(record, truth, fold["lexicon"])
            baselines[item["id"]] = record
            stored["baseline_records"] = [baselines[row["image"]] for row in fold["sample"]
                                           if row["image"] in baselines]
            report["status"] = "baseline_running"
            write_json(output, report)
            print(f"[{group}/baseline] {index}/{len(fold['sample'])}", flush=True)
        stored["baseline_metrics"] = _metrics(stored["baseline_records"])
        vision_by_id = index_unique(fold["vision"], "id")
        selected = []
        for record in stored["baseline_records"]:
            item = input_item(record["truth"])
            selected.extend(select_candidates([item], [record["prediction"]], [vision_by_id[item["id"]]],
                                               fold["lexicon"], min_confidence=args.min_confidence))
        reviews = index_unique(stored["review_records"], "id")
        if not set(reviews) <= {item["id"] for item, _, _ in selected}:
            raise ValueError("Resume contains unselected reviews")
        for index, (item, base, vision) in enumerate(selected, start=1):
            if item["id"] in reviews:
                continue
            if client is None:
                client = load_client(config)
            reviews[item["id"]] = review_image(client, item, base, vision, fold["lexicon"], config,
                                                raw_dir / item["image"], raw_dir)
            stored["review_records"] = sorted(reviews.values(), key=lambda row: row["id"])
            report["status"] = "review_running"
            write_json(output, report)
            print(f"[{group}/review] {index}/{len(selected)}", flush=True)
        stored["policy"] = policy_records(stored["baseline_records"], fold["vision"], stored["review_records"],
                                          fold["lexicon"], min_confidence=args.min_confidence,
                                          review_min_confidence=args.review_min_confidence)
        write_json(output, report)
    report["status"] = "complete"
    write_json(output, report)
    print(json.dumps({group: {"baseline": data["baseline_metrics"],
                             "review": data["policy"]["metrics"],
                             "gained": data["policy"]["gained_correct"],
                             "lost": data["policy"]["lost_correct"]}
                      for group, data in report["groups"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
