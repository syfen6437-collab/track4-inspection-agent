from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from evaluate_holdout import _make_holdout, _metrics
from track4_agent.data import load_assets, write_json
from track4_agent.vision_classifier import (
    feature_bank, fit_head, fit_knn, predict_head, predict_knn, model_identity,
)
from track4_agent.vision_review import digest_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="完整标签范围的跨桥监督视觉模型评测")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--groups", default="范家坪1号大桥,青树湾1号大桥")
    parser.add_argument("--min-class-count", type=int, default=1,
                        help="默认保留所有训练标签，与测试候选头一致；4仅用于复现旧实验")
    parser.add_argument("--bridge-method", choices=("linear", "knn"), default="linear")
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="仅用于特征抽取冒烟，不允许生成指标")
    parser.add_argument("--output", type=Path, default=ROOT / "logs" / "vision_heads_holdout.json")
    args = parser.parse_args()
    if args.min_class_count < 1:
        parser.error("--min-class-count must be positive")
    if args.limit and not args.cache_only:
        parser.error("--limit requires --cache-only")
    train, _, lexicon = load_assets(ROOT)
    started = time.time()
    features = feature_bank(
        ROOT, train[:args.limit] if args.limit else train,
        ROOT / "models" / "SigLIP2-base-patch16-224",
        cache_path=ROOT / "models" / f"siglip2_train_features_{args.device}.npz",
        device=args.device, batch_size=args.batch_size, threads=args.threads,
    )
    if args.cache_only:
        print(f"[cache] images={len(features)} elapsed={time.time()-started:.1f}s", flush=True)
        return
    positions = {row["image"]: index for index, row in enumerate(train)}
    report = {
        "model_id": "google/siglip2-base-patch16-224", "license": "Apache-2.0",
        "device": args.device, "seed": 20261005,
        "min_class_count": args.min_class_count,
        "bridge_method": args.bridge_method, "knn_k": args.knn_k,
        "training_manifest_sha256": digest_json(train),
        "feature_cache_sha256": sha256_file(ROOT / "models" / f"siglip2_train_features_{args.device}.npz"),
        "model_identity": model_identity(ROOT / "models" / "SigLIP2-base-patch16-224"),
        "head_implementation_sha256": sha256_file(CODE / "track4_agent" / "vision_classifier.py"),
        "evaluation_scope": "All selected holdout labels counted, including unseen and rare labels",
        "groups": {},
    }
    for group in (value.strip() for value in args.groups.split(",") if value.strip()):
        sample, training, split = _make_holdout(train, args.sample_size, 20261005, group)
        group_report = {"split": split, "results": {}}
        for category in ("桥梁", "轨道"):
            fitting = [row for row in training if row["questionCategory"] == category]
            queries = [row for row in sample if row["questionCategory"] == category]
            counts = Counter(row["defectType"] for row in fitting)
            supported = [row for row in fitting if counts[row["defectType"]] >= args.min_class_count]
            x_query = features[[positions[row["image"]] for row in queries]]
            methods = ["multiclass"] + (["atomic_multilabel"] if category == "轨道" else [])
            for method in methods:
                fit_rows = fitting if method == "atomic_multilabel" else supported
                x = features[[positions[row["image"]] for row in fit_rows]]
                if category == "桥梁" and method == "multiclass" and args.bridge_method == "knn":
                    head = fit_knn(x, [row["defectType"] for row in fit_rows], k=args.knn_k)
                    predicted, probabilities = predict_knn(head, x_query)
                else:
                    head = fit_head(x, [row["defectType"] for row in fit_rows], multilabel=method == "atomic_multilabel")
                    predicted, probabilities = predict_head(head, x_query, lexicon["categories"][category]["allowed_labels"])
                records = []
                for row, label, probability in zip(queries, predicted, probabilities):
                    records.append({
                        "truth": row,
                        "prediction": {"questionCategory": category, "defectType": label,
                                       "ratingScale(1-5)": "", "defectDescription": ""},
                        "scores": dict(zip(head.get("vocabulary", head["labels"]),
                                           map(float, probability))),
                    })
                metrics = _metrics(records)
                # Rating and description are not generated by this classifier.
                metrics.pop("rating_accuracy", None)
                group_report["results"][f"{category}/{method}"] = {
                    "metrics": metrics,
                    "training_count": len(fit_rows),
                    "training_label_counts": dict(Counter(row["defectType"] for row in fit_rows)),
                    "unsupported_truth_counts": dict(Counter(row["defectType"] for row in queries
                        if row["defectType"] not in {r["defectType"] for r in fit_rows})),
                    "records": records,
                }
                print(f"[{group}/{category}/{method}] exact={metrics['defect_type_exact_accuracy']:.4f} "
                      f"atomic_macro_f1={metrics['atomic_macro_f1']:.4f} n={len(records)}", flush=True)
        report["groups"][group] = group_report
        report["elapsed_seconds"] = round(time.time() - started, 3)
        write_json(args.output, report)
    print(f"[evaluation] saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
