from __future__ import annotations

"""Generate frozen-vision candidate labels for the unlabeled test set.

The classifier is fitted only on the prepared training manifest. The output
is an auxiliary artifact for candidate selection and never overwrites the
official result.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from track4_agent.data import load_assets, write_json
from track4_agent.vision_classifier import feature_bank, fit_head, predict_head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "logs" / "vision_test_predictions.json")
    args = parser.parse_args()

    train, test, lexicon = load_assets(ROOT)
    model_dir = ROOT / "models" / "SigLIP2-base-patch16-224"
    train_features = feature_bank(
        ROOT, train, model_dir,
        cache_path=ROOT / "models" / f"siglip2_train_features_{args.device}.npz",
        device=args.device, batch_size=args.batch_size, threads=args.threads,
    )
    test_features = feature_bank(
        ROOT, test, model_dir,
        cache_path=ROOT / "models" / f"siglip2_test_features_{args.device}.npz",
        device=args.device, batch_size=args.batch_size, threads=args.threads,
    )
    train_positions = {row["image"]: index for index, row in enumerate(train)}
    records = []
    for category in ("桥梁", "轨道"):
        fit_rows = [row for row in train if row["questionCategory"] == category]
        query_rows = [row for row in test if row["questionCategory"] == category]
        x_fit = train_features[[train_positions[row["image"]] for row in fit_rows]]
        query_positions = {row["image"]: index for index, row in enumerate(test)}
        x_query = test_features[[query_positions[row["image"]] for row in query_rows]]
        multilabel = category == "轨道"
        head = fit_head(x_fit, [row["defectType"] for row in fit_rows], multilabel=multilabel)
        labels, probabilities = predict_head(
            head, x_query, lexicon["categories"][category]["allowed_labels"]
        )
        for row, label, probability in zip(query_rows, labels, probabilities):
            scores = dict(zip(head["labels"], map(float, probability)))
            records.append({
                "id": row["id"],
                "questionCategory": category,
                "filename": row["filename"],
                "defectType": label,
                "confidence": max(scores.values()),
                "scores": scores,
            })
    records.sort(key=lambda row: row["id"].lower())
    payload = {
        "model_id": "google/siglip2-base-patch16-224",
        "fit_scope": "all prepared training labels; no test labels",
        "count": len(records),
        "category_counts": dict(Counter(row["questionCategory"] for row in records)),
        "label_counts": dict(Counter(row["defectType"] for row in records)),
        "records": records,
    }
    write_json(args.output, payload)
    print(json.dumps({key: payload[key] for key in ("count", "category_counts", "label_counts",)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
