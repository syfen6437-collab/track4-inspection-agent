from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from evaluate_holdout import _make_holdout, _metrics
from track4_agent.data import load_assets, write_json


def _features(
    rows: list[dict[str, Any]],
    model: Any,
    processor: Any,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    features: list[np.ndarray] = []
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    inverse = np.zeros(len(rows), dtype=np.int64)
    inverse[np.asarray(order)] = np.arange(len(order))
    shuffled = [rows[index] for index in order]

    for start in range(0, len(shuffled), batch_size):
        batch_rows = shuffled[start : start + batch_size]
        images = []
        for row in batch_rows:
            with Image.open(WORKSPACE / "data" / "raw" / row["image"]) as source:
                images.append(ImageOps.exif_transpose(source).convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to("cuda", dtype=torch.float16)
        with torch.inference_mode():
            embedding = model.get_image_features(pixel_values=pixel_values)
            if not isinstance(embedding, torch.Tensor):
                embedding = embedding.pooler_output
            embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
        features.extend(embedding.cpu().numpy())
        del inputs, pixel_values, embedding, images
        if (start // batch_size + 1) % 20 == 0 or start + batch_size >= len(shuffled):
            print(f"[siglip2] embedded {min(start + batch_size, len(shuffled))}/{len(shuffled)}", flush=True)

    shuffled_features = np.stack(features)
    return shuffled_features[inverse]


def _fit_linear_head(
    train_features: np.ndarray,
    train_labels: list[str],
    query_features: np.ndarray,
) -> tuple[list[str], dict[str, int]]:
    labels = sorted(set(train_labels))
    label_to_index = {label: index for index, label in enumerate(labels)}
    x_train = torch.from_numpy(train_features).to("cuda", dtype=torch.float32)
    x_query = torch.from_numpy(query_features).to("cuda", dtype=torch.float32)
    mean = x_train.mean(dim=0)
    scale = x_train.std(dim=0).clamp_min(1e-4)
    x_train = (x_train - mean) / scale
    x_query = (x_query - mean) / scale
    targets = torch.tensor(
        [label_to_index[label] for label in train_labels], device="cuda", dtype=torch.long
    )
    counts = torch.bincount(targets, minlength=len(labels)).float()
    class_weights = (counts.sum() / counts.clamp_min(1)).sqrt()
    class_weights = class_weights / class_weights.mean()

    head = torch.nn.Linear(x_train.shape[1], len(labels), device="cuda")
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.02, weight_decay=0.05)
    for epoch in range(250):
        optimizer.zero_grad(set_to_none=True)
        logits = head(x_train)
        loss = torch.nn.functional.cross_entropy(logits, targets, weight=class_weights)
        loss.backward()
        optimizer.step()
        if epoch in {79, 159}:
            for group in optimizer.param_groups:
                group["lr"] *= 0.3

    with torch.inference_mode():
        predictions = head(x_query).argmax(dim=-1).cpu().tolist()
    predicted_labels = [labels[index] for index in predictions]
    del head, optimizer, x_train, x_query, targets, class_weights
    torch.cuda.empty_cache()
    return predicted_labels, dict(zip(labels, counts.cpu().int().tolist()))


def main() -> None:
    parser = argparse.ArgumentParser(description="SigLIP2冻结特征 + 监督分类头整桥留出探针")
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--bridge-group", default="范家坪1号大桥")
    parser.add_argument("--min-class-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "logs" / "siglip2_holdout_probe.json")
    args = parser.parse_args()

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["MODELSCOPE_OFFLINE"] = "1"
    torch.manual_seed(20261005)
    train, _, _ = load_assets(WORKSPACE)
    sample, remaining, split = _make_holdout(
        train, args.sample_size, 20261005, args.bridge_group
    )
    feature_rows = remaining + sample
    model_dir = WORKSPACE / "models" / "SigLIP2-base-patch16-224"
    model = AutoModel.from_pretrained(
        model_dir,
        dtype=torch.float16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    embeddings = _features(feature_rows, model, processor, args.batch_size, 20261005)
    positions = {row["image"]: index for index, row in enumerate(feature_rows)}
    train_x = np.stack([embeddings[positions[row["image"]]] for row in remaining])
    sample_x = np.stack([embeddings[positions[row["image"]]] for row in sample])

    report: dict[str, Any] = {
        "model": "google/siglip2-base-patch16-224",
        "model_license": "Apache-2.0",
        "training_method": "frozen image embeddings + class-balanced PyTorch linear classifier",
        "split": split,
        "minimum_training_examples_per_class": args.min_class_count,
        "results": {},
    }
    for category in ("桥梁", "轨道"):
        train_indices = [i for i, row in enumerate(remaining) if row["questionCategory"] == category]
        sample_indices = [i for i, row in enumerate(sample) if row["questionCategory"] == category]
        label_counts = Counter(remaining[i]["defectType"] for i in train_indices)
        labels = {label for label, count in label_counts.items() if count >= args.min_class_count}
        usable_train = [i for i in train_indices if remaining[i]["defectType"] in labels]
        usable_sample = [i for i in sample_indices if sample[i]["defectType"] in labels]
        if not usable_train or not usable_sample:
            report["results"][category] = {
                "evaluated": 0,
                "excluded_unseen_or_rare_labels": dict(Counter(
                    sample[i]["defectType"] for i in sample_indices
                    if sample[i]["defectType"] not in labels
                )),
            }
            continue

        category_train_y = [remaining[i]["defectType"] for i in usable_train]
        predictions, fitted_counts = _fit_linear_head(
            train_x[usable_train],
            category_train_y,
            sample_x[usable_sample],
        )
        records = [
            {
                "truth": sample[index],
                "prediction": {
                    **{key: sample[index].get(key, "") for key in (
                        "questionCategory", "bridgeName", "defectLocation", "filename"
                    )},
                    "defectType": str(prediction),
                    "defectDescription": "",
                    "ratingScale(1-5)": "",
                },
                "raw": [],
            }
            for index, prediction in zip(usable_sample, predictions)
        ]
        metrics = _metrics(records)
        report["results"][category] = {
            **metrics,
            "evaluated": len(records),
            "training_class_count": len(labels),
            "training_label_counts": fitted_counts,
            "excluded_unseen_or_rare_labels": dict(Counter(
                sample[i]["defectType"] for i in sample_indices
                if sample[i]["defectType"] not in labels
            )),
        }

    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
