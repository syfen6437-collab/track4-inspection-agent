"""Compare frozen visual views on whole bridges and a label-blind random track holdout."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from evaluate_holdout import _bridge_group, _metrics, _paired_comparison, _report_lock, _write_report
from track4_agent.data import load_assets
from track4_agent.vision_classifier import fit_head, fit_knn, model_identity, predict_head, predict_knn
from track4_agent.vision_review import candidate_path, digest_json, index_unique, sha256_file


def natural_splits(train, groups, track_size, seed):
    index_unique(train, "image")
    track = sorted((row for row in train if row["questionCategory"] == "轨道"), key=lambda x: x["image"])
    if not 0 < track_size < len(track):
        raise ValueError("Track holdout size must leave both fitting and query images")
    random.Random(seed).shuffle(track)
    queries = track[:track_size]
    fitting = track[track_size:]
    splits = {"轨道/random": (fitting, queries)}
    if len(set(groups)) != len(groups):
        raise ValueError("Bridge groups must be unique")
    for group in groups:
        bridge = [row for row in train if row["questionCategory"] == "桥梁"]
        queries = [row for row in bridge if _bridge_group(row) == group]
        fitting = [row for row in bridge if _bridge_group(row) != group]
        if not queries or not fitting:
            raise ValueError(f"Bridge split has no fitting or query images: {group}")
        splits[f"桥梁/{group}"] = (fitting, queries)
    return splits


def checked_bank(workspace, train, path, views):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"].item()))
        if metadata.get("views", "global") != views:
            raise ValueError("Feature view differs from requested comparison")
        keys = list(map(str, saved["keys"]))
        vectors = np.array(saved["vectors"], dtype=np.float32, copy=True)
    if len(set(keys)) != len(keys) or vectors.ndim != 2 or len(vectors) != len(keys) or not np.isfinite(vectors).all():
        raise ValueError("Invalid feature cache schema or nonfinite vectors")
    positions = {key: i for i, key in enumerate(keys)}
    selected = []
    for row in train:
        stat = (workspace / "data" / "raw" / row["image"]).stat()
        key = json.dumps((row["image"], stat.st_size, stat.st_mtime_ns), ensure_ascii=False)
        if key not in positions:
            raise ValueError(f"Feature cache is incomplete or image changed: {row['image']}")
        selected.append(positions[key])
    return vectors[selected], metadata


def check_content_audit(workspace, train, splits, audit):
    if audit.get("training_manifest_sha256") != digest_json(train):
        raise ValueError("Training audit manifest differs")
    records = index_unique(audit["image_records"], "image")
    if set(records) != {row["image"] for row in train}:
        raise ValueError("Training audit coverage differs")
    for row in train:
        record = records[row["image"]]
        if not record.get("pixel_sha256") or sha256_file(workspace / "data" / "raw" / row["image"]) != record["file_sha256"]:
            raise ValueError("Training image differs from content audit")
    for fitting, queries in splits.values():
        fit_content = {records[row["image"]]["pixel_sha256"] for row in fitting}
        query_content = {records[row["image"]]["pixel_sha256"] for row in queries}
        if fit_content & query_content:
            raise ValueError("Exact pixel content crosses fitting and query partitions")


def label_metrics(records):
    metrics = _metrics(records)
    metrics.pop("rating_accuracy", None)
    for value in metrics["category_metrics"].values():
        value.pop("rating_correct", None)
    return metrics


def evaluate_views(train, lexicon, splits, banks, seed):
    positions = {row["image"]: i for i, row in enumerate(train)}
    results = {}
    for name, (fitting, queries) in splits.items():
        category = queries[0]["questionCategory"]
        variants = {}
        for view, bank in banks.items():
            x_fit = bank[[positions[row["image"]] for row in fitting]]
            x_query = bank[[positions[row["image"]] for row in queries]]
            labels = [row["defectType"] for row in fitting]
            if category == "桥梁":
                head = fit_knn(x_fit, labels, k=5)
                predictions, _ = predict_knn(head, x_query)
            else:
                head = fit_head(x_fit, labels, multilabel=True, seed=seed)
                predictions, _ = predict_head(head, x_query, lexicon["categories"][category]["allowed_labels"])
            records = [{"truth": row, "prediction": {"questionCategory": category,
                        "defectType": label, "ratingScale(1-5)": ""}} for row, label in zip(queries, predictions, strict=True)]
            variants[view] = {"records": records, "metrics": label_metrics(records)}
        paired = _paired_comparison(variants["global"]["records"], variants["quadrants"]["records"])
        paired["baseline_metrics"] = label_metrics(variants["global"]["records"])
        paired["candidate_metrics"] = label_metrics(variants["quadrants"]["records"])
        results[name] = {"fitting_count": len(fitting), "query_count": len(queries),
                         "fitting_sha256": digest_json(fitting), "query_sha256": digest_json(queries),
                         "variants": variants, "comparison": paired}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-cache", type=Path, required=True)
    parser.add_argument("--quadrants-cache", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=ROOT / "runs" / "training_audit_20261012" / "report.json")
    parser.add_argument("--groups", default="范家坪1号大桥,青树湾1号大桥")
    parser.add_argument("--track-size", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = candidate_path(ROOT, args.output)
    torch.set_num_threads(4)
    with _report_lock(output):
        if output.exists():
            raise ValueError("Output exists; choose a new output path")
        train, _, lexicon = load_assets(ROOT)
        groups = [x.strip() for x in args.groups.split(",") if x.strip()]
        splits = natural_splits(train, groups, args.track_size, args.seed)
        audit = json.loads(args.audit.read_text(encoding="utf-8"))
        check_content_audit(ROOT, train, splits, audit)
        banks = {}
        metadata = {}
        paths = {"global": args.global_cache, "quadrants": args.quadrants_cache}
        for name, path in paths.items():
            banks[name], metadata[name] = checked_bank(ROOT, train, path, name)
        for field in ("model", "device", "dtype"):
            if metadata["global"].get(field) != metadata["quadrants"].get(field):
                raise ValueError(f"Comparison confound: feature {field} differs")
        if metadata["global"]["model"] != model_identity(ROOT / "models" / "SigLIP2-base-patch16-224"):
            raise ValueError("Cached model weights/processor identity changed")
        report = {"kind": "natural distribution label-only paired visual view comparison",
                  "status": "complete", "seed": args.seed, "track_holdout_is_shared_once": True,
                  "training_manifest_sha256": digest_json(train), "audit_sha256": sha256_file(args.audit),
                  "metadata": metadata, "cache_sha256": {k: sha256_file(v) for k, v in paths.items()},
                  "implementation_sha256": {str(p.relative_to(CODE)): sha256_file(p) for p in (
                      Path(__file__), CODE / "track4_agent" / "vision_classifier.py", CODE / "evaluate_holdout.py")},
                  "results": evaluate_views(train, lexicon, splits, banks, args.seed)}
        _write_report(output, report)
        for name, result in report["results"].items():
            print(name, {k: v["metrics"]["defect_type_exact_accuracy"] for k, v in result["variants"].items()}, flush=True)


if __name__ == "__main__":
    main()
