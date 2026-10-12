from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from run_pipeline import load_client, load_config
from track4_agent.data import OUTPUT_KEYS, _build_lexicon, load_assets
from track4_agent.inference import _macro_atomic_f1, infer_item
from track4_agent.model import candidate_labels, scene_key
from track4_agent.vision_review import candidate_path, digest_json, index_unique, sha256_file


def _bridge_group(row: dict[str, Any]) -> str:
    name = str(row.get("bridgeName", "")).strip()
    return re.sub(r"[（(](左幅|右幅)[）)]$", "", name).strip()


def _round_robin_sample(
    rows: list[dict[str, Any]], size: int, rng: random.Random
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["defectType"])].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    labels = list(buckets)
    rng.shuffle(labels)
    selected: list[dict[str, Any]] = []
    while len(selected) < min(size, len(rows)) and labels:
        for label in list(labels):
            if buckets[label]:
                selected.append(buckets[label].pop())
            if not buckets[label]:
                labels.remove(label)
            if len(selected) >= size:
                break
    return selected


def _make_holdout(
    train: list[dict[str, Any]], size: int, seed: int, bridge_group: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if size <= 0:
        raise ValueError("Sample size must be positive")
    index_unique(train, "image")
    rng = random.Random(seed)
    bridge_pool = [
        row for row in train
        if row["questionCategory"] == "桥梁" and _bridge_group(row) == bridge_group
    ]
    track_pool = [row for row in train if row["questionCategory"] == "轨道"]
    bridge_size = min(len(bridge_pool), round(size * 0.7))
    track_size = min(len(track_pool), size - bridge_size)
    sample = _round_robin_sample(bridge_pool, bridge_size, rng)
    sample.extend(_round_robin_sample(track_pool, track_size, rng))
    selected_images = {row["image"] for row in sample}
    heldout_bridge_images = {row["image"] for row in bridge_pool}
    training = [
        row for row in train
        if row["image"] not in selected_images and row["image"] not in heldout_bridge_images
    ]
    if len(sample) < size:
        raise ValueError(
            f"Only selected {len(sample)} of {size} requested samples; "
            f"check holdout bridge {bridge_group!r} and category counts"
        )
    return sample, training, {
        "bridge_group_held_out": bridge_group,
        "bridge_group_image_count": len(bridge_pool),
        "sample_size": len(sample),
        "sample_category_counts": dict(Counter(row["questionCategory"] for row in sample)),
        "sample_label_counts": dict(Counter(
            f"{row['questionCategory']}|{row['defectType']}" for row in sample
        )),
    }


def _holdout_lexicon(
    training: list[dict[str, Any]], full_lexicon: dict[str, Any]
) -> dict[str, Any]:
    lexicon = _build_lexicon(training)
    for category, full_data in full_lexicon["categories"].items():
        category_data = lexicon["categories"].setdefault(category, {
            "sample_count": 0,
            "allowed_labels": [],
            "label_specs": {},
            "scene_label_counts": {},
        })
        category_data["allowed_labels"] = list(full_data["allowed_labels"])
        for label in full_data["allowed_labels"]:
            category_data["label_specs"].setdefault(label, {
                "count": 0,
                "descriptions": [],
                "rating_distribution": {},
                "default_rating": "",
            })
    return lexicon


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    exact = sum(row["truth"]["defectType"] == row["prediction"]["defectType"] for row in records)
    rating = sum(
        row["truth"]["ratingScale(1-5)"] == row["prediction"]["ratingScale(1-5)"]
        for row in records
    )
    pairs = [(row["truth"]["defectType"], row["prediction"]["defectType"]) for row in records]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_category[row["truth"]["questionCategory"]].append(row)
        by_label[row["truth"]["defectType"]].append(row)
        by_scene[scene_key(row["truth"])].append(row)
    nonhealthy = [row for row in records if row["truth"]["defectType"] != "完好"]
    nonhealthy_recall = sum(row["prediction"]["defectType"] != "完好" for row in nonhealthy)
    return {
        "sample_size": len(records),
        "defect_type_exact_accuracy": round(exact / len(records), 6) if records else 0.0,
        "atomic_macro_f1": round(_macro_atomic_f1(pairs), 6),
        "rating_accuracy": round(rating / len(records), 6) if records else 0.0,
        "nonhealthy_recall": round(nonhealthy_recall / len(nonhealthy), 6) if nonhealthy else None,
        "category_accuracy": {
            category: round(
                sum(r["truth"]["defectType"] == r["prediction"]["defectType"] for r in rows)
                / len(rows),
                6,
            )
            for category, rows in by_category.items()
        },
        "category_metrics": {
            category: {
                "sample_size": len(rows),
                "exact_correct": sum(r["truth"]["defectType"] == r["prediction"]["defectType"] for r in rows),
                "rating_correct": sum(r["truth"]["ratingScale(1-5)"] == r["prediction"]["ratingScale(1-5)"] for r in rows),
                "atomic_macro_f1": round(_macro_atomic_f1([
                    (r["truth"]["defectType"], r["prediction"]["defectType"]) for r in rows
                ]), 6),
                "missed_nonhealthy": sum(r["truth"]["defectType"] != "完好" and r["prediction"]["defectType"] == "完好" for r in rows),
                "false_nonhealthy": sum(r["truth"]["defectType"] == "完好" and r["prediction"]["defectType"] != "完好" for r in rows),
            }
            for category, rows in by_category.items()
        },
        "scene_accuracy": {
            scene: {"support": len(rows), "exact_correct": sum(
                r["truth"]["defectType"] == r["prediction"]["defectType"] for r in rows
            )} for scene, rows in by_scene.items()
        },
        "per_label": {
            label: {
                "support": len(rows),
                "exact_accuracy": round(
                    sum(r["truth"]["defectType"] == r["prediction"]["defectType"] for r in rows)
                    / len(rows),
                    6,
                ),
                "predicted_labels": dict(Counter(r["prediction"]["defectType"] for r in rows)),
            }
            for label, rows in by_label.items()
        },
        "predicted_label_counts": dict(Counter(row["prediction"]["defectType"] for row in records)),
    }


def _variants(names: list[str], base: dict[str, Any]) -> dict[str, dict[str, Any]]:
    variants = {
        "baseline": {"label_scope": "scene", "review_selection": "legacy"},
        "all_labels": {"label_scope": "all", "review_selection": "legacy"},
        "strict_review": {"label_scope": "scene", "review_selection": "strict"},
        "coarse_anchor": {
            "presence_context_policy": "coarse_as_hint",
            "bridge_review_enabled": False,
        },
        "independent": {
            "presence_context_policy": "none",
            "bridge_review_enabled": False,
        },
        "multiview_checklist": {
            "presence_context_policy": "none",
            "bridge_checklist": True,
            "bridge_aerial_crops": True,
            "bridge_aerial_review_crops": True,
            "review_aerial_healthy": True,
            "bridge_review_enabled": True,
            "calibration_review": True,
        },
        "support_review": {
            "bridge_review_enabled": True,
            "bridge_review_scenes": ["support"],
            "calibration_review": True,
        },
        "bottom_review": {
            "bridge_review_enabled": True,
            "bridge_review_scenes": ["bottom"],
            "calibration_review": True,
        },
        "full_label_scope": {
            "label_scope": "all",
            "review_selection": "strict",
        },
        "training_evidence": {
            "label_scope": "scene", "review_selection": "legacy",
            "training_description_guidance": True,
            "track_atomic_checklist": True,
        },
        "bridge_descriptions": {
            "training_description_guidance": True,
            "track_atomic_checklist": False,
        },
        "track_atoms": {
            "training_description_guidance": False,
            "track_atomic_checklist": True,
        },
    }
    if not names or len(set(names)) != len(names) or any(name not in variants for name in names):
        raise ValueError(f"Provide unique known variants from: {', '.join(variants)}")
    return {name: {**base, **variants[name]} for name in names}


def _allow_review(item: dict[str, Any], config: dict[str, Any]) -> bool:
    # Match run_inference, not the older calibration-only review override.
    return item["questionCategory"] != "桥梁" or bool(config.get("bridge_review_enabled", True))


def _input_item(row: dict[str, Any]) -> dict[str, Any]:
    return {"id": row["image"], **{key: row.get(key, "") for key in (
        "image", "questionCategory", "bridgeName", "defectLocation", "filename")}}


def _label_coverage(sample, lexicon, config):
    coverage = {}
    for category in dict.fromkeys(row["questionCategory"] for row in sample):
        rows = [row for row in sample if row["questionCategory"] == category]
        excluded = Counter(row["defectType"] for row in rows if row["defectType"] not in candidate_labels(
            lexicon, category, scene_key(row), include_all=(config.get("label_scope", "scene") == "all"
                or (category == "轨道" and config.get("track_atomic_checklist", False)))))
        coverage[category] = {"sample_size": len(rows), "included_targets": len(rows) - sum(excluded.values()),
                              "excluded_label_counts": dict(excluded)}
    return coverage


def _context(workspace, config, sample, training, lexicon, split, variants):
    model_dir = Path(config["model_dir"])
    if not model_dir.is_absolute():
        model_dir = workspace / model_dir
    model_files = sorted(path for path in model_dir.iterdir() if path.is_file()
                         and path.suffix in {".safetensors", ".json", ".txt", ".jinja", ".model"})
    if not any(path.suffix == ".safetensors" for path in model_files):
        raise ValueError(f"No model weights in {model_dir}")
    sources = [Path(__file__), CODE_DIR / "run_pipeline.py", *[
        CODE_DIR / "track4_agent" / name for name in (
            "data.py", "model.py", "inference.py", "vision_review.py")]]
    dependencies = {}
    for name in ("torch", "transformers", "accelerate", "pillow", "safetensors"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "not installed"
    return {
        "version": 1, "kind": "resumable whole-bridge prompt ablation",
        "config": config, "variants": variants, "variant_order": list(variants),
        "split": split, "sample_sha256": digest_json(sample),
        "training_sha256": digest_json(training), "lexicon_sha256": digest_json(lexicon),
        "sources": {str(path.relative_to(CODE_DIR)): sha256_file(path) for path in sources},
        "model_files": {path.name: sha256_file(path) for path in model_files},
        "images": {row["image"]: sha256_file(workspace / "data" / "raw" / row["image"])
                   for row in sample},
        "runtime": {"python": platform.python_version(), "system": platform.system(),
                    "dependencies": dependencies},
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    # Replace only a complete, flushed snapshot; an interrupted write keeps the previous checkpoint.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@contextmanager
def _report_lock(output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(output.suffix + ".lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            lock = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            lock = lambda: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(handle, fcntl.LOCK_UN)
        try:
            lock()
        except OSError as exc:
            raise ValueError("Another evaluation is writing this output") from exc
        try:
            yield
        finally:
            handle.seek(0)
            unlock()


class _ReplayClient:
    def __init__(self, records):
        if not isinstance(records, list) or not records:
            raise ValueError("Checkpoint lacks raw model responses")
        self.responses = iter(row["raw"] for row in records)

    def generate(self, images, prompt):
        try:
            response = next(self.responses)
        except StopIteration as exc:
            raise ValueError("Checkpoint lacks a required model round") from exc
        if not isinstance(response, str):
            raise ValueError("Raw response must be text")
        return response


def _validate_record(record, truth, lexicon, config, raw_dir):
    if record.get("truth") != truth or set(record.get("prediction", {})) != set(OUTPUT_KEYS):
        raise ValueError("Checkpoint truth or prediction schema differs")
    item = _input_item(truth)
    replay = _ReplayClient(record.get("raw"))
    final, raw = infer_item(replay, item, lexicon, config, raw_dir / item["image"],
                            raw_dir=raw_dir, allow_review=_allow_review(item, config))
    if final != record["prediction"] or raw != record["raw"] or next(replay.responses, None) is not None:
        raise ValueError(f"Checkpoint raw responses and prediction disagree: {item['id']}")


def _raw_source(report, variants, name, index):
    options = {key: value for key, value in variants[name].items() if key != "review_selection"}
    for source, config in variants.items():
        if source == name:
            break
        source_options = {key: value for key, value in config.items() if key != "review_selection"}
        records = report["results"][source]["records"]
        if options == source_options and index < len(records):
            return source, records[index]
    return None


def _paired_comparison(base, candidate):
    base_by_id = {row["truth"]["image"]: row for row in base}
    paired = [(base_by_id[row["truth"]["image"]], row) for row in candidate
              if row["truth"]["image"] in base_by_id]
    changes = []
    for before, after in paired:
        if before["truth"] != after["truth"]:
            raise ValueError("Paired evaluation truths differ")
        changes.append({"image": after["truth"]["image"],
                        "category": after["truth"]["questionCategory"],
                        "label_changed": before["prediction"]["defectType"] != after["prediction"]["defectType"],
                        "gained": before["prediction"]["defectType"] != after["truth"]["defectType"]
                                  and after["prediction"]["defectType"] == after["truth"]["defectType"],
                        "lost": before["prediction"]["defectType"] == after["truth"]["defectType"]
                                and after["prediction"]["defectType"] != after["truth"]["defectType"]})
    return {"paired_sample_size": len(paired),
            "gained_correct": sum(row["gained"] for row in changes),
            "lost_correct": sum(row["lost"] for row in changes),
            "label_changes": sum(row["label_changed"] for row in changes),
            "baseline_metrics": _metrics([before for before, _ in paired]),
            "candidate_metrics": _metrics([after for _, after in paired]),
            "changes": [row for row in changes if row["label_changed"]]}


def _refresh_report(report, sample_size):
    for result in report["results"].values():
        result["metrics"] = _metrics(result["records"])
        result["raw_replay_images"] = sum(row.get("raw_source_variant") is not None for row in result["records"])
        result["model_inferred_images"] = len(result["records"]) - result["raw_replay_images"]
        result["status"] = "complete" if len(result["records"]) == sample_size else "pending"
    report["status"] = "complete" if all(
        result["status"] == "complete" for result in report["results"].values()) else "prepared"
    anchor = report["variant_order"][0]
    report["comparison_anchor"] = anchor
    report["comparisons"] = {
        name: _paired_comparison(report["results"][anchor]["records"], result["records"])
        for name, result in report["results"].items() if name != anchor
    }


def _gpu_preflight(min_free_mib: int):
    if min_free_mib <= 0:
        raise ValueError("Minimum free GPU memory must be positive")
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=10)
    values = next(csv.reader(result.stdout.splitlines()))
    free, total, utilization = map(lambda value: int(value.strip()), values)
    if free < min_free_mib:
        raise RuntimeError(f"GPU is occupied: {free}/{total} MiB free, utilization {utilization}%; "
                           f"need {min_free_mib} MiB free. Checkpoint preserved; release GPU and use --resume.")


def _required_gpu_memory(config, minimum=None):
    from track4_agent.model import ModelClient
    client = ModelClient.__new__(ModelClient)
    client.gpu_max_memory_mib = config.get("gpu_max_memory_mib")
    client.gpu_reserve_mib = config.get("gpu_reserve_mib", 1536)
    client.memory_plan()
    required = (client.gpu_max_memory_mib + client.gpu_reserve_mib
                if client.gpu_max_memory_mib is not None else 10240)
    if minimum is not None:
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum <= 0:
            raise ValueError("Minimum free GPU memory must be a positive integer")
        required = max(required, minimum)
    return required


def run_holdout(workspace, config, sample, training, lexicon, split, variants, output,
                *, resume=False, prepare_only=False, max_images=None, client_factory=None):
    output = candidate_path(workspace, output)
    if max_images is not None and max_images <= 0:
        raise ValueError("Max images must be positive")
    with _report_lock(output):
        if output.exists() and not resume:
            raise ValueError("Output already exists; use --resume or a new --output")
        context = _context(workspace, config, sample, training, lexicon, split, variants)
        raw_dir = workspace / "data" / "raw"
        report = {"context": context, "model": config["model_id"], "split": split,
                  "variant_order": list(variants), "results": {
                      name: {"elapsed_seconds": 0.0, "records": [],
                             "prompt_label_coverage": _label_coverage(sample, lexicon, variant)}
                      for name, variant in variants.items()}}
        if output.exists():
            report = json.loads(output.read_text(encoding="utf-8"))
            if (report.get("context") != context or report.get("split") != split
                    or report.get("variant_order") != list(variants)
                    or set(report.get("results", {})) != set(variants)):
                raise ValueError("Resume context differs; use a new output path")
            for name, result in report["results"].items():
                result["prompt_label_coverage"] = _label_coverage(sample, lexicon, variants[name])
                elapsed = result.get("elapsed_seconds")
                if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                    raise ValueError("Invalid checkpoint elapsed time")
                if not isinstance(result.get("records"), list) or len(result["records"]) > len(sample):
                    raise ValueError("Invalid checkpoint coverage")
                for index, record in enumerate(result["records"]):
                    _validate_record(record, sample[index], lexicon, variants[name], raw_dir)
                    if record.get("raw_source_variant") is not None:
                        source = _raw_source(report, variants, name, index)
                        if (source is None or source[0] != record["raw_source_variant"]
                                or source[1]["truth"] != record["truth"] or source[1]["raw"] != record["raw"]):
                            raise ValueError("Checkpoint raw replay source differs")
        _refresh_report(report, len(sample))
        _write_report(output, report)
        print(f"[prepared] samples={len(sample)} variants={len(variants)} status={report['status']}", flush=True)
        if prepare_only or report["status"] == "complete":
            return report
        client = None
        processed = 0
        for name, variant in variants.items():
            result = report["results"][name]
            for index in range(len(result["records"]), len(sample)):
                if max_images is not None and processed >= max_images:
                    report["status"] = "paused"
                    _write_report(output, report)
                    return report
                truth = sample[index]
                item = _input_item(truth)
                started = time.perf_counter()
                # review_selection only chooses between already generated rounds;
                # use identical raw responses for a paired postprocessing ablation.
                source = _raw_source(report, variants, name, index)
                if source is None and client is None:
                    client = (client_factory or (lambda: load_client(config)))()
                inference_client = _ReplayClient(source[1]["raw"]) if source is not None else client
                prediction, raw = infer_item(inference_client, item, lexicon, variant, raw_dir / item["image"],
                                             raw_dir=raw_dir, allow_review=_allow_review(item, variant))
                record = {"truth": truth, "prediction": prediction, "raw": raw,
                          "raw_source_variant": source[0] if source is not None else None}
                _validate_record(record, truth, lexicon, variant, raw_dir)
                result["records"].append(record)
                result["elapsed_seconds"] = round(result["elapsed_seconds"] + time.perf_counter() - started, 3)
                processed += 1
                _refresh_report(report, len(sample))
                if report["status"] != "complete":
                    report["status"] = "running"
                _write_report(output, report)
                method = f"raw replay from {source[0]}" if source is not None else "model inference"
                print(f"[{name}] {index + 1}/{len(sample)} saved ({method})", flush=True)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="按整桥留出的4B提示对照评测")
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--bridge-group", default="范家坪1号大桥")
    parser.add_argument("--variants", default="baseline,all_labels,strict_review,full_label_scope")
    parser.add_argument("--config", type=Path, default=CODE_DIR / "config" / "qwen3-vl-4b-review.json")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "runs" / "prompt_ablation_v1" / "holdout.json")
    parser.add_argument("--prepare-only", action="store_true", help="Validate inputs and prepare checkpoints without loading GPU model")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-images", type=int, help="Pause after this many newly evaluated images, including raw-response replay")
    parser.add_argument("--min-free-gpu-mib", type=int, default=None, help="Additional GPU free-memory floor; capped configs also require weight budget plus reserve")
    args = parser.parse_args()

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["MODELSCOPE_OFFLINE"] = "1"

    train, _, full_lexicon = load_assets(WORKSPACE)
    config = load_config(args.config.resolve())
    sample, training, split_report = _make_holdout(
        train, args.sample_size, int(config["seed"]), args.bridge_group
    )
    lexicon = _holdout_lexicon(training, full_lexicon)
    names = [name.strip() for name in args.variants.split(",") if name.strip()]
    variants = _variants(names, config)

    def client_factory():
        _gpu_preflight(_required_gpu_memory(config, args.min_free_gpu_mib))
        return load_client(config)

    report = run_holdout(WORKSPACE, config, sample, training, lexicon, split_report,
                         variants, args.output, resume=args.resume, prepare_only=args.prepare_only,
                         max_images=args.max_images, client_factory=client_factory)

    print(json.dumps({
        "status": report["status"], "split": split_report,
        "variants": {name: {"status": value["status"], "model_inferred_images": value["model_inferred_images"],
                            "raw_replay_images": value["raw_replay_images"],
                            "prompt_label_coverage": value["prompt_label_coverage"],
                            "metrics": {key: metric for key, metric in value["metrics"].items()
                                        if key in {"sample_size", "defect_type_exact_accuracy", "rating_accuracy",
                                                   "atomic_macro_f1", "category_accuracy"}}}
                     for name, value in report["results"].items()},
        "comparisons": {name: {key: value for key, value in comparison.items()
                               if key in {"paired_sample_size", "gained_correct", "lost_correct", "label_changes"}}
                        for name, comparison in report["comparisons"].items()},
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
