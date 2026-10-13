"""Independently review isolated track changes with local Qwen and per-image checkpoints."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from apply_track_adapter import _validate_base
from evaluate_holdout import _gpu_preflight, _report_lock, _required_gpu_memory, _write_report
from run_pipeline import load_client, load_config
from track4_agent.data import OUTPUT_KEYS, load_assets
from track4_agent.track_consensus import checked_review, merge_track_reviews
from track4_agent.validation import validate_result
from track4_agent.vision_review import (
    candidate_path, digest_json, index_unique, probability, review_context, review_image, sha256_file,
)

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent


def _load_rows(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Result must be a JSON array: {path}")
    return value


def _validate_candidate(base, candidate, manifest, lexicon):
    _validate_base(base, manifest, lexicon)
    _validate_base(candidate, manifest, lexicon)
    for before, after in zip(base, candidate, strict=True):
        if before["filename"] != after["filename"]:
            raise ValueError("Candidate row order or filename differs")
        if before["questionCategory"] == "桥梁" and before != after:
            raise ValueError(f"Bridge row changed in track candidate: {before['filename']}")
        if any(before[key] != after[key] for key in OUTPUT_KEYS if key != "defectType"):
            raise ValueError("Adapter candidate may change only the label")


def _checked_adapter_audit(audit, base, candidate, base_hash, train_hash):
    if audit.get("base_result_sha256") != base_hash:
        raise ValueError("Adapter audit belongs to a different baseline")
    if audit.get("training_manifest_sha256") != train_hash:
        raise ValueError("Adapter training manifest differs")
    decisions = index_unique(audit.get("records", []), "filename")
    if set(decisions) != {row["filename"] for row in base}:
        raise ValueError("Adapter audit coverage differs")
    threshold = probability(audit.get("threshold"))
    for before, after in zip(base, candidate, strict=True):
        record = decisions[before["filename"]]
        if (record.get("original_label") != before["defectType"]
                or record.get("final_label") != after["defectType"]
                or record.get("questionCategory") != before["questionCategory"]):
            raise ValueError("Adapter audit and result disagree")
        if before["questionCategory"] != "轨道" or before == after:
            continue
        # Legacy intermediate candidates may contain healthy proposals, rejected downstream.
        labels, scores = record.get("adapter_labels", []), record.get("adapter_confidences", [])
        if len(labels) != 2 or len(scores) != 2:
            raise ValueError("Adapter consensus evidence is missing")
        if labels != [after["defectType"]] * 2 or min(probability(s) for s in scores) < threshold:
            raise ValueError("Adapter consensus evidence disagrees")
        if before["defectType"] == "完好":
            raise ValueError("Baseline healthy boundary was changed")
    return decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CODE / "config" / "qwen3-vl-4b-offload.json")
    parser.add_argument("--base-result", type=Path, default=ROOT / "result" / "result.json")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--adapter-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--revalidate-only", action="store_true")
    args = parser.parse_args()
    probability(args.min_confidence)
    if args.threads <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("Threads and limit must be positive")
    output, audit_path = candidate_path(ROOT, args.output), candidate_path(ROOT, args.audit)
    checkpoint = audit_path.with_name(audit_path.stem + ".checkpoint.json")
    validation = audit_path.with_name("validation_report.json")
    inputs = {p.resolve() for p in (args.base_result, args.candidate, args.adapter_audit, args.config)}
    if len({output, audit_path, checkpoint, validation}) != 4 or inputs & {output, audit_path, checkpoint, validation}:
        raise ValueError("Input and output paths overlap")
    for name in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "MODELSCOPE_OFFLINE"):
        os.environ[name] = "1"
    train, manifest, lexicon = load_assets(ROOT)
    base, candidate = _load_rows(args.base_result), _load_rows(args.candidate)
    _validate_candidate(base, candidate, manifest, lexicon)
    adapter_audit = json.loads(args.adapter_audit.read_text(encoding="utf-8"))
    decisions = _checked_adapter_audit(adapter_audit, base, candidate, sha256_file(args.base_result), digest_json(train))
    bases, candidates = index_unique(base, "filename"), index_unique(candidate, "filename")
    selected = [item for item in manifest if item["questionCategory"] == "轨道"
                and bases[item["filename"]]["defectType"] != "完好"
                and candidates[item["filename"]]["defectType"] not in ("完好", bases[item["filename"]]["defectType"])]
    if args.limit is not None:
        selected = selected[:args.limit]
    config = load_config(args.config.resolve())
    context = review_context(ROOT, config, selected, base, candidate, lexicon,
                             min_confidence=args.min_confidence, scenes=())
    context.update(adapter_audit_sha256=sha256_file(args.adapter_audit), threads=args.threads,
                   implementation={p.name: sha256_file(p) for p in
                                   (Path(__file__), CODE / "track4_agent" / "track_consensus.py")})
    with _report_lock(checkpoint):
        if checkpoint.exists():
            if not args.resume:
                raise ValueError("Checkpoint exists; pass --resume")
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            if state.get("context") != context:
                raise ValueError("Checkpoint input/config/model/code changed; choose a new directory")
        else:
            if args.revalidate_only or output.exists() or audit_path.exists():
                raise ValueError("Checkpoint missing or untracked output exists; choose a new directory")
            state = {"context": context, "status": "running", "records": []}
        records = state["records"]
        index_unique(records, "filename")
        if [r["filename"] for r in records] != [i["filename"] for i in selected[:len(records)]]:
            raise ValueError("Checkpoint reviews are stale, duplicated or reordered")
        for record, item in zip(records, selected):
            checked_review(record, item, candidates[item["filename"]], lexicon)
        if args.revalidate_only and len(records) != len(selected):
            raise ValueError("Revalidation requires complete checkpoints; no model calls allowed")
        _write_report(checkpoint, state)
        if len(records) < len(selected):
            import torch
            torch.set_num_threads(args.threads)
            _gpu_preflight(_required_gpu_memory(config))
            client = load_client(config)
            for item in selected[len(records):]:
                row = candidates[item["filename"]]
                scores = decisions[item["filename"]]["adapter_confidences"]
                vision = {"defectType": row["defectType"], "confidence": min(scores)}
                review = review_image(client, item, bases[item["filename"]], vision, lexicon, config,
                                      ROOT / "data" / "raw" / item["image"], ROOT / "data" / "raw")
                records.append(review)
                _write_report(checkpoint, state)
                print(f"[qwen-review] {len(records)}/{len(selected)} valid={review['valid']}", flush=True)
        final, audit_rows = merge_track_reviews(base, candidate, manifest, records, lexicon, args.min_confidence)
        accepted = sum(row["accepted"] for row in audit_rows)
        _write_report(output, final)
        _write_report(audit_path, {"policy": "two-adapter consensus plus independent Qwen agreement/confidence",
                                 "context_sha256": digest_json(context), "reviewed_count": len(records),
                                 "accepted_count": accepted, "bridge_unchanged": all(a == b for a, b in zip(base, final)
                                     if a["questionCategory"] == "桥梁"), "records": audit_rows})
        report = validate_result(ROOT, manifest, lexicon, result_path=output, report_path=validation)
        if not report["valid"]:
            raise ValueError("Candidate failed validation")
        state["status"] = "complete"
        _write_report(checkpoint, state)
        print(json.dumps({"output": str(output), "reviewed_count": len(records), "accepted_count": accepted}, ensure_ascii=False))


if __name__ == "__main__":
    main()
