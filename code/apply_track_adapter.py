"""Create an isolated track-only candidate from two independently trained adapters.

The official result is read-only. A row changes only when both independently
trained visual adapters select the same legal track label above the confidence
threshold, and the candidate does not replace a baseline healthy label; bridge
rows and all other fields remain from the model-generated baseline. The audit
records every automatic decision.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from run_pipeline import load_config
from track4_agent.data import OUTPUT_KEYS, load_assets, write_json
from track4_agent.validation import validate_result
from track4_agent.vision_classifier import (
    decode_atomic_with_confidence, model_identity,
)
from track4_agent.vision_review import candidate_path, digest_json, index_unique, sha256_file


def _load_report(path: Path) -> tuple[dict, dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("selected_epoch") is None:
        raise ValueError(f"Adapter report is incomplete: {path}")
    state_path = path.with_suffix(".pt")
    if report.get("adapter_sha256") != sha256_file(state_path):
        raise ValueError(f"Adapter hash differs from report: {path}")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if state.get("model_identity") != report.get("model_identity"):
        raise ValueError(f"Adapter model identity differs: {path}")
    if state.get("atoms") != report.get("atoms"):
        raise ValueError(f"Adapter vocabulary differs: {path}")
    if not isinstance(state.get("mean"), torch.Tensor) or not isinstance(state.get("scale"), torch.Tensor):
        raise ValueError(f"Adapter normalization is missing: {path}")
    return report, state


def _validate_base(results, manifest, lexicon):
    by_name = index_unique(manifest, "filename")
    if len(results) != len(manifest):
        raise ValueError("Base result count differs from test manifest")
    for result in results:
        if list(result) != OUTPUT_KEYS:
            raise ValueError(f"Base result schema differs for {result.get('filename')}")
        item = by_name.get(result["filename"])
        if item is None:
            raise ValueError(f"Base result has unknown filename: {result['filename']}")
        for key in ("questionCategory", "bridgeName", "defectLocation", "filename"):
            if result[key] != item.get(key, ""):
                raise ValueError(f"Base result metadata differs for {result['filename']}: {key}")
        allowed = lexicon["categories"][result["questionCategory"]]["allowed_labels"]
        if result["defectType"] not in allowed or result["ratingScale(1-5)"] not in {"", "1", "2", "3", "4", "5"}:
            raise ValueError(f"Base result has an illegal label or rating: {result['filename']}")


def _state_model(base_model, state, device):
    vision = base_model.vision_model.to(device).eval()
    unknown = set(state["vision"]) - set(dict(vision.named_parameters()))
    if unknown:
        raise ValueError(f"Unknown adapter parameters: {sorted(unknown)}")
    vision.load_state_dict(state["vision"], strict=False)
    classifier = torch.nn.Linear(len(state["mean"]), len(state["atoms"])).to(device).eval()
    classifier.load_state_dict(state["classifier"])
    return vision, classifier, state["mean"].to(device), state["scale"].to(device)


def _predict_adapter(base_model, processor, rows, state, allowed, batch_size, device):
    vision, classifier, mean, scale = _state_model(base_model, state, device)
    predictions = {}
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                images = []
                for row in batch:
                    with Image.open(ROOT / "data" / "raw" / row["image"]) as source:
                        images.append(ImageOps.exif_transpose(source).convert("RGB"))
                pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
                embedding = torch.nn.functional.normalize(vision(pixel_values=pixels).pooler_output.float(), dim=-1)
                probabilities = classifier((embedding - mean) / scale).sigmoid().cpu().numpy()
                labels, confidence = decode_atomic_with_confidence(probabilities, state["atoms"], allowed)
                for row, label, score in zip(batch, labels, confidence, strict=True):
                    predictions[row["filename"]] = {"label": label, "confidence": float(score)}
    finally:
        del vision, classifier, mean, scale
        torch.cuda.empty_cache()
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-result", type=Path, default=ROOT / "result" / "result.json")
    parser.add_argument("--adapter", type=Path, action="append", required=True,
                        help="Exactly two completed track_adapter report.json paths")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if len(args.adapter) != 2:
        parser.error("Provide exactly two --adapter report paths")
    if not 0 < args.threshold <= 1 or args.batch_size < 1:
        parser.error("Threshold must be in (0, 1] and batch size must be positive")
    output = candidate_path(ROOT, args.output)
    audit_path = candidate_path(ROOT, args.audit)
    if output.exists() or audit_path.exists():
        raise ValueError("Candidate output already exists; choose a new runs directory")
    for name in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "MODELSCOPE_OFFLINE"):
        os.environ[name] = "1"
    train, manifest, lexicon = load_assets(ROOT)
    base_result = json.loads(args.base_result.read_text(encoding="utf-8"))
    _validate_base(base_result, manifest, lexicon)
    reports_states = [_load_report(path.resolve()) for path in args.adapter]
    reports = [item[0] for item in reports_states]
    states = [item[1] for item in reports_states]
    if reports[0]["model_identity"] != reports[1]["model_identity"]:
        raise ValueError("Adapters use different base model files")
    if states[0]["atoms"] != states[1]["atoms"]:
        raise ValueError("Adapters use different atom vocabularies")
    model_path = ROOT / "models" / "SigLIP2-base-patch16-224"
    if model_identity(model_path) != reports[0]["model_identity"]:
        raise ValueError("Base visual model files changed")
    track_rows = [row for row in manifest if row["questionCategory"] == "轨道"]
    allowed = lexicon["categories"]["轨道"]["allowed_labels"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA GPU is required to create the adapter candidate")
    torch.set_num_threads(4)
    from transformers import AutoModel, AutoProcessor
    base_model = AutoModel.from_pretrained(model_path, dtype=torch.float32,
                                            local_files_only=True, attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    predictions = [_predict_adapter(base_model, processor, track_rows, state, allowed, args.batch_size, device)
                   for state in states]
    del base_model, processor
    torch.cuda.empty_cache()
    base_by_name = index_unique(base_result, "filename")
    candidate = []
    audit = []
    changed = 0
    for row in base_result:
        final = dict(row)
        decision = {"filename": row["filename"], "questionCategory": row["questionCategory"],
                    "original_label": row["defectType"], "final_label": row["defectType"],
                    "accepted": False, "reason": "bridge row unchanged"}
        if row["questionCategory"] == "轨道":
            first, second = (prediction[row["filename"]] for prediction in predictions)
            decision.update({"adapter_labels": [first["label"], second["label"]],
                             "adapter_confidences": [first["confidence"], second["confidence"]]})
            if (first["label"] == second["label"] and first["confidence"] >= args.threshold
                    and second["confidence"] >= args.threshold and row["defectType"] != "完好"):
                final["defectType"] = first["label"]
                decision.update({"final_label": first["label"], "accepted": first["label"] != row["defectType"],
                                 "reason": "two independent adapters agree above threshold; baseline healthy protected"})
            else:
                decision["reason"] = ("baseline healthy protected"
                                       if row["defectType"] == "完好"
                                       else "adapters disagree or confidence is below threshold")
        if final["defectType"] != row["defectType"]:
            changed += 1
        candidate.append(final)
        audit.append(decision)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, candidate)
    write_json(audit_path, {
        "policy": "two independently trained track adapters agree above threshold; baseline healthy protected",
        "threshold": args.threshold, "adapter_reports": [str(path.resolve()) for path in args.adapter],
        "adapter_report_sha256": [sha256_file(path.resolve()) for path in args.adapter],
        "base_result_sha256": sha256_file(args.base_result.resolve()), "training_manifest_sha256": digest_json(train),
        "changed_count": changed, "track_count": len(track_rows), "bridge_unchanged": True,
        "records": audit,
    })
    validate_result(ROOT, manifest, lexicon, result_path=output,
                    report_path=audit_path.with_name("validation_report.json"))
    print(json.dumps({"output": str(output), "audit": str(audit_path), "changed_count": changed,
                      "track_count": len(track_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
