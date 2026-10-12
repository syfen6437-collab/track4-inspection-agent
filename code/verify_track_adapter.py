"""Reload a saved visual adapter and reproduce held-out labels without training."""
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

from evaluate_holdout import _gpu_preflight, _report_lock, _write_report
from track4_agent.data import load_assets
from track4_agent.vision_classifier import decode_atomic, model_identity
from track4_agent.vision_review import candidate_path, digest_json, index_unique, sha256_file


def check_snapshot(state, report):
    if state.get("model_identity") != report["model_identity"] or state.get("selected_epoch") != report["selected_epoch"]:
        raise ValueError("Adapter model or selected epoch differs from report")
    if state.get("atoms") != report["atoms"]:
        raise ValueError("Adapter target vocabulary differs")
    for section in ("vision", "classifier"):
        if not isinstance(state.get(section), dict) or not state[section]:
            raise ValueError("Missing adapter tensor section")
        if not all(isinstance(v, torch.Tensor) and torch.isfinite(v).all() for v in state[section].values()):
            raise ValueError("Adapter contains nonfinite or non-tensor parameters")
    for name in ("mean", "scale"):
        if not isinstance(state.get(name), torch.Tensor) or state[name].ndim != 1 or not torch.isfinite(state[name]).all():
            raise ValueError("Invalid feature normalization")
    if state["mean"].shape != state["scale"].shape or not (state["scale"] > 0).all():
        raise ValueError("Invalid feature scaling")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("Sample size must be positive")
    report_path = candidate_path(ROOT, args.report)
    output = candidate_path(ROOT, args.output)
    for name in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "MODELSCOPE_OFFLINE"):
        os.environ[name] = "1"
    torch.set_num_threads(4)
    with _report_lock(output):
        if output.exists():
            raise ValueError("Output exists; choose a new verification path")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "complete":
            raise ValueError("Training report is incomplete")
        state_path = report_path.with_suffix(".pt")
        if sha256_file(state_path) != report["adapter_sha256"]:
            raise ValueError("Saved adapter hash differs")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        check_snapshot(state, report)
        train, _, lexicon = load_assets(ROOT)
        if digest_json(train) != report["training_manifest_sha256"]:
            raise ValueError("Training manifest changed")
        by_image = index_unique(train, "image")
        records = report["test"]["candidate"]["records"][:args.sample_size]
        audit_path = candidate_path(ROOT, Path(report["config"]["audit"]))
        if sha256_file(audit_path) != report["audit_sha256"]:
            raise ValueError("Training content audit changed")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        image_records = index_unique(audit["image_records"], "image")
        for row in records:
            image_id = row["truth"]["image"]
            if sha256_file(ROOT / "data" / "raw" / image_id) != image_records[image_id]["file_sha256"]:
                raise ValueError("Verification image changed from training experiment")
        model_path = ROOT / "models" / "SigLIP2-base-patch16-224"
        if model_identity(model_path) != state["model_identity"]:
            raise ValueError("Base model files changed")
        _gpu_preflight(3072)
        from transformers import AutoModel, AutoProcessor
        base = AutoModel.from_pretrained(model_path, dtype=torch.float32, local_files_only=True, attn_implementation="sdpa")
        vision = base.vision_model.to("cuda").eval()
        del base
        unknown = set(state["vision"]) - set(dict(vision.named_parameters()))
        if unknown:
            raise ValueError(f"Unknown adapter parameters: {sorted(unknown)}")
        vision.load_state_dict(state["vision"], strict=False)
        classifier = torch.nn.Linear(len(state["mean"]), len(state["atoms"])).to("cuda").eval()
        classifier.load_state_dict(state["classifier"])
        mean, scale = state["mean"].to("cuda"), state["scale"].to("cuda")
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        results = []
        batch_size = int(report["config"]["batch_size"])
        with torch.inference_mode():
            for offset in range(0, len(records), batch_size):
                batch = records[offset:offset + batch_size]
                images = []
                for row in batch:
                    item = by_image[row["truth"]["image"]]
                    if item != row["truth"] or item["questionCategory"] != "轨道":
                        raise ValueError("Verification sample differs from training report")
                    with Image.open(ROOT / "data" / "raw" / item["image"]) as source:
                        images.append(ImageOps.exif_transpose(source).convert("RGB"))
                pixels = processor(images=images, return_tensors="pt")["pixel_values"].to("cuda")
                embedding = torch.nn.functional.normalize(vision(pixel_values=pixels).pooler_output.float(), dim=-1)
                probabilities = classifier((embedding - mean) / scale).sigmoid().cpu().numpy()
                labels = decode_atomic(probabilities, state["atoms"], lexicon["categories"]["轨道"]["allowed_labels"])
                for row, label in zip(batch, labels, strict=True):
                    results.append({"image": row["truth"]["image"], "label": label,
                                    "matches_saved_prediction": label == row["prediction"]["defectType"]})
        verification = {"status": "complete", "report_sha256": sha256_file(report_path),
                        "adapter_sha256": report["adapter_sha256"], "sample_count": len(results),
                        "all_labels_reproduced": all(row["matches_saved_prediction"] for row in results),
                        "records": results, "implementation_sha256": sha256_file(Path(__file__))}
        _write_report(output, verification)
        print(json.dumps({key: value for key, value in verification.items() if key != "records"}), flush=True)
        if not verification["all_labels_reproduced"]:
            raise RuntimeError("Reloaded adapter predictions differ; do not promote")


if __name__ == "__main__":
    main()
