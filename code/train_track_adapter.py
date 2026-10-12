"""Isolated supervised vision fine-tuning; never produces or changes official test answers."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from compare_visual_views import check_content_audit, checked_bank, label_metrics, natural_splits
from evaluate_holdout import _gpu_preflight, _paired_comparison, _report_lock, _write_report
from track4_agent.data import load_assets
from track4_agent.vision_classifier import atomic_parts, decode_atomic, fit_head, model_identity, predict_head
from track4_agent.vision_review import candidate_path, digest_json, sha256_file


def training_partition(rows, validation_size, seed):
    ordered = sorted(rows, key=lambda row: row["image"])
    if not 0 < validation_size < len(ordered):
        raise ValueError("Validation size must leave fitting images")
    random.Random(seed).shuffle(ordered)
    return ordered[validation_size:], ordered[:validation_size]


def trainable_vision_parameters(vision, last_blocks):
    layers = vision.encoder.layers
    if not 0 < last_blocks <= len(layers):
        raise ValueError("Invalid number of trainable vision blocks")
    vision.requires_grad_(False)
    for layer in layers[-last_blocks:]:
        layer.requires_grad_(True)
    vision.post_layernorm.requires_grad_(True)
    vision.head.requires_grad_(True)
    return {name for name, parameter in vision.named_parameters() if parameter.requires_grad}


def selected_state(vision, names, classifier):
    return {"vision": {name: value.detach().cpu().clone() for name, value in vision.state_dict().items() if name in names},
            "classifier": {name: value.detach().cpu().clone() for name, value in classifier.state_dict().items()}}


def restore_state(vision, classifier, state):
    vision.load_state_dict(state["vision"], strict=False)
    classifier.load_state_dict(state["classifier"])


def class_weights(targets):
    if targets.ndim != 2 or not len(targets):
        raise ValueError("Expected nonempty multilabel targets")
    positive = targets.sum(0)
    return ((len(targets) - positive) / positive.clamp_min(1)).sqrt().clamp(0.5, 3.0)


def records_for(rows, predictions):
    return [{"truth": row, "prediction": {"questionCategory": "轨道", "defectType": label,
             "ratingScale(1-5)": ""}} for row, label in zip(rows, predictions, strict=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-cache", type=Path, default=ROOT / "models" / "siglip2_train_features_cuda_fp32.npz")
    parser.add_argument("--audit", type=Path, default=ROOT / "runs" / "training_audit_20261012" / "report.json")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--test-size", type=int, default=240)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--last-blocks", type=int, default=2)
    parser.add_argument("--vision-lr", type=float, default=0.00001)
    parser.add_argument("--head-lr", type=float, default=0.0002)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or any(not math.isfinite(x) or x <= 0 for x in (args.vision_lr, args.head_lr)):
        parser.error("Epochs, batch size and learning rates must be positive and finite")
    output = candidate_path(ROOT, args.output)
    for name in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "MODELSCOPE_OFFLINE"):
        os.environ[name] = "1"
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    started = time.perf_counter()
    with _report_lock(output):
        if output.exists():
            raise ValueError("Output exists; use a new experiment directory")
        train, _, lexicon = load_assets(ROOT)
        fitting, testing = natural_splits(train, [], args.test_size, args.seed)["轨道/random"]
        fitting, validation = training_partition(fitting, args.validation_size, args.seed + 1)
        audit = json.loads(args.audit.read_text(encoding="utf-8"))
        check_content_audit(ROOT, train, {"outer": (fitting + validation, testing), "inner": (fitting, validation)}, audit)
        bank, metadata = checked_bank(ROOT, train, args.global_cache, "global")
        model_path = ROOT / "models" / "SigLIP2-base-patch16-224"
        if metadata.get("model") != model_identity(model_path) or metadata.get("device") != "cuda" or metadata.get("dtype") != "float32":
            raise ValueError("Expected the current CUDA FP32 global feature cache")
        positions = {row["image"]: i for i, row in enumerate(train)}
        features = bank[[positions[row["image"]] for row in fitting]]
        head = fit_head(features, [row["defectType"] for row in fitting], multilabel=True, seed=args.seed)
        atoms = head["labels"]
        allowed = lexicon["categories"]["轨道"]["allowed_labels"]
        report = {"kind": "training-only track vision last-block fine-tuning with nested validation",
                  "status": "prepared", "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "model_id": "google/siglip2-base-patch16-224", "precision": "float32",
                  "training_manifest_sha256": digest_json(train), "feature_cache_sha256": sha256_file(args.global_cache),
                  "model_identity": metadata["model"], "audit_sha256": sha256_file(args.audit),
                  "implementation_sha256": {str(p.relative_to(CODE)): sha256_file(p) for p in (
                      Path(__file__), CODE / "track4_agent" / "vision_classifier.py", CODE / "compare_visual_views.py")},
                  "split": {"fitting": [r["image"] for r in fitting], "validation": [r["image"] for r in validation],
                            "test": [r["image"] for r in testing]}, "atoms": atoms, "epochs": []}
        _write_report(output, report)
        _gpu_preflight(3072)
        from transformers import AutoModel, AutoProcessor
        base = AutoModel.from_pretrained(model_path, dtype=torch.float32, local_files_only=True, attn_implementation="sdpa")
        vision = base.vision_model.to("cuda").eval()
        del base
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        names = trainable_vision_parameters(vision, args.last_blocks)
        classifier = torch.nn.Linear(len(head["mean"]), len(atoms)).to("cuda")
        with torch.no_grad():
            classifier.weight.copy_(torch.from_numpy(head["weight"]).to("cuda"))
            classifier.bias.copy_(torch.from_numpy(head["bias"]).to("cuda"))
        mean = torch.from_numpy(head["mean"]).to("cuda")
        scale = torch.from_numpy(head["scale"]).to("cuda")
        all_rows = fitting + validation + testing
        pixels = []
        for index, row in enumerate(all_rows):
            with Image.open(ROOT / "data" / "raw" / row["image"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            pixels.append(processor(images=[image], return_tensors="pt")["pixel_values"][0])
            if (index + 1) % 100 == 0:
                print(f"[pixels] {index + 1}/{len(all_rows)}", flush=True)
        pixels = torch.stack(pixels)
        targets = torch.tensor([[atom in atomic_parts(row["defectType"]) for atom in atoms] for row in fitting], dtype=torch.float32)
        weights = class_weights(targets).to("cuda")
        train_count, val_count = len(fitting), len(validation)

        def logits(indices):
            embedding = torch.nn.functional.normalize(vision(pixel_values=pixels[indices].to("cuda")).pooler_output.float(), dim=-1)
            return classifier((embedding - mean) / scale)

        def predictions(start, size):
            vision.eval()
            classifier.eval()
            outputs = []
            with torch.inference_mode():
                for offset in range(start, start + size, args.batch_size):
                    ids = list(range(offset, min(offset + args.batch_size, start + size)))
                    outputs.append(logits(ids).sigmoid().cpu().numpy())
            return decode_atomic(np.concatenate(outputs), atoms, allowed)

        baseline_validation = records_for(validation, predictions(train_count, val_count))
        frozen_test_labels = predictions(train_count + val_count, len(testing))
        best_score = label_metrics(baseline_validation)["atomic_macro_f1"]
        best_epoch = 0
        best = selected_state(vision, names, classifier)
        report["baseline_validation"] = {"records": baseline_validation, "metrics": label_metrics(baseline_validation)}
        report["trainable_parameters"] = sum(p.numel() for p in vision.parameters() if p.requires_grad) + sum(p.numel() for p in classifier.parameters())
        optimizer = torch.optim.AdamW([
            {"params": [p for p in vision.parameters() if p.requires_grad], "lr": args.vision_lr},
            {"params": classifier.parameters(), "lr": args.head_lr}], weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        generator = torch.Generator().manual_seed(args.seed)
        for epoch in range(1, args.epochs + 1):
            # Keep the encoder in eval mode: training gradients still flow, but
            # stochastic dropout cannot confound the frozen baseline comparison.
            vision.eval()
            classifier.train()
            losses = []
            ordering = torch.randperm(train_count, generator=generator).tolist()
            for offset in range(0, train_count, args.batch_size):
                ids = ordering[offset:offset + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits(ids), targets[ids].to("cuda"), pos_weight=weights)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss; experiment is not complete")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group["params"]], 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            scheduler.step()
            val_records = records_for(validation, predictions(train_count, val_count))
            metrics = label_metrics(val_records)
            if metrics["atomic_macro_f1"] > best_score:
                best_score, best_epoch = metrics["atomic_macro_f1"], epoch
                best = selected_state(vision, names, classifier)
            report["epochs"].append({"epoch": epoch, "training_loss": sum(losses) / len(losses), "validation_metrics": metrics})
            report["status"] = "training"
            report["selected_epoch"] = best_epoch
            _write_report(output, report)
            print(f"[epoch {epoch}] loss={sum(losses)/len(losses):.4f} val_f1={metrics['atomic_macro_f1']:.4f} selected={best_epoch}", flush=True)
        restore_state(vision, classifier, best)
        # Held-out test labels are consulted only now, after checkpoint selection.
        candidate = records_for(testing, predictions(train_count + val_count, len(testing)))
        baseline = records_for(testing, frozen_test_labels)
        paired = _paired_comparison(baseline, candidate)
        paired["baseline_metrics"] = label_metrics(baseline)
        paired["candidate_metrics"] = label_metrics(candidate)
        report["test"] = {"baseline": {"records": baseline, "metrics": label_metrics(baseline)},
                          "candidate": {"records": candidate, "metrics": label_metrics(candidate)}, "comparison": paired}
        state_path = output.with_suffix(".pt")
        descriptor, temporary = tempfile.mkstemp(prefix="adapter-", suffix=".tmp", dir=output.parent)
        os.close(descriptor)
        try:
            torch.save({**best, "mean": torch.from_numpy(head["mean"]), "scale": torch.from_numpy(head["scale"]),
                        "atoms": atoms, "selected_epoch": best_epoch, "model_identity": metadata["model"]}, temporary)
            os.replace(temporary, state_path)
        finally:
            if Path(temporary).exists():
                Path(temporary).unlink()
        report.update(status="complete", elapsed_seconds=round(time.perf_counter() - started, 3),
                      adapter_sha256=sha256_file(state_path), selected_epoch=best_epoch)
        _write_report(output, report)
        print(json.dumps({"status": "complete", "selected_epoch": best_epoch,
                          "baseline": label_metrics(baseline), "candidate": label_metrics(candidate),
                          "gained_correct": paired["gained_correct"], "lost_correct": paired["lost_correct"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
