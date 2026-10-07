"""Frozen visual features and supervised heads learned only from training labels."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps


def atomic_parts(label: str) -> set[str]:
    if label == "完好":
        return set()
    return {x.strip() for x in re.split(r"[、,+，]", label) if x.strip()}


def decode_atomic(probabilities: np.ndarray, atoms: list[str], allowed: list[str]) -> list[str]:
    """Project independently learned defect probabilities onto the public ontology."""
    values = np.clip(np.asarray(probabilities), 1e-7, 1 - 1e-7)
    membership = np.asarray([[atom in atomic_parts(label) for atom in atoms] for label in allowed])
    log_scores = np.log(values) @ membership.T + np.log1p(-values) @ (~membership).T
    return [allowed[i] for i in log_scores.argmax(axis=1)]


def model_identity(model_dir: Path) -> dict[str, str]:
    identity = {}
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        digest = hashlib.sha256()
        with (model_dir / name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        identity[name] = digest.hexdigest()
    return identity


def feature_bank(
    workspace: Path, rows: list[dict[str, Any]], model_dir: Path, *,
    cache_path: Path, device: str = "cpu", batch_size: int = 8, threads: int = 4,
) -> np.ndarray:
    """Cache feature vectors with weight, processor and input-file provenance.

    Cache entries contain no targets. Extracting a held-out image is allowed;
    its label is withheld later when fitting the supervised head.
    """
    for name in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "MODELSCOPE_OFFLINE"):
        os.environ[name] = "1"
    torch.set_num_threads(threads)
    identity = model_identity(model_dir)
    fingerprints = []
    for row in rows:
        stat = (workspace / "data" / "raw" / row["image"]).stat()
        fingerprints.append((row["image"], stat.st_size, stat.st_mtime_ns))
    cache: dict[str, np.ndarray] = {}
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"].item()))
            if metadata["model"] == identity and metadata["device"] == device:
                cache = {str(key): value for key, value in zip(saved["keys"], saved["vectors"])}
    keys = [json.dumps(value, ensure_ascii=False) for value in fingerprints]
    missing = [i for i, key in enumerate(keys) if key not in cache]
    if not missing:
        return np.stack([cache[key] for key in keys])

    from transformers import AutoModel, AutoProcessor
    dtype = torch.float32 if device == "cpu" else torch.float16
    base = AutoModel.from_pretrained(model_dir, dtype=dtype, local_files_only=True,
                                     attn_implementation="sdpa")
    vision = base.vision_model.to(device).eval()
    del base
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)

    def save() -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, keys=np.array(list(cache)), vectors=np.stack(list(cache.values())),
                            metadata=json.dumps({"model": identity, "device": device}, ensure_ascii=False))
        temporary.replace(cache_path)

    print(f"[features] cached={len(rows)-len(missing)} missing={len(missing)} device={device}", flush=True)
    for start in range(0, len(missing), batch_size):
        indices = missing[start:start + batch_size]
        images = []
        for index in indices:
            with Image.open(workspace / "data" / "raw" / rows[index]["image"]) as source:
                images.append(ImageOps.exif_transpose(source).convert("RGB"))
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device, dtype=dtype)
        with torch.inference_mode():
            embedding = vision(pixel_values=pixels).pooler_output.float()
            embedding = torch.nn.functional.normalize(embedding, dim=-1).cpu().numpy()
        for index, vector in zip(indices, embedding):
            cache[keys[index]] = vector
        del pixels, embedding, images
        if (start // batch_size + 1) % 10 == 0 or start + batch_size >= len(missing):
            save()
            print(f"[features] {min(start+batch_size,len(missing))}/{len(missing)}", flush=True)
    del vision, processor
    if device != "cpu":
        torch.cuda.empty_cache()
    return np.stack([cache[key] for key in keys])


def fit_head(features: np.ndarray, labels: list[str], *, multilabel: bool = False,
             seed: int = 20261005) -> dict[str, Any]:
    """Fit a deterministic CPU linear head; query labels are never an argument."""
    torch.manual_seed(seed)
    x = torch.tensor(features, dtype=torch.float32)
    mean = x.mean(0)
    scale = x.std(0).clamp_min(1e-4)
    x = (x - mean) / scale
    vocabulary = sorted(set().union(*(atomic_parts(label) for label in labels))) if multilabel else sorted(set(labels))
    if not vocabulary:
        raise ValueError("No supervised targets available")
    if multilabel:
        target = torch.tensor([[atom in atomic_parts(label) for atom in vocabulary] for label in labels], dtype=torch.float32)
        positive = target.sum(0)
        weights = ((len(labels) - positive) / positive.clamp_min(1)).sqrt().clamp(0.5, 3.0)
    else:
        target = torch.tensor([vocabulary.index(label) for label in labels], dtype=torch.long)
        counts = torch.bincount(target, minlength=len(vocabulary)).float()
        weights = (counts.sum() / counts.clamp_min(1)).sqrt()
        weights /= weights.mean()
    head = torch.nn.Linear(x.shape[1], len(vocabulary))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.02, weight_decay=0.05)
    for epoch in range(250):
        optimizer.zero_grad(set_to_none=True)
        logits = head(x)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=weights)
                if multilabel else torch.nn.functional.cross_entropy(logits, target, weight=weights))
        loss.backward()
        optimizer.step()
        if epoch in {79, 159}:
            for group in optimizer.param_groups:
                group["lr"] *= 0.3
    return {"labels": vocabulary, "multilabel": multilabel, "mean": mean.numpy(), "scale": scale.numpy(),
            "weight": head.weight.detach().numpy(), "bias": head.bias.detach().numpy()}


def predict_head(head: dict[str, Any], features: np.ndarray, allowed: list[str]) -> tuple[list[str], np.ndarray]:
    x = (np.asarray(features) - head["mean"]) / head["scale"]
    logits = x @ head["weight"].T + head["bias"]
    if head["multilabel"]:
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -60, 60)))
        return decode_atomic(probabilities, head["labels"], allowed), probabilities
    probabilities = torch.tensor(logits).softmax(-1).numpy()
    return [head["labels"][i] for i in probabilities.argmax(1)], probabilities
