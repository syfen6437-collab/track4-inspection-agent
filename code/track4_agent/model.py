from __future__ import annotations

import difflib
import gc
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


def download_model(model_id: str, model_dir: Path) -> Path:
    model_dir = model_dir.resolve()
    config_file = model_dir / "config.json"
    if config_file.exists():
        print(f"[download] model already present: {model_dir}")
        return model_dir
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    from modelscope import snapshot_download

    print(f"[download] downloading {model_id} to {model_dir}", flush=True)
    path = snapshot_download(model_id, local_dir=str(model_dir))
    print(f"[download] completed: {path}", flush=True)
    return Path(path)


def resize_global(image: Image.Image, max_side: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    copy = image.copy()
    copy.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return copy


def quadrant_crops(image: Image.Image, crop_size: int) -> list[Image.Image]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    overlap_x = max(1, int(width * 0.08))
    overlap_y = max(1, int(height * 0.08))
    mid_x, mid_y = width // 2, height // 2
    boxes = [
        (0, 0, min(width, mid_x + overlap_x), min(height, mid_y + overlap_y)),
        (max(0, mid_x - overlap_x), 0, width, min(height, mid_y + overlap_y)),
        (0, max(0, mid_y - overlap_y), min(width, mid_x + overlap_x), height),
        (max(0, mid_x - overlap_x), max(0, mid_y - overlap_y), width, height),
    ]
    crops: list[Image.Image] = []
    for box in boxes:
        crop = image.crop(box)
        crop.thumbnail((crop_size, crop_size), Image.Resampling.LANCZOS)
        crops.append(crop)
    return crops


def crop_montage(image: Image.Image, crop_size: int) -> Image.Image:
    """Combine four overlapping quadrants into one 2x2 detail image."""
    crops = quadrant_crops(image, crop_size)
    montage = Image.new("RGB", (crop_size * 2, crop_size * 2), "white")
    positions = ((0, 0), (crop_size, 0), (0, crop_size), (crop_size, crop_size))
    for crop, position in zip(crops, positions):
        x = position[0] + max(0, (crop_size - crop.width) // 2)
        y = position[1] + max(0, (crop_size - crop.height) // 2)
        montage.paste(crop, (x, y))
    return montage


def lexicon_prompt(lexicon: dict[str, Any], category: str) -> str:
    category_data = lexicon["categories"][category]
    # Keep the prompt compact while exposing the training distribution. This
    # is especially important for bridge labels, where many rare fine-grained
    # combinations coexist with a few dominant classes.
    labels = sorted(
        category_data["allowed_labels"],
        key=lambda label: category_data["label_specs"][label].get("count", 0),
        reverse=True,
    )
    return "、".join(labels)


def build_prompt(
    item: dict[str, Any],
    lexicon: dict[str, Any],
    *,
    review: bool = False,
    prior_prediction: dict[str, Any] | None = None,
) -> str:
    category = item["questionCategory"]
    label_text = lexicon_prompt(lexicon, category)
    prior = ""
    if prior_prediction:
        prior = "\n上一轮候选结果如下，请基于新视角复核，不要无条件沿用：\n" + json.dumps(
            prior_prediction, ensure_ascii=False
        )
    review_instruction = (
        "当前输入包含整图之外的局部裁剪或相邻帧，请重点核对细裂缝、锈蚀、剥落和渗水痕迹。"
        if review
        else "先观察整体结构和主要病害，不要把阴影、施工接缝或拍摄反光误判为病害。"
    )
    return f"""你是城市桥梁与轨道结构病害巡检专家。{review_instruction}

输入元数据：
- 问题类型：{category}
- 结构名称：{item.get('bridgeName', '') or '轨道设施'}
- 照片编号：{item['filename']}

只能从以下训练集合法病害类型中选择 defectType，必须保持文字完全一致：
{label_text}

要求：
1. defectDescription 使用不超过30个汉字描述可见证据、部位、范围和程度，不得臆测图外信息。
2. ratingScale 只能是空字符串或字符串 "1" 到 "5"。完好或训练集中通常不评分的情况优先为空。
3. confidence 是 0 到 1 的小数，表示对病害类型判断的把握。
4. evidence 使用不超过15个汉字列出最关键的可见依据。
5. 只输出一个 JSON 对象，不要 Markdown，不要解释。

JSON 字段必须严格为：
{{"defectType":"合法标签","defectDescription":"描述","ratingScale":"","confidence":0.0,"evidence":"依据"}}
{prior}"""


def _extract_json(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    candidates = [cleaned]
    first, last = cleaned.find("{"), cleaned.rfind("}")
    if 0 <= first < last:
        candidates.append(cleaned[first : last + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None


def _closest_label(raw: str, allowed: list[str]) -> str:
    raw = str(raw or "").strip()
    if raw in allowed:
        return raw
    for label in sorted(allowed, key=len, reverse=True):
        if label and label in raw:
            return label
    matches = difflib.get_close_matches(raw, allowed, n=1, cutoff=0.25)
    if matches:
        return matches[0]
    # Training-derived fallback: the first label is the most frequent label.
    return allowed[0]


def sanitize_prediction(
    raw_text: str,
    lexicon: dict[str, Any],
    category: str,
) -> tuple[dict[str, Any], bool]:
    parsed = _extract_json(raw_text)
    was_valid_json = parsed is not None
    parsed = parsed or {}
    category_data = lexicon["categories"][category]
    allowed = sorted(
        category_data["allowed_labels"],
        key=lambda candidate: category_data["label_specs"][candidate].get("count", 0),
        reverse=True,
    )
    label = _closest_label(str(parsed.get("defectType", raw_text)), allowed)

    description = str(parsed.get("defectDescription", "")).strip()
    if not description:
        description = str(parsed.get("evidence", "")).strip()
    if not description:
        description = re.sub(r"\s+", " ", raw_text).strip()[:180]

    rating = str(parsed.get("ratingScale", parsed.get("ratingScale(1-5)", ""))).strip()
    observed_ratings = category_data["label_specs"][label].get("rating_distribution", {})
    if rating not in {"", "1", "2", "3", "4", "5"} or (observed_ratings and rating not in observed_ratings):
        rating = str(category_data["label_specs"][label].get("default_rating", ""))

    try:
        confidence = float(parsed.get("confidence", 0.0))
        if not math.isfinite(confidence):
            confidence = 0.0
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    evidence = str(parsed.get("evidence", "")).strip()
    prediction = {
        "defectType": label,
        "defectDescription": description[:300],
        "ratingScale": rating,
        "confidence": confidence,
        "evidence": evidence[:300],
        "raw": raw_text,
    }
    return prediction, was_valid_json and str(parsed.get("defectType", "")) in allowed


def needs_review(prediction: dict[str, Any], threshold: float) -> bool:
    if prediction.get("confidence", 0.0) < threshold:
        return True
    description = str(prediction.get("defectDescription", ""))
    label = str(prediction.get("defectType", ""))
    contradiction_terms = ("裂缝", "破损", "锈蚀", "渗水", "剥落", "露筋")
    if label == "完好" and any(term in description for term in contradiction_terms):
        return True
    return False


@dataclass
class ModelClient:
    model_path: Path
    max_new_tokens: int = 72
    global_max_side: int = 448
    crop_size: int = 224

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelClass

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for the six-hour baseline")
        torch.manual_seed(20261005)
        torch.cuda.manual_seed_all(20261005)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            min_pixels=28 * 28 * 16,
            max_pixels=self.global_max_side * self.global_max_side,
            trust_remote_code=True,
        )
        self.model = ModelClass.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: "11GiB", "cpu": "22GiB"},
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"[model] loaded on {self.device}", flush=True)

    def generate(self, images: list[Image.Image], prompt: str) -> str:
        content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": "你必须进行真实视觉判断并严格输出 JSON。"},
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids)]
        text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        del inputs, output_ids, trimmed
        gc.collect()
        self.torch.cuda.empty_cache()
        return text.strip()
