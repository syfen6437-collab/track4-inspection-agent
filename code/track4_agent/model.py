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


def scene_key(item: dict[str, Any]) -> str:
    if item.get("questionCategory") == "轨道":
        return "track"
    filename = str(item.get("filename", ""))
    if re.search(r"DJI_|^S\d|航拍", filename, re.IGNORECASE):
        return "aerial"
    if re.search(r"桥面|铺装|道路", filename):
        return "deck"
    if re.search(r"支座|墩", filename):
        return "support"
    if re.search(r"梁底|跨中|横隔板", filename):
        return "bottom"
    return "generic"


def candidate_labels(
    lexicon: dict[str, Any], category: str, scene: str, limit: int = 12,
    *, include_all: bool = False,
) -> list[str]:
    category_data = lexicon["categories"][category]
    if include_all:
        # A candidate scope is useful for fast prompts, but it must never be
        # mistaken for the ontology. Rare labels such as 伸缩缝病害 and 剥落
        # are valid training targets and must remain selectable in experiments.
        return list(category_data["allowed_labels"])
    scene_counts = category_data.get("scene_label_counts", {}).get(scene, {})
    # Test filenames use “桥面” while the public training set mostly calls
    # equivalent views DJI/S*.  Reuse that visual domain instead of the tiny
    # residual generic bucket; this is metadata routing, not an answer lookup.
    if category == "桥梁" and scene == "deck" and not scene_counts:
        scene_counts = category_data.get("scene_label_counts", {}).get("aerial", {})
    if category == "桥梁" and scene == "generic":
        scene_counts = {}
    if not scene_counts:
        scene_counts = category_data.get("label_specs", {})
        scene_counts = {label: spec.get("count", 0) for label, spec in scene_counts.items()}
    labels = [label for label, _ in sorted(scene_counts.items(), key=lambda pair: pair[1], reverse=True)]
    reviewed_counts = lexicon.get("review_guidance", {}).get("confirmed_label_counts", {})
    for label in reviewed_counts:
        if label in category_data["allowed_labels"] and label not in labels:
            labels.append(label)
    # Always retain the base defect classes for a new bridge/test scene whose
    # filename has no exact training counterpart.
    base = ("完好", "渗水/泛碱", "已处治病害（修补）", "锈蚀/碳化", "混凝土外观瑕疵")
    for label in base:
        if label in category_data["allowed_labels"] and label not in labels:
            labels.append(label)
    if category == "轨道":
        # The track taxonomy has only a few dozen legal combinations. Keeping
        # all of them prevents a rare but valid multi-defect label from being
        # excluded before the model sees the image.
        return [label for label in category_data["allowed_labels"] if label in labels or label in reviewed_counts]
    return labels[:limit]


def lexicon_prompt(
    lexicon: dict[str, Any], category: str, scene: str = "generic", *,
    label_scope: str = "scene",
) -> str:
    return "、".join(candidate_labels(
        lexicon, category, scene, include_all=label_scope == "all"
    ))


def review_guidance_prompt(lexicon: dict[str, Any]) -> str:
    """Summarize reviewed evidence without binding any image to a test answer."""
    guidance = lexicon.get("review_guidance", {})
    evidence = guidance.get("confirmed_evidence", {})
    if not evidence:
        return ""
    lines = []
    for label, notes in list(evidence.items())[:8]:
        lines.append(f"- {label}: {'；'.join(notes[:2])}")
    return (
        "\n训练集人工审核证据摘要（仅用于观察重点，不是测试答案，也不能替代当前图像判断）：\n"
        + "\n".join(lines)
    )


def build_prompt(
    item: dict[str, Any],
    lexicon: dict[str, Any],
    *,
    review: bool = False,
    prior_prediction: dict[str, Any] | None = None,
    label_scope: str = "scene",
) -> str:
    category = item["questionCategory"]
    scene = scene_key(item)
    label_text = lexicon_prompt(lexicon, category, scene, label_scope=label_scope)
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
    scene_instruction = {
        "aerial": "这是桥梁航拍/远景图，重点检查桥面铺装、标线、伸缩缝、护栏和明显渗水；远景看不清时不要臆测细小病害。",
        "deck": "这是桥面近景图，重点检查铺装裂缝、坑槽、接缝、积水/渗水和明显污染；车辆阴影与沥青纹理不是病害。",
        "support": "这是桥墩或支座近景图，重点检查支座、墩顶、盖梁的锈蚀、锈水、渗水泛碱、剥落和露筋。",
        "bottom": "这是梁底/跨中近景图，重点检查混凝土裂缝、剥落露筋、渗水泛碱、锈蚀和已处治修补痕。",
        "track": "这是轨道结构近景图，先判断裂缝、破损、锈蚀、渗水泛碱或完好，再选择组合标签。",
        "generic": "先判断照片中的结构部位，再选择最符合可见证据的病害。",
    }[scene]
    reviewed_guidance = review_guidance_prompt(lexicon)
    return f"""你是城市桥梁与轨道结构病害巡检专家。{scene_instruction} {review_instruction}

输入元数据：
- 问题类型：{category}
- 结构名称：{item.get('bridgeName', '') or '轨道设施'}
- 照片编号：{item['filename']}

只能从以下训练集合法病害类型中选择 defectType，必须保持文字完全一致：
{label_text}
{reviewed_guidance}

要求：
1. 先在内部判断 visible_defect：只有看到裂缝、剥落、锈蚀、渗水/泛碱、明显修补或明确异常色斑时才选病害；正常纹理、阴影、施工接缝和远景不可辨细节应选“完好”。
2. defectDescription 使用不超过30个汉字描述可见证据、部位、范围和程度，不得臆测图外信息。
3. ratingScale 只能是空字符串或字符串 "1" 到 "5"。完好或训练集中通常不评分的情况优先为空。
4. confidence 是 0 到 1 的小数，表示对病害类型判断的把握。
5. evidence 使用不超过15个汉字列出最关键的可见依据。
6. 只输出一个 JSON 对象，不要 Markdown，不要解释。

JSON 字段必须严格为：
{{"defectType":"合法标签","defectDescription":"描述","ratingScale":"","confidence":0.0,"evidence":"依据"}}
{prior}"""


def build_presence_prompt(item: dict[str, Any], *, review: bool = False) -> str:
    scene = scene_key(item)
    scene_hint = {
        "aerial": "航拍桥面远景",
        "deck": "桥面近景",
        "support": "桥墩或支座近景",
        "bottom": "梁底或跨中近景",
        "track": "轨道结构近景",
        "generic": "桥梁结构照片",
    }[scene]
    return f"""你是基础设施病害初筛专家。这是一张{scene_hint}。{('请重点检查细小缺陷。' if review else '只依据图中可见证据。')}
先判断是否存在明确可见病害，再给出一个粗类别。阴影、反光、正常施工接缝、混凝土纹理和远景不可辨细节不算病害。
粗类别只能从：完好、裂缝、破损/剥落、渗水/泛碱、锈蚀、修补、异常色斑、其他 中选择。
只输出 JSON：{{"has_defect":true,"coarse_type":"锈蚀","evidence":"不超过15个汉字","confidence":0.0}}"""


def build_checklist_prompt(
    item: dict[str, Any], lexicon: dict[str, Any], *, review: bool = False,
    label_scope: str = "scene",
) -> str:
    scene = scene_key(item)
    limit = len(lexicon["categories"][item["questionCategory"]]["allowed_labels"])
    labels = candidate_labels(
        lexicon, item["questionCategory"], scene,
        limit=limit if item["questionCategory"] == "轨道" else 18,
        include_all=label_scope == "all",
    )
    examples = []
    specs = lexicon["categories"][item["questionCategory"]]["label_specs"]
    for label in labels:
        descriptions = specs.get(label, {}).get("descriptions", [])
        examples.append(f"- {label}: {'；'.join(descriptions[:2])}")
    return f"""你是城市桥梁结构病害复核专家。这是{scene}场景照片。{('请放大核对局部痕迹。' if review else '只依据当前图片可见内容。')}
请逐项检查候选标签是否有明确视觉证据。正常阴影、反光、施工接缝和普通混凝土纹理不要勾选。
候选标签及训练集描述：
{chr(10).join(examples)}
只输出一个 JSON，selected 必须是上述候选标签的原文数组；没有明确病害时 selected 为 ["完好"]：
{{"selected": ["标签1"], "description":"不超过30个汉字的可见证据", "ratingScale":"", "confidence":0.0}}"""


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


def checklist_prediction(
    raw_text: str,
    lexicon: dict[str, Any],
    category: str,
    scene: str,
) -> tuple[dict[str, Any], bool]:
    parsed = _extract_json(raw_text) or {}
    category_data = lexicon["categories"][category]
    allowed = category_data["allowed_labels"]
    candidates = candidate_labels(lexicon, category, scene, limit=14)
    selected = parsed.get("selected", [])
    if not isinstance(selected, list):
        selected = [selected]
    atomic_allowed = {
        part.strip()
        for candidate in allowed
        for part in re.split(r"[、,+，]", candidate)
        if part.strip()
    }
    selected = [
        str(value).strip()
        for value in selected
        if str(value).strip() in allowed or str(value).strip() in atomic_allowed
    ]
    selected = list(dict.fromkeys(selected))
    if not selected:
        selected = ["完好"] if "完好" in allowed else [candidates[0]]

    # Prefer an exact training label; otherwise choose the legal combination
    # with the greatest overlap with the model-selected labels.
    label = selected[0]
    if len(selected) > 1:
        def score(candidate: str) -> tuple[int, int, int]:
            overlap = len(set(re.split(r"[、,+，]", candidate)) & set(selected))
            return (overlap, int(candidate in candidates), -len(candidate))
        label = max(allowed, key=score)
    description = str(parsed.get("description", "")).strip() or ("未见明确病害" if label == "完好" else "可见结构异常")
    rating = str(parsed.get("ratingScale", "")).strip()
    observed = category_data["label_specs"].get(label, {}).get("rating_distribution", {})
    if rating not in {"", "1", "2", "3", "4", "5"} or (observed and rating not in observed):
        rating = str(category_data["label_specs"].get(label, {}).get("default_rating", ""))
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    prediction = {
        "defectType": label,
        "defectDescription": description[:300],
        "ratingScale": rating,
        "confidence": confidence,
        "evidence": description[:60],
        "raw": raw_text,
    }
    return prediction, bool(_extract_json(raw_text)) and bool(selected)


def _closest_label(raw: str, allowed: list[str], fallback: str | None = None) -> str:
    raw = str(raw or "").strip()
    if raw in allowed:
        return raw
    for label in sorted(allowed, key=len, reverse=True):
        if label and label in raw:
            return label
    matches = difflib.get_close_matches(raw, allowed, n=1, cutoff=0.25)
    if matches:
        return matches[0]
    # Keep fallback scoped to the current scene rather than the global bridge
    # majority, which otherwise turns most ambiguous images into 完好.
    return fallback or allowed[0]


def sanitize_prediction(
    raw_text: str,
    lexicon: dict[str, Any],
    category: str,
    scene: str = "generic",
) -> tuple[dict[str, Any], bool]:
    parsed = _extract_json(raw_text)
    was_valid_json = parsed is not None
    parsed = parsed or {}
    category_data = lexicon["categories"][category]
    allowed = candidate_labels(lexicon, category, scene)
    global_allowed = category_data["allowed_labels"]
    raw_label = str(parsed.get("defectType", raw_text))
    label = _closest_label(raw_label, allowed, fallback=allowed[0])
    # Exact globally legal output is retained even if it is rare in this scene;
    # the model may have seen visual evidence that the filename cannot express.
    if raw_label.strip() in global_allowed:
        label = raw_label.strip()

    description = str(parsed.get("defectDescription", "")).strip()
    if not description:
        description = str(parsed.get("evidence", "")).strip()
    if not description:
        description = re.sub(r"\s+", " ", raw_text).strip()[:180]

    rating = str(parsed.get("ratingScale", parsed.get("ratingScale(1-5)", ""))).strip()
    observed_ratings = category_data["label_specs"][label].get("rating_distribution", {})
    # An empty rating is valid for healthy/track labels, but the bridge
    # training labels assign a rating to every observed non-healthy class.
    # Use the training-derived mode when the model omits that required value.
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
    return prediction, was_valid_json and str(parsed.get("defectType", "")) in global_allowed


def needs_review(prediction: dict[str, Any], threshold: float) -> bool:
    if prediction.get("confidence", 0.0) < threshold:
        return True
    description = str(prediction.get("defectDescription", ""))
    label = str(prediction.get("defectType", ""))
    contradiction_terms = ("裂缝", "破损", "锈蚀", "渗水", "剥落", "露筋")
    if label == "完好":
        # Healthy evidence often explicitly says “无裂缝/未见锈蚀”.  Those
        # negated terms are not contradictions and must not trigger a costly
        # second visual pass.
        positive = re.sub(
            r"(?:无|未见|未发现|没有|未观察到|未检测到)[^，。；,;]{0,4}(?:裂缝|破损|锈蚀|渗水|剥落|露筋)",
            "",
            description,
        )
        if any(term in positive for term in contradiction_terms):
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

    def embed_image(self, image: Image.Image) -> Any:
        """Return a normalized pooled Qwen vision feature for nearest-neighbor QA.

        This is an auxiliary model-only retrieval path. It never reads test
        labels and is useful for comparing visual similarity against the
        training lexicon before spending time on generative inference.
        """
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": "视觉特征"}],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        device = self.device
        pixel_values = inputs["pixel_values"].to(device, dtype=self.torch.bfloat16)
        grid_thw = inputs["image_grid_thw"].to(device)
        with self.torch.inference_mode():
            output = self.model.model.visual(pixel_values, grid_thw)
            feature = output.last_hidden_state.float().mean(dim=0)
            feature = self.torch.nn.functional.normalize(feature, dim=0)
        del inputs, pixel_values, grid_thw, output
        gc.collect()
        self.torch.cuda.empty_cache()
        return feature.detach().cpu()
