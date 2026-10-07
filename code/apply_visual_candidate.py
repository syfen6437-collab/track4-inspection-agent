from __future__ import annotations

"""Build an isolated candidate by combining local visual heads with Qwen text.

The visual head supplies only a model-generated candidate label.  Qwen is
called again for every changed image so the description and rating are based
on the same pixels rather than a hand-written template.  The official result
is never modified by this command.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from run_pipeline import load_client, load_config
from track4_agent.data import load_assets, write_json
from track4_agent.inference import _first_pass_images
from track4_agent.model import _extract_json, scene_key
from track4_agent.vision_review import candidate_path, digest_json, sha256_file


def _candidate_prompt(item: dict[str, Any], label: str, original: dict[str, Any]) -> str:
    category = item["questionCategory"]
    scene = scene_key(item)
    return f"""你是城市桥梁与轨道结构病害巡检专家。这是一张{scene}场景照片。
独立的本地视觉分类器给出的候选病害类型是“{label}”。请重新查看当前图片，
只依据图中可见证据生成与该候选标签相符的简洁描述；如果图片明显不支持该候选，
仍选择最符合图像的合法标签。原模型结果仅供对照，不要盲目沿用：
{json.dumps(original, ensure_ascii=False)}

只能从训练集合法标签中选择 defectType，保持文字完全一致。只输出一个 JSON，
不要 Markdown 或解释：
{{"defectType":"{label}","defectDescription":"不超过30个汉字的可见证据",
"ratingScale":"","confidence":0.0,"evidence":"不超过15个汉字的依据"}}"""


def _candidate_is_contradicted(label: str, prediction: dict[str, Any]) -> bool:
    """Reject a visual candidate when the same model evidence explicitly denies it.

    The candidate label is produced by a separate frozen visual head.  Qwen is
    only a consistency check, so a response such as ``无钢结构`` must never be
    accepted as confirmation of ``钢结构锈蚀``.
    """
    text = " ".join(str(prediction.get(key, "")) for key in ("defectDescription", "evidence"))
    if not text:
        return True
    if label == "完好":
        return False
    negatives = ("无", "未见", "没有", "不存在", "不明显", "未发现", "无明显")
    atoms: set[str] = set()
    if label == "已处治病害（修补）":
        atoms = {"修补", "处治", "修复", "补丁"}
    elif label == "粉红色色斑":
        atoms = {"粉红", "红色", "色斑"}
    elif label == "渗水/泛碱":
        atoms = {"渗水", "泛碱", "水痕", "水渍", "泛白", "盐霜"}
    elif label == "锈蚀/碳化":
        atoms = {"锈", "碳化"}
    else:
        if "裂缝(" in label:
            atoms.add("裂缝")
        if "破损" in label:
            atoms.add("破损")
        if "渗水泛碱" in label:
            atoms.update(("渗水", "泛碱"))
        if "钢结构锈蚀" in label:
            atoms.update(("钢结构", "锈蚀"))
        if "钢筋锈蚀" in label:
            atoms.update(("钢筋", "锈蚀"))
        if "支座锈蚀" in label:
            atoms.update(("支座", "锈蚀"))
    for atom in atoms:
        # A short negation window catches the wording used by the model while
        # avoiding a global ``无`` check for multi-defect descriptions.
        for match in re.finditer(re.escape(atom), text):
            before = text[max(0, match.start() - 8):match.start()]
            if any(negative in before for negative in negatives):
                return True
        if atom == "钢结构锈蚀" and any(phrase in text for phrase in ("无钢结构", "无钢构件")):
            return True
        if atom == "钢筋锈蚀" and "无钢筋" in text:
            return True
    if label == "已处治病害（修补）" and not any(
        keyword in text for keyword in ("修补", "处治", "修复", "补丁")
    ):
        return True
    return False


def _strict_review(raw: str, allowed: set[str], fallback: str) -> dict[str, Any] | None:
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        return None
    label = str(parsed.get("defectType", "")).strip()
    if label not in allowed:
        return None
    description = str(parsed.get("defectDescription", "")).strip()
    if not description:
        description = str(parsed.get("evidence", "")).strip()
    if not description:
        return None
    rating = str(parsed.get("ratingScale", parsed.get("ratingScale(1-5)", ""))).strip()
    if rating not in {"", "1", "2", "3", "4", "5"}:
        rating = ""
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    result = {
        "defectType": label or fallback,
        "defectDescription": description[:300],
        "ratingScale": rating,
        "confidence": confidence,
        "evidence": str(parsed.get("evidence", ""))[:300],
    }
    if _candidate_is_contradicted(result["defectType"], result):
        return None
    return result


def _scene(item: dict[str, Any]) -> str:
    return scene_key(item) if item["questionCategory"] == "桥梁" else "track"


def main() -> None:
    parser = argparse.ArgumentParser(description="生成隔离的视觉头+Qwen候选结果")
    parser.add_argument("--config", type=Path, default=CODE / "config" / "qwen3-vl-4b-review.json")
    parser.add_argument("--base", type=Path, default=ROOT / "result" / "result.json")
    parser.add_argument("--vision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--descriptions", type=Path, required=True)
    parser.add_argument("--bridge-scenes", default="support,bottom")
    parser.add_argument("--track", action="store_true", help="将轨道视觉原子多标签候选纳入")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--revalidate-only", action="store_true",
        help="只用已保存的Qwen原始响应重新执行矛盾证据校验，不重新调用模型",
    )
    args = parser.parse_args()

    for name in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "MODELSCOPE_OFFLINE"):
        os.environ[name] = "1"
    output = candidate_path(ROOT, args.output)
    audit_path = candidate_path(ROOT, args.audit)
    descriptions_path = candidate_path(ROOT, args.descriptions)
    base = json.loads(args.base.resolve().read_text(encoding="utf-8"))
    vision_doc = json.loads(args.vision.resolve().read_text(encoding="utf-8"))
    vision = vision_doc["records"]
    train, manifest, lexicon = load_assets(ROOT)
    by_key = {(row["questionCategory"], row["filename"]): row for row in manifest}
    base_by_key = {(row["questionCategory"], row["filename"]): row for row in base}
    vision_by_key = {(row["questionCategory"], row["filename"]): row for row in vision}
    bridge_scenes = {value.strip() for value in args.bridge_scenes.split(",") if value.strip()}
    allowed = {category: set(data["allowed_labels"])
               for category, data in lexicon["categories"].items()}

    changes: list[dict[str, Any]] = []
    for key, original in base_by_key.items():
        category, _ = key
        item = by_key.get(key)
        visual = vision_by_key.get(key)
        if not item or not visual or float(visual.get("confidence", 0.0)) < args.min_confidence:
            continue
        scene = _scene(item)
        enabled = (category == "桥梁" and scene in bridge_scenes) or (category == "轨道" and args.track)
        if not enabled or visual["defectType"] == original["defectType"]:
            continue
        label = str(visual["defectType"])
        if label not in allowed[category]:
            raise ValueError(f"Illegal visual label {label!r} for {key}")
        changes.append({"key": key, "item": item, "original": original,
                        "visual": visual, "label": label})

    saved: dict[str, dict[str, Any]] = {}
    if args.resume and descriptions_path.exists():
        saved_doc = json.loads(descriptions_path.read_text(encoding="utf-8"))
        saved = {row["key"]: row for row in saved_doc.get("records", [])}
    client = None
    raw_dir = ROOT / "data" / "raw"
    config = load_config(args.config.resolve())
    for index, change in enumerate(changes, start=1):
        key_text = "|".join(change["key"])
        if key_text in saved:
            if args.revalidate_only:
                existing = saved[key_text]
                fallback = str(existing.get("visual", {}).get("defectType", change["label"]))
                parsed = _strict_review(
                    str(existing.get("raw", "")), allowed[item["questionCategory"]], fallback
                )
                existing["prediction"] = parsed
                existing["valid"] = parsed is not None
            continue
        if client is None:
            client = load_client(config)
        item = change["item"]
        images = _first_pass_images(raw_dir / item["image"], item, config)
        raw = client.generate(images, _candidate_prompt(item, change["label"], change["original"]))
        parsed = _strict_review(raw, allowed[item["questionCategory"]], change["label"])
        saved[key_text] = {
            "key": key_text,
            "id": item["id"],
            "questionCategory": item["questionCategory"],
            "filename": item["filename"],
            "visual": change["visual"],
            "raw": raw,
            "valid": parsed is not None,
            "prediction": parsed,
        }
        descriptions_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(descriptions_path, {"model_id": config["model_id"], "count": len(saved),
                                       "records": sorted(saved.values(), key=lambda row: row["key"])})
        print(f"[describe] {index}/{len(changes)} {item['filename']}", flush=True)

    merged = []
    audit = []
    for original in base:
        key = (original["questionCategory"], original["filename"])
        item = by_key[key]
        visual = vision_by_key.get(key)
        key_text = "|".join(key)
        description = saved.get(key_text, {})
        prediction = description.get("prediction") if description.get("valid") else None
        final = dict(original)
        accepted = bool(prediction and prediction["defectType"] == visual["defectType"])
        if accepted:
            final["defectType"] = visual["defectType"]
            final["defectDescription"] = prediction["defectDescription"]
            final["ratingScale(1-5)"] = prediction["ratingScale"]
        merged.append(final)
        if visual and visual["defectType"] != original["defectType"] and ((key[0] == "桥梁" and _scene(item) in bridge_scenes) or (key[0] == "轨道" and args.track)):
            audit.append({"id": item["id"], "filename": item["filename"], "scene": _scene(item),
                          "original": original, "visual": visual, "qwen": description,
                          "accepted": accepted, "final": final})

    context = {"base_sha256": sha256_file(args.base.resolve()),
               "vision_sha256": sha256_file(args.vision.resolve()),
               "config": config, "bridge_scenes": sorted(bridge_scenes),
               "track": args.track, "min_confidence": args.min_confidence,
               "candidate_count": len(changes), "accepted_count": sum(row["accepted"] for row in audit),
               "model_id": config["model_id"], "policy": "local visual head label + local Qwen description"}
    write_json(output, merged)
    write_json(audit_path, {"context": context, "records": audit})
    print(json.dumps({"candidate_count": len(changes), "accepted_count": sum(row["accepted"] for row in audit),
                      "output": str(output), "audit": str(audit_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
