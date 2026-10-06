from __future__ import annotations

import json
import os
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
OUTPUT_KEYS = [
    "questionCategory",
    "bridgeName",
    "defectLocation",
    "filename",
    "defectType",
    "defectDescription",
    "ratingScale(1-5)",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any, *, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=indent)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_review_guidance(
    workspace: Path, train_rows: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Load human QA as aggregate training guidance, never as test answers."""
    candidates = [
        workspace / "qa" / "review_annotations.jsonl",
        workspace / "code" / "review_annotations.jsonl",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return {}
    rows = read_jsonl(path)
    train_labels = {
        str(row.get("image")): str(row.get("defectType", ""))
        for row in (train_rows or [])
    }
    confirmed = [row for row in rows if row.get("status") == "confirmed"]
    confirmed_labels = {
        str(row.get("id")): str(row.get("reviewedLabel") or "") or train_labels.get(str(row.get("id")), "")
        for row in confirmed
    }
    label_counts = Counter(confirmed_labels.values())
    label_counts.pop("", None)
    notes_by_label: dict[str, list[str]] = defaultdict(list)
    for row in confirmed:
        label = confirmed_labels.get(str(row.get("id")), "")
        note = str(row.get("note") or "").strip()
        if label and note and note not in notes_by_label[label]:
            notes_by_label[label].append(note[:100])
    return {
        "confirmed_count": len(confirmed),
        "needs_review_count": sum(row.get("status") == "needs_review" for row in rows),
        "confirmed_label_counts": dict(label_counts.most_common()),
        "confirmed_evidence": {
            label: notes[:3] for label, notes in notes_by_label.items()
        },
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _safe_extract(zip_path: Path, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    marker = raw_dir / ".extracted.json"
    if marker.exists():
        return

    with zipfile.ZipFile(zip_path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        for index, info in enumerate(members, start=1):
            parts = Path(info.filename.replace("\\", "/")).parts
            if not parts:
                continue
            # The archive has a single top-level folder named 赛题四.
            relative = Path(*parts[1:]) if len(parts) > 1 else Path(parts[0])
            if not relative.parts:
                continue
            target = (raw_dir / relative).resolve()
            if raw_dir.resolve() not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            if index % 250 == 0:
                print(f"[prepare] extracted {index}/{len(members)} files", flush=True)

    write_json(marker, {"archive": str(zip_path), "file_count": len(members)})


def _normalize_name(value: str) -> str:
    value = str(value or "").strip().lower()
    value = value.replace("（", "(").replace("）", ")")
    value = re.sub(r"\s+", "", value)
    return value


def _bridge_base(value: str) -> str:
    value = re.sub(r"[（(](左幅|右幅)[）)]", "", str(value or ""))
    return value.strip()


def _side_from_text(value: str) -> str:
    if "左幅" in value:
        return "左幅"
    if "右幅" in value:
        return "右幅"
    return ""


def _iter_images(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: str(path.relative_to(directory)).lower(),
    )


def _dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in OUTPUT_KEYS)
        if key in seen:
            continue
        seen.add(key)
        deduped.append({field: str(row.get(field, "")) for field in OUTPUT_KEYS})
    return deduped, len(rows) - len(deduped)


def _match_training_image(
    row: dict[str, Any],
    candidates_by_basename: dict[str, list[Path]],
    train_dir: Path,
) -> Path:
    basename = _normalize_name(row["filename"])
    candidates = candidates_by_basename.get(basename, [])
    if not candidates:
        raise FileNotFoundError(f"No image for label row: {row}")
    if len(candidates) == 1:
        return candidates[0]

    category = row["questionCategory"]
    if category == "轨道":
        track = [path for path in candidates if path.parent.name == "轨道"]
        if len(track) == 1:
            return track[0]

    base = _normalize_name(_bridge_base(row["bridgeName"]))
    filtered = [
        path
        for path in candidates
        if base and (base in _normalize_name(path.parent.name) or _normalize_name(path.parent.name) in base)
    ]
    if len(filtered) == 1:
        return filtered[0]

    side = _side_from_text(row["bridgeName"])
    side_filtered = [path for path in (filtered or candidates) if not side or side in path.name]
    if len(side_filtered) == 1:
        return side_filtered[0]

    relative_candidates = [str(path.relative_to(train_dir)) for path in candidates]
    raise RuntimeError(f"Ambiguous image mapping for {row}: {relative_candidates}")


def _build_lexicon(rows: list[dict[str, Any]]) -> dict[str, Any]:
    category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        category_rows[row["questionCategory"]].append(row)

    lexicon: dict[str, Any] = {"categories": {}}
    for category, items in category_rows.items():
        counts = Counter(item["defectType"] for item in items)
        label_specs: dict[str, Any] = {}
        for label, count in counts.most_common():
            relevant = [item for item in items if item["defectType"] == label]
            descriptions: list[str] = []
            for item in relevant:
                description = item["defectDescription"].strip()
                if description and description not in descriptions:
                    descriptions.append(description)
                if len(descriptions) >= 3:
                    break
            ratings = Counter(item["ratingScale(1-5)"] for item in relevant)
            label_specs[label] = {
                "count": count,
                "descriptions": descriptions,
                "rating_distribution": dict(ratings.most_common()),
                "default_rating": ratings.most_common(1)[0][0] if ratings else "",
            }
        # Scene-conditioned counts keep the visual prompt small.  The scene is
        # derived only from the training filename and is never a test answer.
        scene_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for item in items:
            name = str(item.get("filename", ""))
            if category == "轨道":
                scene = "track"
            elif re.search(r"DJI_|^S\d|航拍", name, re.IGNORECASE):
                scene = "aerial"
            elif re.search(r"桥面|铺装|道路", name):
                scene = "deck"
            elif re.search(r"支座|墩", name):
                scene = "support"
            elif re.search(r"梁底|跨中|横隔板", name):
                scene = "bottom"
            else:
                scene = "generic"
            scene_counts[scene][item["defectType"]] += 1

        lexicon["categories"][category] = {
            "sample_count": len(items),
            "allowed_labels": list(counts.keys()),
            "label_specs": label_specs,
            "scene_label_counts": {
                scene: dict(counts.most_common()) for scene, counts in scene_counts.items()
            },
        }
    return lexicon


def prepare_dataset(workspace: Path, archive_path: Path) -> dict[str, Any]:
    raw_dir = workspace / "data" / "raw"
    processed_dir = workspace / "data" / "processed"
    _safe_extract(archive_path.resolve(), raw_dir)

    train_dir = raw_dir / "训练集"
    test_dir = raw_dir / "初赛测试集"
    source_rows = read_json(train_dir / "汇总数据.json")
    rows, duplicate_count = _dedupe_rows(source_rows)

    train_images = _iter_images(train_dir)
    test_images = _iter_images(test_dir)
    candidates_by_basename: dict[str, list[Path]] = defaultdict(list)
    for path in train_images:
        candidates_by_basename[_normalize_name(path.name)].append(path)

    training_manifest: list[dict[str, Any]] = []
    for row in rows:
        image_path = _match_training_image(row, candidates_by_basename, train_dir)
        training_manifest.append(
            {
                **row,
                "image": str(image_path.relative_to(raw_dir)).replace("\\", "/"),
            }
        )

    test_manifest: list[dict[str, Any]] = []
    for path in test_images:
        relative = path.relative_to(test_dir)
        folder = relative.parts[0]
        category = "轨道" if folder == "轨道" else "桥梁"
        bridge_name = "" if category == "轨道" else folder
        side = _side_from_text(path.name)
        if bridge_name and side:
            bridge_name = f"{bridge_name}（{side}）"
        test_manifest.append(
            {
                "id": str(relative).replace("\\", "/"),
                "image": str(path.relative_to(raw_dir)).replace("\\", "/"),
                "questionCategory": category,
                "bridgeName": bridge_name,
                "defectLocation": "",
                "filename": path.name,
            }
        )

    lexicon = _build_lexicon(rows)
    write_jsonl(processed_dir / "train_manifest.jsonl", training_manifest)
    write_jsonl(processed_dir / "test_manifest.jsonl", test_manifest)
    write_json(processed_dir / "label_lexicon.json", lexicon)

    duplicate_test_names = [
        name for name, count in Counter(item["filename"].lower() for item in test_manifest).items() if count > 1
    ]
    summary = {
        "source_label_rows": len(source_rows),
        "deduplicated_label_rows": len(rows),
        "removed_exact_duplicates": duplicate_count,
        "train_images": len(train_images),
        "test_images": len(test_images),
        "training_manifest": len(training_manifest),
        "test_manifest": len(test_manifest),
        "duplicate_test_basenames": duplicate_test_names,
        "category_counts": dict(Counter(row["questionCategory"] for row in rows)),
        "test_category_counts": dict(Counter(row["questionCategory"] for row in test_manifest)),
    }
    write_json(processed_dir / "dataset_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def load_assets(workspace: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    processed = workspace / "data" / "processed"
    train = read_jsonl(processed / "train_manifest.jsonl")
    test = read_jsonl(processed / "test_manifest.jsonl")
    lexicon = read_json(processed / "label_lexicon.json")
    review_guidance = _load_review_guidance(workspace, train)
    if review_guidance:
        lexicon["review_guidance"] = review_guidance
    return train, test, lexicon


def copy_submission_code(workspace: Path, destination: Path) -> None:
    source = workspace / "code"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    review_file = workspace / "qa" / "review_annotations.jsonl"
    if review_file.exists():
        shutil.copy2(review_file, destination / "review_annotations.jsonl")
