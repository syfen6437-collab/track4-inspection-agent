from __future__ import annotations

import json
import hashlib
import shutil
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

from .data import OUTPUT_KEYS, copy_submission_code, read_json, write_json


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def validate_result(
    workspace: Path,
    manifest: list[dict[str, Any]],
    lexicon: dict[str, Any],
    result_path: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    result_path = result_path or workspace / "result" / "result.json"
    errors: list[str] = []
    try:
        results = json.loads(result_path.read_text(encoding="utf-8-sig"),
                             object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeError, ValueError) as exc:
        errors.append(f"result.json is not valid UTF-8 JSON: {exc}")
        results = []
    if not isinstance(results, list):
        errors.append("result.json root must be an array")
        results = []
    if len(results) != len(manifest):
        errors.append(f"result count {len(results)} != manifest count {len(manifest)}")

    expected_names = Counter(item["filename"] for item in manifest)
    if any(not isinstance(name, str) or not name.strip() for name in expected_names):
        errors.append("manifest contains an empty or non-string filename")
    if any(count != 1 for count in expected_names.values()):
        errors.append("manifest contains duplicate filenames; image matching is ambiguous")
    by_name = {item["filename"]: item for item in manifest}
    actual_names = Counter(item.get("filename", "") for item in results
                           if isinstance(item, dict) and isinstance(item.get("filename", ""), str))
    if expected_names != actual_names:
        errors.append("filename multiset does not match test manifest")

    for index, result in enumerate(results):
        if not isinstance(result, dict):
            errors.append(f"row {index}: not an object")
            continue
        if list(result.keys()) != OUTPUT_KEYS:
            errors.append(f"row {index}: keys/order mismatch: {list(result.keys())}")
        for key in OUTPUT_KEYS:
            if not isinstance(result.get(key), str):
                errors.append(f"row {index}: {key} must be a string")
        filename = result.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            errors.append(f"row {index}: empty or non-string filename")
        expected = by_name.get(filename) if isinstance(filename, str) else None
        if expected is not None:
            for key in ("questionCategory", "bridgeName", "defectLocation", "filename"):
                if result.get(key) != expected.get(key, ""):
                    errors.append(f"row {index}: {key} does not match input metadata for {filename}")
        category = result.get("questionCategory")
        if not isinstance(category, str):
            continue
        if category not in lexicon["categories"]:
            errors.append(f"row {index}: invalid category {category!r}")
            continue
        label = result.get("defectType", "")
        if label not in lexicon["categories"][category]["allowed_labels"]:
            errors.append(f"row {index}: invalid label {label!r} for {category}")
        rating = result.get("ratingScale(1-5)", "")
        if not isinstance(rating, str) or rating not in {"", "1", "2", "3", "4", "5"}:
            errors.append(f"row {index}: invalid rating {rating!r}")

    report = {
        "valid": not errors,
        "result_count": len(results),
        "expected_count": len(manifest),
        "error_count": len(errors),
        "errors": errors[:100],
    }
    write_json(report_path or workspace / "logs" / "validation_report.json", report)
    if errors:
        raise ValueError("Result validation failed:\n" + "\n".join(errors[:20]))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_submission_package(workspace: Path) -> Path:
    provenance_path = workspace / "logs" / "submission_provenance.json"
    if not provenance_path.exists():
        raise ValueError("Missing submission provenance; register the result before packaging")
    provenance = read_json(provenance_path)
    result_hash = hashlib.sha256((workspace / "result" / "result.json").read_bytes()).hexdigest()
    if provenance.get("result_sha256") != result_hash:
        raise ValueError("Result differs from the registered submission provenance; package was not changed")
    staging = workspace / "submission_staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "code").mkdir(parents=True, exist_ok=True)
    (staging / "design").mkdir(parents=True, exist_ok=True)
    (staging / "result").mkdir(parents=True, exist_ok=True)
    copy_submission_code(workspace, staging / "code")

    submission_logs = staging / "code" / "logs"
    submission_logs.mkdir(parents=True, exist_ok=True)
    for name in (
        "calibration_metrics.json",
        "calibration_records.json",
        "qwen3_vl_4b_calibration.json",
        "qwen3_vl_4b_smoke.json",
        "inference_summary.json",
        "raw_responses.jsonl",
        "validation_report.json",
        "submission_provenance.json",
    ):
        source = workspace / "logs" / name
        if source.exists():
            shutil.copy2(source, submission_logs / name)
    dataset_summary = workspace / "data" / "processed" / "dataset_summary.json"
    if dataset_summary.exists():
        shutil.copy2(dataset_summary, submission_logs / "dataset_summary.json")

    design_source = workspace / "design" / "AI智能体设计方案.docx"
    result_source = workspace / "result" / "result.json"
    if not design_source.exists():
        raise FileNotFoundError(design_source)
    if not result_source.exists():
        raise FileNotFoundError(result_source)

    shutil.copy2(design_source, staging / "design" / design_source.name)
    shutil.copy2(result_source, staging / "result" / result_source.name)

    # Keep one canonical artifact in the workspace. Re-running package replaces
    # this file instead of creating timestamped copies that drift apart.
    output = workspace / "track4_submission.tar.gz"
    with tarfile.open(output, "w:gz") as archive:
        for folder in ("code", "design", "result"):
            archive.add(staging / folder, arcname=folder, recursive=True)

    with tarfile.open(output, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        roots = {member.name.split("/", 1)[0] for member in members}
        oversized = [member.name for member in members if member.size >= 1024**3]
        if roots != {"code", "design", "result"}:
            raise RuntimeError(f"Unexpected package roots: {roots}")
        if oversized:
            raise RuntimeError(f"Files exceed 1 GB: {oversized}")
        required = {"design/AI智能体设计方案.docx", "result/result.json"}
        names = {member.name for member in members}
        missing = required - names
        if missing:
            raise RuntimeError(f"Missing package files: {missing}")

    package_report = {
        "package": str(output),
        "result_sha256": result_hash,
        "source_commit": provenance.get("source_commit"),
        "package_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "size_bytes": output.stat().st_size,
        "file_count": len(members),
        "roots": sorted(roots),
        "largest_files": sorted(
            ({"name": member.name, "size": member.size} for member in members),
            key=lambda item: item["size"],
            reverse=True,
        )[:10],
    }
    write_json(workspace / "logs" / "package_report.json", package_report)
    print(json.dumps(package_report, ensure_ascii=False, indent=2))
    return output
