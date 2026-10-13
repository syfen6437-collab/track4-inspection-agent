from __future__ import annotations

import json
import hashlib
import gzip
import os
import shutil
import subprocess
import tarfile
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from .data import OUTPUT_KEYS, copy_submission_code, load_assets, read_json, write_json


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


MAX_SUBMISSION_BYTES = 1_000_000_000
PACKAGE_MANIFEST = "code/logs/submission_manifest.json"


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_submission_package(output: Path) -> dict[str, Any]:
    if output.stat().st_size >= MAX_SUBMISSION_BYTES:
        raise ValueError("Submission archive must be smaller than 1 GB")
    # tar readers may stop at their end marker before checking the gzip trailer.
    with gzip.open(output, "rb") as handle:
        while handle.read(1024 * 1024):
            pass
    hashes = {}
    with tarfile.open(output, "r:gz") as archive:
        members = archive.getmembers()
        names = set()
        roots = set()
        for member in members:
            name = member.name
            path = PurePosixPath(name)
            if (name in names or path.is_absolute() or ".." in path.parts
                    or "\\" in name or not path.parts or path.as_posix() != name.rstrip("/")
                    or not (member.isfile() or member.isdir())):
                raise ValueError(f"Unsafe or duplicate archive member: {name}")
            names.add(name)
            roots.add(path.parts[0])
            if member.isfile():
                if member.size >= MAX_SUBMISSION_BYTES:
                    raise ValueError(f"File exceeds 1 GB: {name}")
                with archive.extractfile(member) as handle:
                    hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
        if roots != {"code", "design", "result"}:
            raise ValueError(f"Unexpected package roots: {roots}")
        required = {"code/run_pipeline.py", "code/README.md", "code/MODEL.md",
                    "code/requirements.txt", "code/logs/raw_responses.jsonl",
                    "code/logs/submission_provenance.json", "code/logs/test_manifest.jsonl",
                    "code/logs/label_lexicon.json", PACKAGE_MANIFEST,
                    "design/AI智能体设计方案.docx", "result/result.json"}
        if required - hashes.keys():
            raise ValueError(f"Missing package files: {sorted(required - hashes.keys())}")
        with archive.extractfile(PACKAGE_MANIFEST) as handle:
            manifest = json.load(handle, object_pairs_hook=_unique_object,
                                 parse_constant=_reject_constant)
        expected = manifest.get("file_sha256")
        actual = {name: digest for name, digest in hashes.items() if name != PACKAGE_MANIFEST}
        if manifest.get("version") != 1 or expected != actual:
            raise ValueError("Archive contents differ from the submission manifest")
        with archive.extractfile("code/logs/submission_provenance.json") as handle:
            provenance = json.load(handle, object_pairs_hook=_unique_object,
                                   parse_constant=_reject_constant)
        result_hash = hashes["result/result.json"]
        if (manifest.get("result_sha256") != result_hash
                or provenance.get("result_sha256") != result_hash
                or manifest.get("result_source_commit") != provenance.get("source_commit")):
            raise ValueError("Packaged result differs from its registered provenance")
    return {
        "valid": True,
        "package": str(output.resolve()),
        "result_sha256": result_hash,
        "source_commit": provenance.get("source_commit"),
        "engineering_commit": manifest.get("engineering_commit"),
        "engineering_dirty": manifest.get("engineering_dirty"),
        "design_sha256": hashes["design/AI智能体设计方案.docx"],
        "package_sha256": _sha256_file(output),
        "size_bytes": output.stat().st_size,
        "file_count": len(hashes),
        "roots": sorted(roots),
        "all_file_hashes_verified": True,
        "largest_files": sorted(
            ({"name": member.name, "size": member.size} for member in members if member.isfile()),
            key=lambda item: item["size"], reverse=True,
        )[:10],
    }


def _engineering_revision(workspace: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace,
                                         stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--", "code", "design"],
            cwd=workspace, stderr=subprocess.DEVNULL))
        return {"engineering_commit": commit, "engineering_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"engineering_commit": None, "engineering_dirty": None}


def build_submission_package(workspace: Path) -> Path:
    provenance_path = workspace / "logs" / "submission_provenance.json"
    if not provenance_path.exists():
        raise ValueError("Missing submission provenance; register the result before packaging")
    provenance = read_json(provenance_path)
    result_hash = _sha256_file(workspace / "result" / "result.json")
    if provenance.get("result_sha256") != result_hash:
        raise ValueError("Result differs from the registered submission provenance; package was not changed")
    # Build in an owned temporary directory; never remove another staging tree.
    with tempfile.TemporaryDirectory(prefix="track4-package-", dir=workspace) as folder:
        return _build_verified_package(workspace, Path(folder), provenance, result_hash)


def _build_verified_package(workspace, staging, provenance, result_hash):
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
        "delivery_audit.json",
    ):
        source = workspace / "logs" / name
        if source.exists():
            shutil.copy2(source, submission_logs / name)
    dataset_summary = workspace / "data" / "processed" / "dataset_summary.json"
    if dataset_summary.exists():
        shutil.copy2(dataset_summary, submission_logs / "dataset_summary.json")
    # Small input snapshots let reviewers replay saved emissions without images,
    # model weights, or assumptions about another checkout's reviewed lexicon.
    _, _, lexicon = load_assets(workspace)
    shutil.copy2(workspace / "data" / "processed" / "test_manifest.jsonl",
                 submission_logs / "test_manifest.jsonl")
    write_json(submission_logs / "label_lexicon.json", lexicon)

    design_source = workspace / "design" / "AI智能体设计方案.docx"
    result_source = workspace / "result" / "result.json"
    if not design_source.exists():
        raise FileNotFoundError(design_source)
    if not result_source.exists():
        raise FileNotFoundError(result_source)

    shutil.copy2(design_source, staging / "design" / design_source.name)
    shutil.copy2(result_source, staging / "result" / result_source.name)

    files = [path for path in staging.rglob("*") if path.is_file()]
    if any(path.stat().st_size >= MAX_SUBMISSION_BYTES for path in files):
        raise ValueError("A submission file exceeds 1 GB")
    manifest = {
        "version": 1,
        "result_sha256": result_hash,
        "result_source_commit": provenance.get("source_commit"),
        **_engineering_revision(workspace),
        "file_sha256": {path.relative_to(staging).as_posix(): _sha256_file(path)
                        for path in sorted(files)},
    }
    write_json(staging / PACKAGE_MANIFEST, manifest)

    # Keep one canonical artifact in the workspace. Re-running package replaces
    # this file instead of creating timestamped copies that drift apart.
    output = workspace / "track4_submission.tar.gz"
    temporary = staging / "submission.tar.gz"
    with tarfile.open(temporary, "w:gz") as archive:
        for folder in ("code", "design", "result"):
            archive.add(staging / folder, arcname=folder, recursive=True)

    package_report = verify_submission_package(temporary)
    package_report["package"] = str(output)
    report_file = staging / "package_report.json"
    write_json(report_file, package_report)
    with temporary.open("ab") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    # Replace the archive last so a report write failure preserves the old package.
    os.replace(report_file, workspace / "logs" / "package_report.json")
    os.replace(temporary, output)
    print(json.dumps(package_report, ensure_ascii=False, indent=2))
    return output
