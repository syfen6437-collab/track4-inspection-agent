"""Audit saved baseline emissions without generating or changing predictions."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import tarfile

from track4_agent.data import OUTPUT_KEYS
from track4_agent.inference import _parse_presence
from track4_agent.model import sanitize_prediction, scene_key
from track4_agent.validation import (
    _reject_constant, _unique_object, verify_submission_package,
)


def _json(text):
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def replay_results(manifest, lexicon, results, emissions):
    """Support the registered first-pass baseline; refuse other round policies."""
    if not all(isinstance(rows, list) for rows in (manifest, results, emissions)):
        raise ValueError("Manifest, results and emissions must be arrays")
    if not manifest or not (len(manifest) == len(results) == len(emissions)):
        raise ValueError("Manifest, result and raw-response counts differ or are empty")

    def index(rows, key):
        mapping = {}
        for row in rows:
            value = row.get(key) if isinstance(row, dict) else None
            if not isinstance(value, str) or not value.strip() or value in mapping:
                raise ValueError(f"Empty or duplicate {key}")
            mapping[value] = row
        return mapping

    inputs = index(manifest, "id")
    expected_names = index(manifest, "filename")
    predictions = index(results, "filename")
    raw_by_id = index(emissions, "id")
    if inputs.keys() != raw_by_id.keys() or expected_names.keys() != predictions.keys():
        raise ValueError("Raw-response IDs or result filenames differ from the manifest")

    categories = Counter()
    for item in manifest:
        records = raw_by_id[item["id"]].get("records")
        if item.get("questionCategory") not in {"桥梁", "轨道"}:
            raise ValueError("Unsupported questionCategory")
        expected_rounds = [0, 1] if item["questionCategory"] == "桥梁" else [1]
        if (not isinstance(records, list) or any(not isinstance(r, dict) for r in records)
                or [r.get("round") for r in records] != expected_rounds):
            raise ValueError("Unsupported model rounds; this audit supports the first-pass baseline only")
        if expected_rounds == [0, 1]:
            presence = records[0]
            if not isinstance(presence.get("raw"), str) or _parse_presence(presence["raw"]) != presence.get("presence"):
                raise ValueError(f"Presence cache differs from raw text: {item['id']}")
        record = records[-1]
        if not isinstance(record.get("raw"), str):
            raise ValueError("Raw model response must be text")
        normalized, valid = sanitize_prediction(
            record["raw"], lexicon, item["questionCategory"], scene_key(item),
        )
        if record.get("prediction") != normalized or record.get("valid") is not valid:
            raise ValueError(f"Prediction cache differs from raw text: {item['id']}")
        expected = {
            "questionCategory": item["questionCategory"],
            "bridgeName": item.get("bridgeName", ""),
            "defectLocation": item.get("defectLocation", ""),
            "filename": item["filename"],
            "defectType": normalized["defectType"],
            "defectDescription": normalized["defectDescription"],
            "ratingScale(1-5)": normalized["ratingScale"],
        }
        actual = predictions[item["filename"]]
        if list(actual) != OUTPUT_KEYS or actual != expected:
            raise ValueError(f"Formal result differs from raw model output: {item['id']}")
        categories[item["questionCategory"]] += 1
    return {"valid": True, "replayed_count": len(manifest),
            "category_counts": dict(categories), "model_loaded": False,
            "predictions_written": False, "scope": "saved first-pass emissions; not a new GPU inference"}


def replay_package(path):
    integrity = verify_submission_package(path)
    with tarfile.open(path, "r:gz") as archive:
        def read(name):
            with archive.extractfile(name) as handle:
                return handle.read().decode("utf-8-sig")
        report = replay_results(
            [_json(line) for line in read("code/logs/test_manifest.jsonl").splitlines() if line.strip()],
            _json(read("code/logs/label_lexicon.json")),
            _json(read("result/result.json")),
            [_json(line) for line in read("code/logs/raw_responses.jsonl").splitlines() if line.strip()],
        )
    return {**report, "package_sha256": integrity["package_sha256"],
            "result_sha256": integrity["result_sha256"],
            "result_source_commit": integrity["source_commit"],
            "engineering_commit": integrity["engineering_commit"]}


def main():
    parser = argparse.ArgumentParser(description="无需 GPU 或图片回放已登记正式结果的模型日志")
    parser.add_argument("--archive", type=Path,
                        default=Path(__file__).resolve().parents[1] / "track4_submission.tar.gz")
    args = parser.parse_args()
    report = replay_package(args.archive)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
