"""Have local Qwen independently review an isolated track adapter candidate.

The adapter labels are never included in the prompt. A changed track row is
accepted only when Qwen's own structured label matches the adapter consensus;
then Qwen supplies the description and rating. Otherwise the model-generated
baseline row is retained. Bridge rows are always copied unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from run_pipeline import load_client, load_config
from track4_agent.data import OUTPUT_KEYS, load_assets, write_json
from track4_agent.inference import _neighbor_index
from track4_agent.validation import validate_result
from track4_agent.vision_review import candidate_path, index_unique, review_image, sha256_file

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))


def _load_rows(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Result must be a JSON array: {path}")
    return value


def _validate_candidate(base, candidate, manifest, lexicon):
    by_name = index_unique(manifest, "filename")
    if len(base) != len(candidate) or len(candidate) != len(manifest):
        raise ValueError("Base, candidate and manifest counts differ")
    for before, after in zip(base, candidate, strict=True):
        if list(before) != OUTPUT_KEYS or list(after) != OUTPUT_KEYS:
            raise ValueError(f"Seven-field schema differs: {before.get('filename')}")
        if before["filename"] != after["filename"]:
            raise ValueError("Candidate row order or filename differs")
        item = by_name[before["filename"]]
        for key in ("questionCategory", "bridgeName", "defectLocation", "filename"):
            if after[key] != item.get(key, ""):
                raise ValueError(f"Candidate metadata differs for {after['filename']}: {key}")
        if before["questionCategory"] == "桥梁" and before != after:
            raise ValueError(f"Bridge row changed in track candidate: {before['filename']}")
        allowed = lexicon["categories"][after["questionCategory"]]["allowed_labels"]
        if after["defectType"] not in allowed or after["ratingScale(1-5)"] not in {"", "1", "2", "3", "4", "5"}:
            raise ValueError(f"Illegal candidate label/rating: {after['filename']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CODE / "config" / "qwen3-vl-4b-offload.json")
    parser.add_argument("--base-result", type=Path, default=ROOT / "result" / "result.json")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--adapter-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    output = candidate_path(ROOT, args.output)
    audit_path = candidate_path(ROOT, args.audit)
    if output.exists() or audit_path.exists():
        raise ValueError("Output exists; use a new runs directory")
    if args.limit is not None and args.limit <= 0:
        parser.error("Limit must be positive")
    for name in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "MODELSCOPE_OFFLINE"):
        os.environ[name] = "1"
    train, manifest, lexicon = load_assets(ROOT)
    base = _load_rows(args.base_result.resolve())
    candidate = _load_rows(args.candidate.resolve())
    _validate_candidate(base, candidate, manifest, lexicon)
    adapter_audit = json.loads(args.adapter_audit.read_text(encoding="utf-8"))
    decisions = index_unique(adapter_audit.get("records", []), "filename")
    changed = [row for row in candidate if row["questionCategory"] == "轨道"
               and row["defectType"] != index_unique(base, "filename")[row["filename"]]["defectType"]]
    if args.limit is not None:
        changed = changed[:args.limit]
    expected_changed = {row["filename"] for row in candidate if row["questionCategory"] == "轨道"
                        and row["defectType"] != index_unique(base, "filename")[row["filename"]]["defectType"]}
    if args.limit is None and set(decisions) != {row["filename"] for row in candidate}:
        raise ValueError("Adapter audit does not cover every manifest row")
    config = load_config(args.config.resolve())
    client = load_client(config)
    neighbors = _neighbor_index(manifest)
    raw_dir = ROOT / "data" / "raw"
    base_by_name = index_unique(base, "filename")
    candidate_by_name = index_unique(candidate, "filename")
    reviewed = {}
    for index, row in enumerate(changed, start=1):
        item = next(item for item in manifest if item["filename"] == row["filename"])
        adapter = decisions[row["filename"]]
        vision = {"defectType": row["defectType"], "confidence": max(adapter.get("adapter_confidences", [0.0]))}
        review = review_image(client, item, base_by_name[row["filename"]], vision, lexicon, config,
                              raw_dir / item["image"], raw_dir, neighbors)
        reviewed[row["filename"]] = review
        print(f"[qwen-review] {index}/{len(changed)} {row['filename']} valid={review['valid']}", flush=True)
    final = [dict(row) for row in base]
    accepted = 0
    audit_rows = []
    for row in final:
        name = row["filename"]
        adapter = decisions.get(name, {})
        review = reviewed.get(name)
        before = dict(row)
        if review is not None and review.get("valid") and review.get("prediction"):
            prediction = review["prediction"]
            agree = prediction["defectType"] == candidate_by_name[name]["defectType"]
            if agree and prediction["defectType"] != "完好":
                row["defectType"] = prediction["defectType"]
                row["defectDescription"] = prediction["defectDescription"]
                row["ratingScale(1-5)"] = prediction["ratingScale"]
                accepted += int(row != before)
                reason = "Qwen independent label agrees with two adapters"
            else:
                reason = "Qwen label disagrees with adapter consensus"
        elif review is not None:
            reason = "Qwen review invalid or contradictory"
        else:
            reason = "not selected for Qwen review"
        audit_rows.append({"filename": name, "questionCategory": row["questionCategory"],
                           "adapter_label": adapter.get("final_label", row["defectType"]),
                           "qwen_review": review, "accepted": row != before, "reason": reason})
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, final)
    write_json(audit_path, {
        "policy": "two-adapter consensus followed by independent Qwen label agreement",
        "base_result_sha256": sha256_file(args.base_result.resolve()),
        "adapter_candidate_sha256": sha256_file(args.candidate.resolve()),
        "adapter_audit_sha256": sha256_file(args.adapter_audit.resolve()),
        "reviewed_count": len(changed), "accepted_count": accepted,
        "bridge_unchanged": all(a == b for a, b in zip(base, final, strict=True)
                                  if a["questionCategory"] == "桥梁"),
        "records": audit_rows,
    })
    validate_result(ROOT, manifest, lexicon, result_path=output,
                    report_path=audit_path.with_name("validation_report.json"))
    print(json.dumps({"output": str(output), "audit": str(audit_path),
                      "reviewed_count": len(changed), "accepted_count": accepted}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
