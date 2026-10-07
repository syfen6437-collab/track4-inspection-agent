from __future__ import annotations

"""Revalidate an existing visual-candidate result without another model call."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from apply_visual_candidate import _strict_review
from track4_agent.data import load_assets, write_json
from track4_agent.vision_review import candidate_path, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    args = parser.parse_args()

    output = candidate_path(ROOT, args.output)
    output_audit = candidate_path(ROOT, args.output_audit)
    result = json.loads(args.input.resolve().read_text(encoding="utf-8"))
    audit_doc = json.loads(args.audit.resolve().read_text(encoding="utf-8"))
    _, _, lexicon = load_assets(ROOT)
    by_key = {(row["questionCategory"], row["filename"]): row for row in result}
    reverted: list[dict[str, Any]] = []
    checked = 0
    for record in audit_doc.get("records", []):
        if not record.get("accepted"):
            continue
        key = (record["original"]["questionCategory"], record["filename"])
        current = by_key.get(key)
        if current is None:
            raise ValueError(f"Missing result row for {key}")
        category = record["original"]["questionCategory"]
        fallback = str(record.get("visual", {}).get("defectType", ""))
        raw = str(record.get("qwen", {}).get("raw", ""))
        parsed = _strict_review(raw, set(lexicon["categories"][category]["allowed_labels"]), fallback)
        checked += 1
        if parsed is None or parsed["defectType"] != fallback:
            original = dict(record["original"])
            by_key[key] = original
            reverted.append({
                "questionCategory": category,
                "filename": record["filename"],
                "candidate": fallback,
                "reason": "qwen_evidence_contradicts_candidate_or_label",
                "raw": raw,
                "original": original,
                "current": current,
            })

    ordered = [by_key[(row["questionCategory"], row["filename"])] for row in result]
    write_json(output, ordered)
    write_json(output_audit, {
        "context": {
            "policy": "revalidate saved local-Qwen evidence; revert only explicitly contradicted visual candidates",
            "input_sha256": sha256_file(args.input.resolve()),
            "source_audit_sha256": sha256_file(args.audit.resolve()),
        },
        "checked": checked,
        "reverted": len(reverted),
        "records": reverted,
    })
    print(json.dumps({"output": str(output), "checked": checked,
                      "reverted": len(reverted), "audit": str(output_audit)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
