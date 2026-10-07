from __future__ import annotations

"""Apply training-derived rating defaults to an isolated result candidate."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from track4_agent.data import OUTPUT_KEYS, load_assets, write_json
from track4_agent.vision_review import candidate_path, digest_json, index_unique, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "result" / "result.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    output = candidate_path(ROOT, args.output)
    audit_path = candidate_path(ROOT, args.audit)
    _, test, lexicon = load_assets(ROOT)
    source = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise ValueError("Input result must be a JSON array")
    by_filename = index_unique(source, "filename")
    expected = {row["filename"] for row in test}
    if set(by_filename) != expected:
        raise ValueError("Input result filenames differ from test manifest")
    calibrated, changes = [], []
    for item in test:
        row = dict(by_filename[item["filename"]])
        if set(row) != set(OUTPUT_KEYS):
            raise ValueError(f"Result schema mismatch: {item['filename']}")
        allowed = lexicon["categories"][item["questionCategory"]]["allowed_labels"]
        if row["defectType"] not in allowed:
            raise ValueError(f"Illegal defectType: {item['filename']}")
        specs = lexicon["categories"][item["questionCategory"]]["label_specs"][row["defectType"]]
        observed = specs.get("rating_distribution", {})
        old = row["ratingScale(1-5)"]
        new = old
        if observed and old not in observed:
            new = str(specs.get("default_rating", ""))
            row["ratingScale(1-5)"] = new
        calibrated.append(row)
        if old != new:
            changes.append({"filename": item["filename"], "questionCategory": item["questionCategory"],
                            "defectType": row["defectType"], "from": old, "to": new})
    context = {
        "policy": "training-label rating default only; no image or test label access",
        "input_sha256": sha256_file(args.input),
        "manifest_sha256": digest_json(test),
        "lexicon_sha256": digest_json(lexicon),
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    write_json(output, calibrated)
    write_json(audit_path, {
        "context": context, "changed": len(changes),
        "by_category": dict(Counter(row["questionCategory"] for row in changes)),
        "by_label": dict(Counter(row["defectType"] for row in changes)),
        "records": changes,
    })
    print(json.dumps({"output": str(output), "changed": len(changes),
                      "by_category": dict(Counter(row["questionCategory"] for row in changes))},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
