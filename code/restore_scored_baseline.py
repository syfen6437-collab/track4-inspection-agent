from __future__ import annotations

"""Restore a complete previously model-generated result from verified Git history."""

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from track4_agent.data import load_assets, write_json
from track4_agent.validation import validate_result


def main() -> None:
    specification = json.loads((CODE / "config" / "submission_baseline.json").read_text(encoding="utf-8"))
    revision = specification["source_revision"]
    payload = subprocess.check_output(["git", "show", f"{revision}:result/result.json"], cwd=ROOT)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != specification["result_sha256"]:
        raise ValueError("Historical baseline hash differs; official result was not changed")
    source_commit = subprocess.check_output(["git", "rev-parse", revision], cwd=ROOT).decode().strip()
    _, manifest, lexicon = load_assets(ROOT)
    verified = ROOT / "runs" / "scored_baseline" / "result.json"
    verified.parent.mkdir(parents=True, exist_ok=True)
    verified.write_bytes(payload)
    validate_result(ROOT, manifest, lexicon, result_path=verified,
                    report_path=verified.parent / "validation.json")
    output = ROOT / "result" / "result.json"
    previous = output.read_bytes() if output.exists() else b""
    previous_digest = hashlib.sha256(previous).hexdigest() if previous else None
    if previous and previous != payload:
        backup = ROOT / "runs" / "withdrawn_submission" / previous_digest / "result.json"
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(previous)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    write_json(ROOT / "logs" / "submission_provenance.json", {
        "status": "restored_scored_baseline",
        "source_commit": source_commit,
        "result_sha256": digest,
        "model_id": specification["model_id"],
        "official_feedback": specification["official_feedback"],
        "result_count": len(json.loads(payload)),
        "category_counts": dict(Counter(row["questionCategory"] for row in json.loads(payload))),
        "withdrawn_candidate": specification["withdrawn_candidate"],
        "candidate_changes_applied": 0,
        "new_score_verified": False,
        "policy": "Restore the entire historical model output, not selected test answers",
    })
    print(json.dumps({"restored": str(output), "source_commit": source_commit,
                      "result_sha256": digest, "previous_sha256": previous_digest,
                      "new_score_verified": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
