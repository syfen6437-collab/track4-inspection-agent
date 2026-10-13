"""Check the actual extracted package without touching the official result."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from track4_agent.validation import verify_submission_package


def main():
    archive = ROOT / "track4_submission.tar.gz"
    integrity = verify_submission_package(archive)
    commands = [
        ["code/run_pipeline.py", "--help"],
        ["code/run_pipeline.py", "package-check", "--archive", str(archive)],
        ["code/replay_submission.py", "--archive", str(archive)],
    ]
    checks = []
    with tempfile.TemporaryDirectory(prefix="track4-extracted-audit-") as folder:
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(folder, filter="data")
        for args in commands:
            completed = subprocess.run(
                [sys.executable, "-X", "utf8", *args], cwd=folder,
                capture_output=True, text=True, encoding="utf-8", timeout=30,
            )
            checks.append({"command": args[0:2], "returncode": completed.returncode,
                           "stderr": completed.stderr})
            if completed.returncode:
                raise RuntimeError(completed.stderr or completed.stdout)
        replay = json.loads(completed.stdout)
    report = {"valid": True, "package_sha256": integrity["package_sha256"],
              "engineering_commit": integrity["engineering_commit"],
              "engineering_dirty": integrity["engineering_dirty"],
              "extracted_command_checks": checks, "raw_replay": replay,
              "temporary_extract_removed": True}
    output = ROOT / "logs" / "extracted_package_check.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
