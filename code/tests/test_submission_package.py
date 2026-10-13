from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track4_agent.validation import build_submission_package, verify_submission_package


class SubmissionPackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for folder in ("code", "design", "result", "logs"):
            (self.root / folder).mkdir()
        for name in ("run_pipeline.py", "README.md", "MODEL.md", "requirements.txt"):
            (self.root / "code" / name).write_text("fixture", encoding="utf-8")
        (self.root / "design" / "AI智能体设计方案.docx").write_bytes(b"design fixture")
        result = b"[]"
        (self.root / "result" / "result.json").write_bytes(result)
        self.provenance = {"result_sha256": hashlib.sha256(result).hexdigest(),
                           "source_commit": "result-source"}
        (self.root / "logs" / "submission_provenance.json").write_text(
            json.dumps(self.provenance), encoding="utf-8")
        (self.root / "logs" / "raw_responses.jsonl").write_text("", encoding="utf-8")
        processed = self.root / "data" / "processed"
        processed.mkdir(parents=True)
        for name, text in (("train_manifest.jsonl", ""), ("test_manifest.jsonl", ""),
                           ("label_lexicon.json", '{"categories":{}}')):
            (processed / name).write_text(text, encoding="utf-8")
        self.output = self.root / "track4_submission.tar.gz"

    def rewrite_archive(self, transform):
        with tarfile.open(self.output, "r:gz") as archive:
            entries = [(member, archive.extractfile(member).read() if member.isfile() else None)
                       for member in archive.getmembers()]
        with tarfile.open(self.output, "w:gz") as archive:
            for member, payload in entries:
                payload = transform(member.name, payload)
                if payload is not None:
                    member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload) if payload is not None else None)

    def test_complete_package_has_verified_hashes_and_keeps_existing_staging(self):
        staging = self.root / "submission_staging"
        staging.mkdir()
        sentinel = staging / "keep.txt"
        sentinel.write_text("unrelated", encoding="utf-8")
        self.assertEqual(build_submission_package(self.root), self.output)
        report = verify_submission_package(self.output)
        self.assertTrue(report["all_file_hashes_verified"])
        self.assertEqual(report["source_commit"], "result-source")
        self.assertEqual(report["roots"], ["code", "design", "result"])
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "unrelated")
        self.assertFalse(list(self.root.glob("track4-package-*")))
        saved = json.loads((self.root / "logs" / "package_report.json").read_text())
        self.assertEqual(saved["package_sha256"], report["package_sha256"])

    def test_interrupted_archive_write_preserves_previous_package(self):
        self.output.write_bytes(b"previous verified package")
        with patch("track4_agent.validation.tarfile.open", side_effect=OSError("interrupted")):
            with self.assertRaisesRegex(OSError, "interrupted"):
                build_submission_package(self.root)
        self.assertEqual(self.output.read_bytes(), b"previous verified package")
        self.assertFalse(list(self.root.glob("track4-package-*")))

    def test_failed_readback_preserves_previous_package(self):
        self.output.write_bytes(b"previous verified package")
        with patch("track4_agent.validation.verify_submission_package", side_effect=ValueError("bad archive")):
            with self.assertRaisesRegex(ValueError, "bad archive"):
                build_submission_package(self.root)
        self.assertEqual(self.output.read_bytes(), b"previous verified package")

    def test_changed_result_is_rejected_before_replacing_package(self):
        self.output.write_bytes(b"previous verified package")
        (self.root / "result" / "result.json").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "differs from the registered"):
            build_submission_package(self.root)
        self.assertEqual(self.output.read_bytes(), b"previous verified package")

    def test_report_write_failure_preserves_previous_package(self):
        self.output.write_bytes(b"previous verified package")
        replace = os.replace
        def fail_report(source, destination):
            if Path(destination).name == "package_report.json":
                raise OSError("report write failed")
            return replace(source, destination)
        with patch("track4_agent.validation.os.replace", side_effect=fail_report):
            with self.assertRaisesRegex(OSError, "report write failed"):
                build_submission_package(self.root)
        self.assertEqual(self.output.read_bytes(), b"previous verified package")

    def test_missing_raw_log_is_rejected(self):
        (self.root / "logs" / "raw_responses.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "Missing package files"):
            build_submission_package(self.root)
        self.assertFalse(self.output.exists())

    def test_changed_archived_file_is_detected(self):
        build_submission_package(self.root)
        self.rewrite_archive(lambda name, data: b"changed" if name == "code/README.md" else data)
        with self.assertRaisesRegex(ValueError, "contents differ"):
            verify_submission_package(self.output)

    def test_manifest_cannot_change_result_provenance(self):
        build_submission_package(self.root)
        def transform(name, data):
            if name == "code/logs/submission_manifest.json":
                value = json.loads(data)
                value["result_source_commit"] = "wrong"
                return json.dumps(value).encode()
            return data
        self.rewrite_archive(transform)
        with self.assertRaisesRegex(ValueError, "registered provenance"):
            verify_submission_package(self.output)

    def test_truncated_gzip_trailer_is_detected(self):
        build_submission_package(self.root)
        self.output.write_bytes(self.output.read_bytes()[:-5])
        with self.assertRaises((EOFError, OSError)):
            verify_submission_package(self.output)

    def test_duplicate_and_parent_path_members_are_rejected(self):
        for name in ("code/README.md", "code/../escape.txt", "code/./README.md"):
            with self.subTest(name=name):
                build_submission_package(self.root)
                with tarfile.open(self.output, "r:gz") as archive:
                    entries = [(member, archive.extractfile(member).read() if member.isfile() else None)
                               for member in archive.getmembers()]
                with tarfile.open(self.output, "w:gz") as archive:
                    for member, payload in entries:
                        archive.addfile(member, io.BytesIO(payload) if payload is not None else None)
                    member = tarfile.TarInfo(name)
                    member.size = 1
                    archive.addfile(member, io.BytesIO(b"x"))
                with self.assertRaisesRegex(ValueError, "Unsafe or duplicate"):
                    verify_submission_package(self.output)


if __name__ == "__main__":
    unittest.main()
