from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import review_server


class ReviewStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.qa = self.root / "qa"
        self.path = self.qa / "review_annotations.jsonl"
        for name, value in (("QA_DIR", self.qa), ("QA_FILE", self.path)):
            replacement = patch.object(review_server, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)

    def test_confirmed_status_and_optional_empty_note_survive_reload(self):
        saved = review_server.record_review("training/a.jpg", "confirmed", "", "")
        self.assertEqual(review_server.load_state()["training/a.jpg"], saved)
        self.assertEqual(saved["status"], "confirmed")
        self.assertEqual(saved["note"], "")
        self.assertEqual(saved["reviewedLabel"], "")

    def test_replace_failure_preserves_previous_records_and_cleans_temporary(self):
        review_server.record_review("training/a.jpg", "confirmed", "裂缝", "原始证据")
        before = self.path.read_bytes()
        with patch("review_server.os.replace", side_effect=OSError("interrupted replace")):
            with self.assertRaisesRegex(OSError, "interrupted replace"):
                review_server.record_review("training/b.jpg", "needs_review", "", "")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.qa.glob("*.tmp")), [])

    def test_concurrent_saves_keep_both_records(self):
        ready = threading.Barrier(2)

        def save(item_id):
            ready.wait(timeout=5)
            return review_server.record_review(item_id, "confirmed", "完好", "")

        with ThreadPoolExecutor(max_workers=2) as executor:
            saved = list(executor.map(save, ("training/a.jpg", "training/b.jpg")))
        self.assertEqual(review_server.load_state(), {row["id"]: row for row in saved})

    def test_updates_preserve_other_records_and_only_write_qa(self):
        official = self.root / "result" / "result.json"
        official.parent.mkdir()
        official.write_bytes(b"unchanged official predictions")
        review_server.record_review("training/a.jpg", "confirmed", "完好", "")
        review_server.record_review("training/b.jpg", "needs_review", "", "证据不足")
        saved = review_server.record_review("training/a.jpg", "needs_review", "", "重新复核")
        state = review_server.load_state()
        self.assertEqual(state["training/a.jpg"], saved)
        self.assertEqual(state["training/b.jpg"]["note"], "证据不足")
        self.assertEqual(official.read_bytes(), b"unchanged official predictions")
        self.assertEqual(len([json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]), 2)


if __name__ == "__main__":
    unittest.main()
