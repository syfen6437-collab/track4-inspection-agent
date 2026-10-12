from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_dedup_heads import deduplicate_fitting, run_experiment
from track4_agent.vision_review import digest_json


class DedupHeadsTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"image": name, "questionCategory": "桥梁", "defectType": label}
                     for name, label in (("b.jpg", "完好"), ("a.jpg", "完好"),
                                         ("c.jpg", "完好"), ("d.jpg", "裂缝"), ("e.jpg", "裂缝"))]
        self.records = [{"image": row["image"], "pixel_sha256": digest}
                        for row, digest in zip(self.rows, ("same", "same", "conflict", "conflict", "unique"))]

    def test_same_label_duplicates_have_a_stable_representative_without_relabeling(self):
        rows, policy = deduplicate_fitting(self.rows, self.records, exclude_conflicts=False)
        self.assertEqual([row["image"] for row in rows], ["a.jpg", "c.jpg", "d.jpg", "e.jpg"])
        self.assertEqual(policy["removed_duplicate_count"], 1)
        self.assertEqual(policy["excluded_conflicting_images"], [])
        self.assertEqual(self.rows[0]["image"], "b.jpg")

    def test_conflicting_groups_can_be_excluded_only_from_experimental_fitting(self):
        rows, policy = deduplicate_fitting(self.rows, self.records, exclude_conflicts=True)
        self.assertEqual([row["image"] for row in rows], ["a.jpg", "e.jpg"])
        self.assertEqual(policy["excluded_conflicting_images"], ["c.jpg", "d.jpg"])
        self.assertEqual(len(self.rows), 5)
        self.assertEqual(self.rows[3]["defectType"], "裂缝")

    def test_unreadable_content_cannot_be_used_for_fitting(self):
        self.records[0]["error"] = "decode failed"
        with self.assertRaisesRegex(ValueError, "Unreadable audited"):
            deduplicate_fitting(self.rows, self.records, exclude_conflicts=True)

    def test_audit_manifest_mismatch_is_rejected_before_fitting(self):
        with self.assertRaisesRegex(ValueError, "audit does not match"):
            run_experiment(Path("."), self.rows, {}, {"training_manifest_sha256": "wrong"}, [], ["桥"], 2)

    def test_evaluated_pixel_overlap_is_rejected_before_training(self):
        audit = {"training_manifest_sha256": digest_json(self.rows), "summary": {"image_error_count": 0},
                 "image_records": self.records}
        fitting = Mock(side_effect=AssertionError("Must not train on leaked data"))
        with patch("evaluate_dedup_heads._make_holdout", return_value=([self.rows[0]], [self.rows[1]], {})), \
                patch("evaluate_dedup_heads.fit_knn", fitting):
            with self.assertRaisesRegex(ValueError, "content overlaps"):
                run_experiment(Path("."), self.rows, {}, audit, [None] * len(self.rows), ["桥"], 2)
        fitting.assert_not_called()


if __name__ == "__main__":
    unittest.main()
