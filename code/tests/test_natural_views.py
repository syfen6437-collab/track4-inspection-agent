from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_visual_views import check_content_audit, label_metrics, natural_splits
from track4_agent.vision_review import digest_json


class NaturalViewsTests(unittest.TestCase):
    def rows(self):
        return [{"image": f"{category}/{i}.jpg", "questionCategory": category,
                 "bridgeName": bridge, "defectType": label} for i, (category, bridge, label) in enumerate([
            ("桥梁", "A（左幅）", "完好"), ("桥梁", "A（右幅）", "修补痕"), ("桥梁", "B", "剥落"),
            ("轨道", "", "完好"), ("轨道", "", "破损"), ("轨道", "", "钢筋锈蚀"), ("轨道", "", "破损")])]

    def test_random_track_selection_is_independent_of_target_labels(self):
        rows = self.rows()
        changed = copy.deepcopy(rows)
        for row in changed:
            row["defectType"] = "different target"
        first = natural_splits(rows, ["A", "B"], 2, 7)
        second = natural_splits(changed, ["A", "B"], 2, 7)
        self.assertEqual([x["image"] for x in first["轨道/random"][1]], [x["image"] for x in second["轨道/random"][1]])
        self.assertEqual(len(first), 3)

    def test_whole_bridge_and_track_queries_never_enter_fitting_rows(self):
        splits = natural_splits(self.rows(), ["A", "B"], 2, 7)
        self.assertEqual(len(splits["桥梁/A"][1]), 2)
        for fitting, queries in splits.values():
            self.assertFalse({r["image"] for r in fitting} & {r["image"] for r in queries})

    def test_invalid_partitions_and_duplicate_identifiers_fail(self):
        for groups, size in ((["A", "A"], 2), (["missing"], 2), (["A"], 0), (["A"], 4)):
            with self.assertRaises(ValueError):
                natural_splits(self.rows(), groups, size, 7)
        with self.assertRaises(ValueError):
            natural_splits(self.rows() + self.rows()[:1], ["A"], 2, 7)

    def test_label_only_metrics_do_not_claim_severity_accuracy(self):
        truth = {"questionCategory": "轨道", "defectType": "完好", "ratingScale(1-5)": ""}
        metrics = label_metrics([{"truth": truth, "prediction": truth}])
        self.assertNotIn("rating_accuracy", metrics)
        self.assertNotIn("rating_correct", metrics["category_metrics"]["轨道"])

    def test_content_audit_rejects_changed_images_and_cross_partition_pixels(self):
        rows = self.rows()
        splits = natural_splits(rows, ["A"], 2, 7)
        records = [{**row, "file_sha256": "file-digest", "pixel_sha256": row["image"]} for row in rows]
        audit = {"training_manifest_sha256": digest_json(rows), "image_records": records}
        with patch("compare_visual_views.sha256_file", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "differs from content audit"):
                check_content_audit(Path("fixture"), rows, splits, audit)
        with patch("compare_visual_views.sha256_file", return_value="file-digest"):
            check_content_audit(Path("fixture"), rows, splits, audit)
            records[0]["pixel_sha256"] = records[2]["pixel_sha256"]
            with self.assertRaisesRegex(ValueError, "crosses fitting and query"):
                check_content_audit(Path("fixture"), rows, splits, audit)


if __name__ == "__main__":
    unittest.main()
