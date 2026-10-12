from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_training_data import _image_record, audit_training, duplicate_groups, fold_content_overlap


class TrainingAuditTests(unittest.TestCase):
    def test_decoded_duplicates_with_different_files_and_conflicting_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "data" / "raw"
            raw.mkdir(parents=True)
            image = Image.new("RGB", (16, 12), "gray")
            image.save(raw / "a.png")
            image.save(raw / "b.bmp")
            row = {"image": "a.png", "questionCategory": "桥梁", "bridgeName": "留出桥",
                   "defectType": "完好", "defectDescription": "无病害", "ratingScale(1-5)": ""}
            first = _image_record(raw, row)
            second = _image_record(raw, {**row, "image": "b.bmp", "bridgeName": "其他桥",
                                        "defectType": "裂缝", "ratingScale(1-5)": "2"})
            self.assertNotEqual(first["file_sha256"], second["file_sha256"])
            self.assertEqual(first["pixel_sha256"], second["pixel_sha256"])
            group = duplicate_groups([first, second], "pixel_sha256")[0]
            self.assertTrue(group["conflicting_labels"])
            self.assertTrue(group["conflicting_ratings"])
            self.assertEqual(len(fold_content_overlap([row], [{**row, "image": "b.bmp"}], [first, second])), 1)

    def test_missing_and_corrupt_images_are_reported_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            path = raw / "bad.jpg"
            path.write_bytes(b"not an image")
            row = {"image": "bad.jpg", "questionCategory": "轨道", "defectType": "完好"}
            self.assertIn("error", _image_record(raw, row))
            self.assertIn("error", _image_record(raw, {**row, "image": "missing.jpg"}))
            self.assertEqual(path.read_bytes(), b"not an image")
            with self.assertRaisesRegex(ValueError, "outside data/raw"):
                _image_record(raw, {**row, "image": "../outside.jpg"})

    def test_audit_does_not_read_test_images_and_checks_whole_bridge_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data" / "raw"
            raw.mkdir(parents=True)
            image = Image.new("RGB", (16, 12), "gray")
            rows = []
            for name, category, bridge in (("heldout.png", "桥梁", "留出桥"),
                                           ("fit.png", "桥梁", "另一桥"),
                                           ("track.png", "轨道", "")):
                image.save(raw / name)
                rows.append({"image": name, "filename": name, "questionCategory": category,
                             "bridgeName": bridge, "defectType": "完好", "defectDescription": "无病害",
                             "ratingScale(1-5)": ""})
            report = audit_training(root, rows, ["留出桥"], sample_size=2, seed=17)
            self.assertEqual(report["summary"]["readable_training_images"], 3)
            self.assertEqual(report["summary"]["pixel_duplicate_group_count"], 1)
            self.assertEqual(report["summary"]["folds_with_evaluated_content_overlap"], 1)
            self.assertEqual(len(report["folds"]["留出桥"]["all_excluded_content_overlap"]), 1)
            self.assertFalse((raw / "初赛测试集").exists())
            self.assertFalse((root / "result").exists())


if __name__ == "__main__":
    unittest.main()
