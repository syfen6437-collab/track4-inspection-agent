from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track4_agent.validation import validate_result


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "candidate.json"
        self.report = self.root / "validation.json"
        self.manifest = [{"questionCategory": "桥梁", "bridgeName": "示例桥（左幅）",
                          "defectLocation": "", "filename": "a.jpg"}]
        self.lexicon = {"categories": {"桥梁": {"allowed_labels": ["完好", "裂缝"]},
                                       "轨道": {"allowed_labels": ["完好", "裂缝"]}}}
        self.rows = [{**self.manifest[0], "defectType": "完好", "defectDescription": "未见病害",
                      "ratingScale(1-5)": ""}]

    def validate(self, rows=None):
        self.path.write_text(json.dumps(self.rows if rows is None else rows, ensure_ascii=False), encoding="utf-8")
        return validate_result(self.root, self.manifest, self.lexicon, self.path, self.report)

    def test_each_field_must_be_a_string(self):
        for field in self.rows[0]:
            for value in (None, 1, True, [], {}):
                with self.subTest(field=field, value=value):
                    rows = copy.deepcopy(self.rows)
                    rows[0][field] = value
                    with self.assertRaisesRegex(ValueError, "must be a string"):
                        self.validate(rows)

    def test_numeric_rating_is_not_accepted_as_a_string(self):
        self.rows[0]["ratingScale(1-5)"] = 2
        with self.assertRaisesRegex(ValueError, "ratingScale.*must be a string"):
            self.validate()

    def test_valid_but_wrong_metadata_is_rejected(self):
        for field, value in (("bridgeName", "示例桥（右幅）"), ("questionCategory", "轨道"),
                             ("defectLocation", "different location")):
            with self.subTest(field=field):
                rows = copy.deepcopy(self.rows)
                rows[0][field] = value
                with self.assertRaisesRegex(ValueError, "does not match input metadata"):
                    self.validate(rows)

    def test_duplicate_manifest_and_result_filenames_are_rejected(self):
        self.manifest *= 2
        with self.assertRaisesRegex(ValueError, "duplicate filenames"):
            self.validate(self.rows * 2)

    def test_rows_can_be_reordered_but_metadata_must_stay_with_filename(self):
        self.manifest.append({**self.manifest[0], "filename": "b.jpg", "bridgeName": "另一桥"})
        rows = self.rows + [{**self.rows[0], **self.manifest[1]}]
        self.assertTrue(self.validate(list(reversed(rows)))["valid"])
        rows[0]["bridgeName"], rows[1]["bridgeName"] = rows[1]["bridgeName"], rows[0]["bridgeName"]
        with self.assertRaisesRegex(ValueError, "does not match input metadata"):
            self.validate(rows)

    def test_duplicate_json_keys_and_nonfinite_constants_are_rejected(self):
        for text in ('[{"filename":"a.jpg","filename":"a.jpg"}]', '[{"confidence":NaN}]',
                     '[{"confidence":Infinity}]', '[{"confidence":-Infinity}]'):
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                before = self.path.read_bytes()
                with self.assertRaisesRegex(ValueError, "not valid UTF-8 JSON"):
                    validate_result(self.root, self.manifest, self.lexicon, self.path, self.report)
                self.assertEqual(self.path.read_bytes(), before)
                report = json.loads(self.report.read_text(encoding="utf-8"))
                self.assertFalse(report["valid"])
                self.assertGreater(report["error_count"], 0)

    def test_non_utf8_input_is_rejected_and_utf8_bom_remains_supported(self):
        text = json.dumps(self.rows, ensure_ascii=False)
        self.path.write_text(text, encoding="utf-16")
        with self.assertRaisesRegex(ValueError, "not valid UTF-8 JSON"):
            validate_result(self.root, self.manifest, self.lexicon, self.path, self.report)
        self.path.write_text(text, encoding="utf-8-sig")
        self.assertTrue(validate_result(self.root, self.manifest, self.lexicon, self.path, self.report)["valid"])


if __name__ == "__main__":
    unittest.main()
