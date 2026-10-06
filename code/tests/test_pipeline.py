from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from track4_agent.model import build_checklist_prompt, candidate_labels, checklist_prediction, crop_montage, sanitize_prediction
from track4_agent.validation import validate_result


LEXICON = {
    "categories": {
        "桥梁": {
            "allowed_labels": ["完好", "裂缝"],
            "label_specs": {
                "完好": {"count": 10, "rating_distribution": {"": 10}, "default_rating": ""},
                "裂缝": {"count": 3, "rating_distribution": {"2": 3}, "default_rating": "2"},
            },
        }
    }
}


class PipelineTests(unittest.TestCase):
    def test_crop_montage_contains_four_tiles(self) -> None:
        montage = crop_montage(Image.new("RGB", (800, 600), "gray"), 128)
        self.assertEqual(montage.size, (256, 256))

    def test_sanitize_prediction_normalizes_rating(self) -> None:
        prediction, valid = sanitize_prediction(
            '{"defectType":"裂缝","defectDescription":"梁底可见细裂缝",'
            '"ratingScale":"5","confidence":0.8,"evidence":"细裂缝"}',
            LEXICON,
            "桥梁",
        )
        self.assertTrue(valid)
        self.assertEqual(prediction["defectType"], "裂缝")
        self.assertEqual(prediction["ratingScale"], "2")

    def test_checklist_prediction_maps_selected_atomic_labels(self) -> None:
        lexicon = {
            "categories": {
                "桥梁": {
                    "allowed_labels": ["完好", "裂缝", "裂缝、破损"],
                    "label_specs": {
                        "完好": {"count": 10, "rating_distribution": {"": 10}, "default_rating": ""},
                        "裂缝": {"count": 3, "rating_distribution": {"2": 3}, "default_rating": "2"},
                        "裂缝、破损": {"count": 1, "rating_distribution": {"3": 1}, "default_rating": "3"},
                    },
                    "scene_label_counts": {"bottom": {"裂缝": 2, "裂缝、破损": 1}},
                }
            }
        }
        prediction, valid = checklist_prediction(
            '{"selected":["裂缝","破损"],"description":"梁底可见裂缝和破损","confidence":0.8}',
            lexicon,
            "桥梁",
            "bottom",
        )
        self.assertTrue(valid)
        self.assertEqual(prediction["defectType"], "裂缝、破损")

    def test_validate_result_accepts_exact_schema(self) -> None:
        manifest = [
            {
                "questionCategory": "桥梁",
                "bridgeName": "示例桥",
                "defectLocation": "",
                "filename": "a.jpg",
            }
        ]
        result = [
            {
                "questionCategory": "桥梁",
                "bridgeName": "示例桥",
                "defectLocation": "",
                "filename": "a.jpg",
                "defectType": "完好",
                "defectDescription": "未见明显病害",
                "ratingScale(1-5)": "",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "result").mkdir()
            (workspace / "logs").mkdir()
            path = workspace / "result" / "result.json"
            path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            report = validate_result(workspace, manifest, LEXICON, path)
            self.assertTrue(report["valid"])


if __name__ == "__main__":
    unittest.main()
