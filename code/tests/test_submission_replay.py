from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_submission import _json, replay_results
from track4_agent.model import sanitize_prediction


class SubmissionReplayTests(unittest.TestCase):
    def setUp(self):
        self.lexicon = {"categories": {category: {
            "allowed_labels": ["完好", "裂缝"],
            "label_specs": {"完好": {"count": 10, "default_rating": "", "rating_distribution": {"": 10}},
                            "裂缝": {"count": 2, "default_rating": "2", "rating_distribution": {"2": 2}}},
        } for category in ("桥梁", "轨道")}}
        self.manifest, self.results, self.emissions = [], [], []
        for i, category in enumerate(("桥梁", "轨道")):
            item = {"id": f"group/image{i}.jpg", "filename": f"image{i}.jpg", "questionCategory": category,
                    "bridgeName": "示例桥" if i == 0 else "", "defectLocation": ""}
            raw = json.dumps({"defectType": "裂缝", "defectDescription": "可见裂缝", "ratingScale": "2",
                              "confidence": 0.8, "evidence": "细裂缝"}, ensure_ascii=False)
            normalized, valid = sanitize_prediction(raw, self.lexicon, category)
            records = [{"round": 1, "raw": raw, "prediction": normalized, "valid": valid}]
            if category == "桥梁":
                records.insert(0, {"round": 0, "raw": '{"has_defect":true}', "presence": {"has_defect": True}})
            self.manifest.append(item)
            self.emissions.append({"id": item["id"], "records": records})
            self.results.append({**{k: item[k] for k in ("questionCategory", "bridgeName", "defectLocation", "filename")},
                                 "defectType": "裂缝", "defectDescription": "可见裂缝", "ratingScale(1-5)": "2"})

    def run_replay(self):
        return replay_results(self.manifest, self.lexicon, self.results, self.emissions)

    def test_two_domains_replay_without_model_or_predictions_written(self):
        before = copy.deepcopy((self.manifest, self.results, self.emissions))
        report = self.run_replay()
        self.assertEqual(report["replayed_count"], 2)
        self.assertFalse(report["model_loaded"])
        self.assertFalse(report["predictions_written"])
        self.assertEqual(before, (self.manifest, self.results, self.emissions))

    def test_altered_result_and_metadata_are_rejected(self):
        for key, value in (("defectType", "完好"), ("bridgeName", "其他桥")):
            with self.subTest(key=key):
                original = self.results[0][key]
                self.results[0][key] = value
                with self.assertRaisesRegex(ValueError, "Formal result differs"):
                    self.run_replay()
                self.results[0][key] = original

    def test_altered_cache_is_rejected_even_if_final_result_is_unchanged(self):
        self.emissions[0]["records"][-1]["prediction"]["confidence"] = 1.0
        with self.assertRaisesRegex(ValueError, "Prediction cache differs"):
            self.run_replay()

    def test_duplicate_raw_id_is_rejected(self):
        self.emissions[1]["id"] = self.emissions[0]["id"]
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            self.run_replay()

    def test_additional_review_round_is_refused(self):
        self.emissions[1]["records"].append({"round": 2})
        with self.assertRaisesRegex(ValueError, "Unsupported model rounds"):
            self.run_replay()

    def test_missing_emission_is_rejected(self):
        self.emissions.pop()
        with self.assertRaisesRegex(ValueError, "counts differ"):
            self.run_replay()

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants(self):
        for text in ('{"id":1,"id":2}', '{"confidence":NaN}'):
            with self.assertRaises(ValueError):
                _json(text)


if __name__ == "__main__":
    unittest.main()
