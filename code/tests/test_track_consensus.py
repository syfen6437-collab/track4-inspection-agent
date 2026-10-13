from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_track_consensus import common_holdout
from review_track_candidate import _checked_adapter_audit
from track4_agent.track_consensus import adapter_decision, merge_track_reviews


class TrackConsensusTests(unittest.TestCase):
    def setUp(self):
        self.item = {"id": "轨道/a.jpg", "filename": "a.jpg", "questionCategory": "轨道", "bridgeName": "", "defectLocation": ""}
        self.base = {"questionCategory": "轨道", "bridgeName": "", "defectLocation": "", "filename": "a.jpg",
                     "defectType": "破损", "defectDescription": "模型认为混凝土破损", "ratingScale(1-5)": ""}
        self.candidate = {**self.base, "defectType": "支座锈蚀"}
        self.lexicon = {"categories": {"轨道": {"allowed_labels": ["完好", "破损", "支座锈蚀"],
                            "label_specs": {l: {"default_rating": ""} for l in ("完好", "破损", "支座锈蚀")}}}}
        self.prediction = {"defectType": "支座锈蚀", "defectDescription": "支座表面有锈蚀", "ratingScale": "",
                           "confidence": 0.9, "evidence": "支座锈蚀"}

    def review(self, confidence=0.9):
        prediction = {**self.prediction, "confidence": confidence}
        raw = json.dumps(prediction, ensure_ascii=False)
        return {"id": self.item["id"], "filename": "a.jpg", "vision_candidate": "支座锈蚀", "vision_confidence": 0.8,
                "valid": True, "prediction": prediction, "raw": raw, "attempts": [raw]}

    def merge(self, reviews):
        return merge_track_reviews([self.base], [self.candidate], [self.item], reviews, self.lexicon)

    def test_agreement_copies_only_model_response_and_protects_metadata(self):
        final, audit = self.merge([self.review()])
        self.assertEqual(final[0]["defectType"], "支座锈蚀")
        self.assertEqual(final[0]["defectDescription"], self.prediction["defectDescription"])
        self.assertTrue(audit[0]["accepted"])
        self.assertEqual(final[0]["filename"], self.base["filename"])

    def test_low_or_zero_confidence_and_disagreement_keep_baseline(self):
        for confidence in (0, 0.54):
            final, audit = self.merge([self.review(confidence)])
            self.assertEqual(final, [self.base])
            self.assertFalse(audit[0]["accepted"])
        self.candidate["defectType"] = "破损"
        with self.assertRaisesRegex(ValueError, "unselected"):
            self.merge([self.review()])

    def test_edited_cached_prediction_or_attempt_is_rejected(self):
        for field, value in (("prediction", {**self.prediction, "confidence": 1}), ("attempts", ["different"])):
            review = self.review()
            review[field] = value
            with self.assertRaises(ValueError):
                self.merge([review])

    def test_adapter_selection_rejects_nonfinite_or_disagreeing_scores(self):
        first, second = {"label": "支座锈蚀", "confidence": 0.8}, {"label": "破损", "confidence": 0.9}
        self.assertFalse(adapter_decision(self.base, first, second)["accepted"])
        first["confidence"] = float("nan")
        with self.assertRaises(ValueError):
            adapter_decision(self.base, first, second)

    def test_audit_must_match_baseline_training_and_confidence_evidence(self):
        decision = adapter_decision(self.base, {"label": "支座锈蚀", "confidence": 0.8},
                                    {"label": "支座锈蚀", "confidence": 0.9})
        audit = {"base_result_sha256": "base", "training_manifest_sha256": "train", "threshold": 0.55,
                 "records": [decision]}
        _checked_adapter_audit(audit, [self.base], [self.candidate], "base", "train")
        for key, value in (("base_result_sha256", "wrong"), ("training_manifest_sha256", "wrong")):
            with self.assertRaises(ValueError):
                _checked_adapter_audit({**audit, key: value}, [self.base], [self.candidate], "base", "train")
        bad = copy.deepcopy(audit)
        bad["records"][0]["adapter_confidences"][1] = 0.01
        with self.assertRaises(ValueError):
            _checked_adapter_audit(bad, [self.base], [self.candidate], "base", "train")

    def test_common_holdout_excludes_fitting_and_model_selection(self):
        train = [{"image": "a", "questionCategory": "轨道"}, {"image": "b", "questionCategory": "轨道"}]
        reports = [{"split": {"test": ["a"], "fitting": ["b"], "validation": []}},
                   {"split": {"test": ["a", "b"], "fitting": [], "validation": []}}]
        self.assertEqual(common_holdout(train, reports), [train[0]])
        reports[1]["split"]["validation"] = ["a"]
        with self.assertRaisesRegex(ValueError, "selection"):
            common_holdout(train, reports)


if __name__ == "__main__":
    unittest.main()
