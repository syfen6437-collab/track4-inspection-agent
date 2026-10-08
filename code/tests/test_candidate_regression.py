from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import apply_visual_candidate
from evaluate_vision_review import policy_records
from track4_agent.vision_review import (
    cached_reviews, evidence_contradicts, merge_reviews, review_image, strict_prediction,
)
from test_vision_review import LEXICON


class CandidateRegressionTests(unittest.TestCase):
    def setUp(self):
        self.item = {"id": "bridge/support.jpg", "image": "test/support.jpg", "filename": "支座.JPG",
                     "questionCategory": "桥梁", "bridgeName": "桥", "defectLocation": ""}
        self.base = {key: self.item[key] for key in
                     ("questionCategory", "bridgeName", "defectLocation", "filename")}
        self.base.update(defectType="完好", defectDescription="无明显病害")
        self.base["ratingScale(1-5)"] = ""
        self.vision = {"id": self.item["id"], "questionCategory": "桥梁", "filename": "支座.JPG",
                       "defectType": "渗水/泛碱", "confidence": 0.8,
                       "scores": {"完好": 0.2, "渗水/泛碱": 0.8}}

    def review(self, confidence=0.8):
        raw = json.dumps({"defectType": "渗水/泛碱", "defectDescription": "盖梁可见水痕",
                          "ratingScale": "2", "confidence": confidence, "evidence": "水痕流挂"},
                         ensure_ascii=False)
        return {"id": self.item["id"], "filename": self.item["filename"],
                "vision_candidate": self.vision["defectType"], "vision_confidence": 0.8,
                "raw": raw, "attempts": [raw], "valid": True,
                "prediction": strict_prediction(raw, LEXICON, self.item)}

    def test_zero_confidence_never_replaces_even_with_zero_threshold(self):
        merged, audit = merge_reviews([self.item], [self.base], [self.vision], [self.review(0)],
                                      LEXICON, review_min_confidence=0)
        self.assertEqual(merged, [self.base])
        self.assertFalse(audit[0]["accepted"])

    def test_nonfinite_confidence_and_missing_fields_rejected(self):
        for value in (float("nan"), float("inf"), True, "0.9"):
            self.assertIsNone(self.review(value)["prediction"])
        parsed = json.loads(self.review()["raw"])
        for key in ("evidence", "confidence", "ratingScale"):
            incomplete = {name: value for name, value in parsed.items() if name != key}
            self.assertIsNone(strict_prediction(json.dumps(incomplete), LEXICON, self.item))

    def test_review_does_not_show_visual_candidate_or_base_answer(self):
        client = Mock()
        client.generate.return_value = self.review()["raw"]
        with patch("track4_agent.vision_review._review_images", return_value=[]), \
                patch("track4_agent.vision_review.build_prompt", return_value="neutral legal-label prompt"):
            review_image(client, self.item, self.base, self.vision, LEXICON, {}, Path("image"), Path("raw"))
        prompt = client.generate.call_args.args[1]
        self.assertNotIn(self.vision["defectType"], prompt)
        self.assertNotIn(self.base["defectDescription"], prompt)
        self.assertNotIn("与该候选标签相符", prompt)

    def test_incomplete_revalidation_cache_fails_without_model(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            cached_reviews([(self.item, self.base, self.vision)], [], require_complete=True)

    def test_negation_scope_does_not_cross_punctuation(self):
        self.assertFalse(evidence_contradicts("裂缝(混凝土裂缝)", {
            "defectDescription": "有裂缝，无渗水", "evidence": "纵向裂缝"}))
        self.assertTrue(evidence_contradicts("钢筋锈蚀、破损", {
            "defectDescription": "混凝土破损，未见钢筋", "evidence": "表面破损"}))
        self.assertTrue(evidence_contradicts("渗水/泛碱", {
            "defectDescription": "混凝土无明显泛白", "evidence": "表面完好"}))

    def test_paired_evaluation_and_candidate_merge_are_identical(self):
        truth = {**self.base, "image": self.item["id"], "defectType": "渗水/泛碱"}
        # The same metadata-only item is supplied by the holdout evaluator.
        item = {**self.item, "image": self.item["id"]}
        review = self.review()
        merged, _ = merge_reviews([item], [self.base], [self.vision], [review], LEXICON)
        evaluated = policy_records([{"truth": truth, "prediction": self.base}], [self.vision], [review],
                                    LEXICON, min_confidence=0.55, review_min_confidence=0.55)
        self.assertEqual(evaluated["records"][0]["prediction"], merged[0])

    def test_cli_revalidation_rejects_missing_record_without_model_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "runs"
            folder.mkdir()
            base, vision, descriptions = folder / "base.json", folder / "vision.json", folder / "cache.json"
            for path, payload in ((base, [self.base]), (vision, {"records": [self.vision]}),
                                  (descriptions, {"context": {"review_min_confidence": 0.55}, "records": []})):
                path.write_text(json.dumps(payload), encoding="utf-8")
            argv = ["candidate", "--base", str(base), "--vision", str(vision), "--output", str(folder / "out.json"),
                    "--audit", str(folder / "audit.json"), "--descriptions", str(descriptions), "--revalidate-only"]
            with patch.object(apply_visual_candidate, "ROOT", root), patch.object(sys, "argv", argv), \
                    patch.object(apply_visual_candidate, "load_assets", return_value=([], [self.item], LEXICON)), \
                    patch.object(apply_visual_candidate, "load_config", return_value={"seed": 1}), \
                    patch.object(apply_visual_candidate, "review_context", return_value={}), \
                    patch.object(apply_visual_candidate, "load_client") as load_client:
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    apply_visual_candidate.main()
                load_client.assert_not_called()
                self.assertFalse((folder / "out.json").exists())


if __name__ == "__main__":
    unittest.main()
