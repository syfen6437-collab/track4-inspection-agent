from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track4_agent.vision_review import candidate_path, merge_reviews, select_candidates, strict_prediction


LEXICON = {
    "categories": {
        "桥梁": {
            "allowed_labels": ["完好", "渗水/泛碱"],
            "label_specs": {
                "完好": {"count": 2, "rating_distribution": {"": 2}, "default_rating": ""},
                "渗水/泛碱": {"count": 2, "rating_distribution": {"2": 2}, "default_rating": "2"},
            },
            "scene_label_counts": {"support": {"完好": 1, "渗水/泛碱": 1}},
        }
    }
}


class VisionReviewTests(unittest.TestCase):
    def test_selection_is_limited_to_support_and_bottom_and_checks_ids(self) -> None:
        manifest = [
            {"id": "桥/支座.JPG", "image": "初赛测试集/桥/支座.JPG", "filename": "支座.JPG",
             "questionCategory": "桥梁", "bridgeName": "桥", "defectLocation": ""},
            {"id": "桥/DJI_1.JPG", "image": "初赛测试集/桥/DJI_1.JPG", "filename": "DJI_1.JPG",
             "questionCategory": "桥梁", "bridgeName": "桥", "defectLocation": ""},
        ]
        results = [
            {**{key: row[key] for key in ("questionCategory", "bridgeName", "defectLocation", "filename")},
             "defectType": "完好", "defectDescription": "无明显病害", "ratingScale(1-5)": ""}
            for row in manifest
        ]
        vision = [
            {"id": row["id"], "questionCategory": "桥梁", "filename": row["filename"],
             "defectType": "渗水/泛碱", "confidence": 0.8, "scores": {}}
            for row in manifest
        ]
        selected = select_candidates(manifest, results, vision, LEXICON)
        self.assertEqual([row[0]["id"] for row in selected], ["桥/支座.JPG"])

    def test_selection_rejects_malformed_base_output(self) -> None:
        manifest = [{"id": "桥/支座.JPG", "image": "初赛测试集/桥/支座.JPG", "filename": "支座.JPG",
                     "questionCategory": "桥梁", "bridgeName": "桥", "defectLocation": ""}]
        base = [{"questionCategory": "桥梁", "bridgeName": "桥", "defectLocation": "", "filename": "支座.JPG",
                 "defectType": "完好", "defectDescription": "无明显病害"}]
        vision = [{"id": "桥/支座.JPG", "questionCategory": "桥梁", "filename": "支座.JPG",
                   "defectType": "渗水/泛碱", "confidence": 0.8, "scores": {}}]
        with self.assertRaises(ValueError):
            select_candidates(manifest, base, vision, LEXICON)

    def test_merge_rejects_confidence_or_attempt_tampering(self) -> None:
        manifest = [{"id": "桥/支座.JPG", "image": "初赛测试集/桥/支座.JPG", "filename": "支座.JPG",
                     "questionCategory": "桥梁", "bridgeName": "桥", "defectLocation": ""}]
        base = [{"questionCategory": "桥梁", "bridgeName": "桥", "defectLocation": "", "filename": "支座.JPG",
                 "defectType": "完好", "defectDescription": "无明显病害", "ratingScale(1-5)": ""}]
        vision = [{"id": "桥/支座.JPG", "questionCategory": "桥梁", "filename": "支座.JPG",
                   "defectType": "渗水/泛碱", "confidence": 0.8, "scores": {}}]
        raw = ('{"defectType":"渗水/泛碱","defectDescription":"盖梁可见水痕",'
               '"ratingScale":"2","confidence":0.8,"evidence":"水痕流挂"}')
        prediction = strict_prediction(raw, LEXICON, manifest[0])
        review = {"id": "桥/支座.JPG", "filename": "支座.JPG", "vision_candidate": "渗水/泛碱",
                  "vision_confidence": 0.7, "valid": True, "prediction": prediction, "raw": raw,
                  "attempts": [raw]}
        with self.assertRaises(ValueError):
            merge_reviews(manifest, base, vision, [review], LEXICON, scenes=("support",))

    def test_strict_prediction_rejects_missing_evidence_and_accepts_full_json(self) -> None:
        valid = ('{"defectType":"渗水/泛碱","defectDescription":"盖梁可见水痕",'
                 '"ratingScale":"2","confidence":0.8,"evidence":"水痕流挂"}')
        prediction = strict_prediction(valid, LEXICON, {
            "questionCategory": "桥梁", "filename": "支座.JPG"})
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction["defectType"], "渗水/泛碱")
        missing = valid.replace(',"evidence":"水痕流挂"', '')
        self.assertIsNone(strict_prediction(missing, LEXICON, {
            "questionCategory": "桥梁", "filename": "支座.JPG"}))

    def test_merge_requires_model_review_and_keeps_candidate_inside_runs(self) -> None:
        manifest = [{"id": "桥/支座.JPG", "image": "初赛测试集/桥/支座.JPG", "filename": "支座.JPG",
                     "questionCategory": "桥梁", "bridgeName": "桥", "defectLocation": ""}]
        base = [{"questionCategory": "桥梁", "bridgeName": "桥", "defectLocation": "", "filename": "支座.JPG",
                 "defectType": "完好", "defectDescription": "无明显病害", "ratingScale(1-5)": ""}]
        vision = [{"id": "桥/支座.JPG", "questionCategory": "桥梁", "filename": "支座.JPG",
                   "defectType": "渗水/泛碱", "confidence": 0.8, "scores": {}}]
        raw = ('{"defectType":"渗水/泛碱","defectDescription":"盖梁可见水痕",'
               '"ratingScale":"2","confidence":0.8,"evidence":"水痕流挂"}')
        prediction = strict_prediction(raw, LEXICON, manifest[0])
        reviews = [{"id": "桥/支座.JPG", "filename": "支座.JPG", "vision_candidate": "渗水/泛碱",
                    "vision_confidence": 0.8, "valid": True, "prediction": prediction, "raw": raw}]
        merged, audit = merge_reviews(manifest, base, vision, reviews, LEXICON,
                                      min_confidence=0.55, scenes=("support",))
        self.assertEqual(merged[0]["defectType"], "渗水/泛碱")
        self.assertTrue(audit[0]["accepted"])
        self.assertTrue(str(candidate_path(Path("E:/workspace"), Path("E:/workspace/runs/a.json"))).endswith("runs\\a.json")
                        or str(candidate_path(Path("E:/workspace"), Path("E:/workspace/runs/a.json"))).endswith("runs/a.json"))
        with self.assertRaises(ValueError):
            merge_reviews(manifest, base, vision, [], LEXICON, scenes=("support",))


if __name__ == "__main__":
    unittest.main()
