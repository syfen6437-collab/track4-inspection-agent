from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_vision_review import fold_predictions, input_item
from track4_agent.vision_review import select_candidates


class ReviewEvaluationTests(unittest.TestCase):
    def test_fold_knn_scores_follow_class_vocabulary_and_trigger_review(self) -> None:
        training = [
            {"image": f"train/{index}.jpg", "questionCategory": "桥梁", "defectType": label}
            for index, label in enumerate(["完好", "完好", "完好", "渗水/泛碱", "渗水/泛碱"])
        ]
        training.append({"image": "train/rail.jpg", "questionCategory": "轨道", "defectType": "破损"})
        sample = [
            {"image": "holdout/support.jpg", "filename": "支座.JPG", "questionCategory": "桥梁"},
            {"image": "holdout/rail.jpg", "filename": "rail.jpg", "questionCategory": "轨道"},
        ]
        features = np.array([[1, 0], [1, 0.1], [0.7, 0.7], [0, 1], [0.1, 1],
                             [1, 0], [0, 2], [1, 0]], dtype=np.float32)
        positions = {row["image"]: index for index, row in enumerate(training + sample)}
        lexicon = {"categories": {
            "桥梁": {"allowed_labels": ["完好", "渗水/泛碱"]},
            "轨道": {"allowed_labels": ["破损"]},
        }}
        with patch("evaluate_vision_review.fit_head", return_value={"labels": ["破损"]}), \
                patch("evaluate_vision_review.predict_head", return_value=(["破损"], np.array([[0.9]]))):
            records = fold_predictions(sample, training, lexicon, features, positions,
                                       bridge_method="knn", knn_k=3)
        bridge = records[0]
        self.assertEqual(bridge["defectType"], "渗水/泛碱")
        self.assertEqual(set(bridge["scores"]), {"完好", "渗水/泛碱"})
        self.assertAlmostEqual(sum(bridge["scores"].values()), 1.0)
        self.assertEqual(bridge["confidence"], bridge["scores"][bridge["defectType"]])
        item = input_item(sample[0])
        base = {key: item[key] for key in ("questionCategory", "bridgeName", "defectLocation", "filename")}
        base.update(defectType="完好", defectDescription="无明显病害")
        base["ratingScale(1-5)"] = ""
        self.assertEqual(len(select_candidates([item], [base], [bridge], lexicon)), 1)
        with self.assertRaisesRegex(ValueError, "scores and confidence"):
            select_candidates([item], [base], [{**bridge, "scores": {"完好": 0.1, "渗水/泛碱": 0.2}}], lexicon)
        with self.assertRaisesRegex(ValueError, "score labels"):
            select_candidates([item], [base], [{**bridge, "scores": {"完好": 1.0}}], lexicon)

    def test_model_input_excludes_holdout_targets(self) -> None:
        item = input_item({"image": "holdout/a.jpg", "filename": "a.jpg", "questionCategory": "桥梁",
                           "defectType": "渗水/泛碱", "defectDescription": "heldout evidence",
                           "ratingScale(1-5)": "3"})
        self.assertNotIn("defectType", item)
        self.assertNotIn("defectDescription", item)
        self.assertNotIn("ratingScale(1-5)", item)


if __name__ == "__main__":
    unittest.main()
