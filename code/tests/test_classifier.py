from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track4_agent.vision_classifier import atomic_parts, decode_atomic


class AtomicPredictionTests(unittest.TestCase):
    def test_healthy_is_not_an_additional_defect(self) -> None:
        self.assertEqual(atomic_parts("完好"), set())
        self.assertEqual(atomic_parts("钢筋锈蚀、破损"), {"钢筋锈蚀", "破损"})

    def test_model_probabilities_preserve_coexisting_defects(self) -> None:
        atoms = ["裂缝(混凝土裂缝)", "钢筋锈蚀", "破损"]
        allowed = ["完好", "裂缝(混凝土裂缝)", "钢筋锈蚀", "破损",
                   "钢筋锈蚀、破损", "裂缝(混凝土裂缝)、破损"]
        probabilities = np.array([[0.1, 0.95, 0.9], [0.02, 0.03, 0.04], [0.9, 0.1, 0.8]])
        predictions = decode_atomic(probabilities, atoms, allowed)
        self.assertEqual(predictions, ["钢筋锈蚀、破损", "完好", "裂缝(混凝土裂缝)、破损"])

    def test_decoder_never_emits_an_unlisted_combination(self) -> None:
        allowed = ["完好", "钢筋锈蚀", "破损"]
        predicted = decode_atomic(np.array([[1.0, 1.0], [0.0, 0.0]]), ["钢筋锈蚀", "破损"], allowed)
        self.assertTrue(all(label in allowed for label in predicted))
        self.assertEqual(predicted[1], "完好")


if __name__ == "__main__":
    unittest.main()
