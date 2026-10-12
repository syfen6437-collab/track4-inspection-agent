from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_track_adapter import class_weights, selected_state, training_partition, trainable_vision_parameters
from verify_track_adapter import check_snapshot


class TrackAdapterTests(unittest.TestCase):
    def test_inner_partition_is_disjoint_deterministic_and_label_blind(self):
        rows = [{"image": str(i), "defectType": "arbitrary"} for i in range(10)]
        fitting, validation = training_partition(rows, 3, 17)
        again = training_partition([{**r, "defectType": "changed"} for r in rows], 3, 17)
        self.assertFalse({r["image"] for r in fitting} & {r["image"] for r in validation})
        self.assertEqual([r["image"] for r in validation], [r["image"] for r in again[1]])
        for size in (0, 10):
            with self.assertRaises(ValueError):
                training_partition(rows, size, 17)

    def test_only_last_blocks_and_pooling_are_unfrozen(self):
        class Vision(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Module()
                self.encoder.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])
                self.post_layernorm = torch.nn.LayerNorm(2)
                self.head = torch.nn.Linear(2, 2)
                self.embeddings = torch.nn.Linear(2, 2)
        vision = Vision()
        names = trainable_vision_parameters(vision, 2)
        self.assertTrue(all(not p.requires_grad for p in vision.encoder.layers[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in vision.encoder.layers[-1].parameters()))
        self.assertFalse(vision.embeddings.weight.requires_grad)
        self.assertIn("head.weight", names)
        snapshot = selected_state(vision, names, torch.nn.Linear(2, 1))
        before = snapshot["vision"]["head.weight"].clone()
        with torch.no_grad():
            vision.head.weight.add_(1)
        self.assertTrue(torch.equal(snapshot["vision"]["head.weight"], before))
        self.assertNotIn("embeddings.weight", snapshot["vision"])
        with self.assertRaises(ValueError):
            trainable_vision_parameters(vision, 5)

    def test_class_weights_are_finite_and_bounded_for_rare_labels(self):
        weights = class_weights(torch.tensor([[1., 0.], [1., 0.], [0., 1.]]))
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue((weights >= 0.5).all())
        self.assertTrue((weights <= 3).all())
        with self.assertRaises(ValueError):
            class_weights(torch.zeros(0, 2))

    def test_saved_state_identity_and_nonfinite_tensors_are_rejected(self):
        state = {"model_identity": {"weights": "hash"}, "selected_epoch": 1, "atoms": ["破损"],
                 "vision": {"weight": torch.ones(1)}, "classifier": {"weight": torch.ones(1)},
                 "mean": torch.zeros(2), "scale": torch.ones(2)}
        report = {key: state[key] for key in ("model_identity", "selected_epoch", "atoms")}
        check_snapshot(state, report)
        with self.assertRaisesRegex(ValueError, "epoch differs"):
            check_snapshot({**state, "selected_epoch": 2}, report)
        with self.assertRaisesRegex(ValueError, "vocabulary differs"):
            check_snapshot({**state, "atoms": ["wrong"]}, report)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            check_snapshot({**state, "vision": {"weight": torch.tensor([float("nan")])}}, report)
        with self.assertRaisesRegex(ValueError, "scaling"):
            check_snapshot({**state, "scale": torch.zeros(2)}, report)


if __name__ == "__main__":
    unittest.main()
