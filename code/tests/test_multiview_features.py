from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track4_agent.vision_classifier import feature_bank, fuse_view_features


class MultiviewFeaturesTests(unittest.TestCase):
    def test_fusion_keeps_global_mean_and_max_in_separate_blocks(self):
        vectors = torch.tensor([[[1., 0.], [0., 1.], [1., 0.], [0., 1.], [1., 0.]]])
        original = vectors.clone()
        expected = torch.tensor([[1., 0., 0.5, 0.5, 1., 1.]])
        expected = torch.nn.functional.normalize(expected, dim=-1)
        self.assertTrue(torch.allclose(fuse_view_features(vectors), expected))
        self.assertTrue(torch.equal(vectors, original))

    def test_fusion_rejects_wrong_view_counts(self):
        for shape in ((5, 3), (2, 4, 3), (1, 6, 3)):
            with self.assertRaises(ValueError):
                fuse_view_features(torch.zeros(shape))

    def test_identical_views_and_zero_vectors_remain_finite(self):
        for vectors in (torch.zeros(2, 5, 3), torch.ones(2, 5, 3)):
            fused = fuse_view_features(vectors)
            self.assertEqual(tuple(fused.shape), (2, 9))
            self.assertTrue(torch.isfinite(fused).all())

    def test_old_global_cache_still_loads_without_loading_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "data" / "raw" / "image.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"fixture")
            key = json.dumps(("image.jpg", image.stat().st_size, image.stat().st_mtime_ns), ensure_ascii=False)
            cache = root / "bank.npz"
            np.savez(cache, metadata=json.dumps({"model": {}, "device": "cpu"}),
                     keys=np.array([key]), vectors=np.array([[1., 2.]]))
            with patch("track4_agent.vision_classifier.model_identity", return_value={}):
                actual = feature_bank(root, [{"image": "image.jpg"}], root, cache_path=cache)
            np.testing.assert_array_equal(actual, [[1., 2.]])

    def test_view_mode_is_bound_to_cache_and_invalid_options_fail_early(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "bank.npz"
            for kwargs in ({"views": "unsupported"}, {"precision": "unsupported"}, {"batch_size": 0}, {"threads": 0}):
                with self.assertRaises(ValueError):
                    feature_bank(root, [], root, cache_path=cache, **kwargs)
            image = root / "data" / "raw" / "image.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"fixture")
            key = json.dumps(("image.jpg", image.stat().st_size, image.stat().st_mtime_ns), ensure_ascii=False)
            np.savez(cache, metadata=json.dumps({"model": {}, "device": "cpu", "views": "global"}),
                     keys=np.array([key]), vectors=np.ones((1, 2)))
            with patch("track4_agent.vision_classifier.model_identity", return_value={}), \
                 patch("transformers.AutoModel.from_pretrained", side_effect=RuntimeError("cache miss needs model")):
                with self.assertRaisesRegex(RuntimeError, "cache miss needs model"):
                    feature_bank(root, [{"image": "image.jpg"}], root, cache_path=cache, views="quadrants")


if __name__ == "__main__":
    unittest.main()
