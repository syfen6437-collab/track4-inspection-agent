from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apply_track_adapter import _load_report, _validate_base


class ApplyTrackAdapterTests(unittest.TestCase):
    def test_base_result_metadata_and_schema_are_protected(self):
        manifest = [{"filename": "a.jpg", "questionCategory": "轨道", "bridgeName": "",
                     "defectLocation": ""}]
        lexicon = {"categories": {"轨道": {"allowed_labels": ["完好"]}}}
        base = [{"questionCategory": "轨道", "bridgeName": "", "defectLocation": "", "filename": "a.jpg",
                 "defectType": "完好", "defectDescription": "模型描述", "ratingScale(1-5)": ""}]
        _validate_base(base, manifest, lexicon)
        bad = dict(base[0]); bad["defectType"] = "非法"
        with self.assertRaises(ValueError):
            _validate_base([bad], manifest, lexicon)
        bad = dict(base[0]); bad["bridgeName"] = "wrong"
        with self.assertRaisesRegex(ValueError, "metadata"):
            _validate_base([bad], manifest, lexicon)

    def test_adapter_report_requires_complete_hash_and_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report_path = root / "report.json"
            state_path = root / "report.pt"
            state = {"model_identity": {"config.json": "x"}, "atoms": ["完好"],
                     "mean": torch.zeros(2), "scale": torch.ones(2), "vision": {}, "classifier": {}}
            state_path.write_bytes(b"state")
            report_path.write_text(json.dumps({"status": "complete", "selected_epoch": 1,
                                               "model_identity": state["model_identity"], "atoms": state["atoms"],
                                               "adapter_sha256": "wrong"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                _load_report(report_path)

    def test_candidate_output_must_be_inside_runs(self):
        from track4_agent.vision_review import candidate_path
        with self.assertRaises(ValueError):
            candidate_path(Path("C:/workspace"), Path("C:/workspace/result/result.json"))

    def test_healthy_baseline_protection_is_explicit_in_candidate_policy(self):
        from track4_agent.track_consensus import adapter_decision
        base = {"filename": "a.jpg", "questionCategory": "轨道", "defectType": "完好"}
        prediction = {"label": "破损", "confidence": 1.0}
        self.assertFalse(adapter_decision(base, prediction, prediction)["accepted"])
        base["defectType"] = "破损"
        prediction["label"] = "完好"
        self.assertFalse(adapter_decision(base, prediction, prediction)["accepted"])

    def test_duplicate_baseline_is_rejected_before_model_loading(self):
        manifest = [{"filename": name, "questionCategory": "轨道", "bridgeName": "", "defectLocation": ""}
                    for name in ("a.jpg", "b.jpg")]
        row = {"questionCategory": "轨道", "bridgeName": "", "defectLocation": "", "filename": "a.jpg",
               "defectType": "完好", "defectDescription": "模型", "ratingScale(1-5)": ""}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _validate_base([row, row], manifest, {"categories": {"轨道": {"allowed_labels": ["完好"]}}})


if __name__ == "__main__":
    unittest.main()
