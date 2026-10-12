from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_holdout import (
    _allow_review, _gpu_preflight, _input_item, _label_coverage, _make_holdout, _paired_comparison,
    _report_lock, _variants, _write_report, run_holdout,
)


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def generate(self, images, prompt):
        self.calls += 1
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return value


def response(label="裂缝"):
    return json.dumps({"defectType": label, "defectDescription": "可见裂缝" if label == "裂缝" else "未见明显病害",
                       "ratingScale": "2" if label == "裂缝" else "", "confidence": 0.9,
                       "evidence": "细裂缝" if label == "裂缝" else "表面完整"}, ensure_ascii=False)


class HoldoutRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        model_dir = self.root / "models" / "fixture"
        model_dir.mkdir(parents=True)
        self.weight = model_dir / "model.safetensors"
        self.weight.write_bytes(b"test fixture, not actual weights")
        self.config = {"model_id": "test/model", "model_dir": str(model_dir), "seed": 17,
                       "global_max_side": 64, "bridge_use_crops": False, "track_use_crops": False,
                       "crop_size": 32, "confidence_threshold": 0.55,
                       "bridge_review_enabled": False, "calibration_review": True}
        self.sample = []
        for index, (category, label) in enumerate((("桥梁", "裂缝"), ("轨道", "完好"))):
            row = {"image": f"training/image-{index}.jpg", "filename": f"image-{index}.jpg",
                   "questionCategory": category, "bridgeName": "留出桥" if category == "桥梁" else "",
                   "defectLocation": "", "defectType": label,
                   "defectDescription": "训练目标描述", "ratingScale(1-5)": "2" if label == "裂缝" else ""}
            image_path = self.root / "data" / "raw" / row["image"]
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 48), "gray").save(image_path)
            self.sample.append(row)
        self.lexicon = {"categories": {category: {
            "allowed_labels": ["完好", "裂缝"],
            "label_specs": {"完好": {"count": 3, "descriptions": [], "default_rating": "",
                                     "rating_distribution": {"": 3}},
                            "裂缝": {"count": 3, "descriptions": [], "default_rating": "2",
                                     "rating_distribution": {"2": 3}}},
        } for category in ("桥梁", "轨道")}}
        self.training = [{"image": "different training example"}]
        self.split = {"sample_size": 2, "bridge_group_held_out": "留出桥"}
        self.output = self.root / "runs" / "experiment" / "holdout.json"
        self.formal = self.root / "result" / "result.json"
        self.formal.parent.mkdir()
        self.formal.write_bytes(b"preserve the scored baseline")

    def run_fixture(self, **kwargs):
        return run_holdout(self.root, self.config, self.sample, self.training, self.lexicon,
                           self.split, _variants(["baseline"], self.config), self.output, **kwargs)

    def test_prepare_and_complete_resume_never_load_gpu_or_change_formal_output(self):
        factory = Mock(side_effect=AssertionError("GPU must not be loaded"))
        prepared = self.run_fixture(prepare_only=True, client_factory=factory)
        self.assertEqual(prepared["status"], "prepared")
        client = FakeClient([response(), response("完好")])
        result = self.run_fixture(resume=True, client_factory=lambda: client)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(client.calls, 2)
        self.assertEqual(result["results"]["baseline"]["metrics"]["defect_type_exact_accuracy"], 1)
        self.run_fixture(resume=True, client_factory=factory)
        factory.assert_not_called()
        self.assertEqual(self.formal.read_bytes(), b"preserve the scored baseline")

    def test_interrupted_run_resumes_only_missing_images(self):
        interrupted = FakeClient([response(), RuntimeError("interrupted GPU inference")])
        with self.assertRaisesRegex(RuntimeError, "interrupted GPU"):
            self.run_fixture(client_factory=lambda: interrupted)
        saved = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(len(saved["results"]["baseline"]["records"]), 1)
        resumed = FakeClient([response("完好")])
        report = self.run_fixture(resume=True, client_factory=lambda: resumed)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(resumed.calls, 1)

    def test_max_images_pauses_at_a_checkpoint(self):
        client = FakeClient([response()])
        report = self.run_fixture(max_images=1, client_factory=lambda: client)
        self.assertEqual(report["status"], "paused")
        self.assertEqual(client.calls, 1)
        report = self.run_fixture(resume=True, prepare_only=True)
        self.assertEqual(len(report["results"]["baseline"]["records"]), 1)

    def test_resume_rejects_changed_configuration(self):
        self.run_fixture(prepare_only=True)
        self.config["seed"] += 1
        with self.assertRaisesRegex(ValueError, "Resume context differs"):
            self.run_fixture(resume=True, prepare_only=True)

    def test_resume_rejects_changed_weights_images_and_training(self):
        self.run_fixture(prepare_only=True)
        original = self.weight.read_bytes()
        self.weight.write_bytes(b"changed weights")
        with self.assertRaisesRegex(ValueError, "Resume context differs"):
            self.run_fixture(resume=True, prepare_only=True)
        self.weight.write_bytes(original)
        image = self.root / "data" / "raw" / self.sample[0]["image"]
        original = image.read_bytes()
        Image.new("RGB", (64, 48), "red").save(image)
        with self.assertRaisesRegex(ValueError, "Resume context differs"):
            self.run_fixture(resume=True, prepare_only=True)
        image.write_bytes(original)
        self.training.append({"image": "changed training example"})
        with self.assertRaisesRegex(ValueError, "Resume context differs"):
            self.run_fixture(resume=True, prepare_only=True)

    def test_resume_rejects_raw_prediction_disagreement_before_model_load(self):
        self.run_fixture(client_factory=lambda: FakeClient([response(), response("完好")]))
        saved = json.loads(self.output.read_text(encoding="utf-8"))
        saved["results"]["baseline"]["records"][0]["prediction"]["defectType"] = "完好"
        _write_report(self.output, saved)
        factory = Mock(side_effect=AssertionError("GPU must not be loaded"))
        with self.assertRaisesRegex(ValueError, "raw responses and prediction disagree"):
            self.run_fixture(resume=True, client_factory=factory)
        factory.assert_not_called()

    def test_resume_rejects_duplicate_or_reordered_records(self):
        self.run_fixture(client_factory=lambda: FakeClient([response(), response("完好")]))
        saved = json.loads(self.output.read_text(encoding="utf-8"))
        saved["results"]["baseline"]["records"][1] = copy.deepcopy(saved["results"]["baseline"]["records"][0])
        _write_report(self.output, saved)
        with self.assertRaisesRegex(ValueError, "Checkpoint truth"):
            self.run_fixture(resume=True, prepare_only=True)

    def test_old_output_is_not_silently_overwritten_and_paths_are_isolated(self):
        self.run_fixture(prepare_only=True)
        with self.assertRaisesRegex(ValueError, "Output already exists"):
            self.run_fixture(prepare_only=True)
        self.output = self.formal
        with self.assertRaisesRegex(ValueError, "inside workspace/runs"):
            self.run_fixture(prepare_only=True)

    def test_atomic_replace_failure_preserves_old_report(self):
        self.output.parent.mkdir(parents=True)
        self.output.write_text("original report", encoding="utf-8")
        with patch("evaluate_holdout.os.replace", side_effect=OSError("interrupted write")):
            with self.assertRaises(OSError):
                _write_report(self.output, {"new": "report"})
        self.assertEqual(self.output.read_text(encoding="utf-8"), "original report")
        self.assertEqual(list(self.output.parent.glob("*.tmp")), [])

    def test_output_lock_rejects_concurrent_writers(self):
        with _report_lock(self.output):
            with self.assertRaisesRegex(ValueError, "Another evaluation"):
                with _report_lock(self.output):
                    self.fail("second writer obtained the lock")

    def test_variant_factors_are_independent_even_with_experimental_config(self):
        variants = _variants(["baseline", "all_labels", "strict_review", "full_label_scope"],
                             {**self.config, "label_scope": "all", "review_selection": "strict"})
        base = variants["baseline"]
        self.assertEqual(base["label_scope"], "scene")
        self.assertEqual(base["review_selection"], "legacy")
        self.assertEqual(variants["all_labels"], {**base, "label_scope": "all"})
        self.assertEqual(variants["strict_review"], {**base, "review_selection": "strict"})
        self.assertEqual(variants["full_label_scope"], {**base, "label_scope": "all", "review_selection": "strict"})
        for names in ([], ["baseline", "baseline"], ["unknown"]):
            with self.assertRaisesRegex(ValueError, "unique known variants"):
                _variants(names, self.config)

    def test_selection_only_ablation_reuses_model_responses(self):
        client = FakeClient([response(), response("完好")])
        variants = _variants(["baseline", "strict_review"], self.config)
        report = run_holdout(self.root, self.config, self.sample, self.training, self.lexicon,
                             self.split, variants, self.output, client_factory=lambda: client)
        self.assertEqual(client.calls, 2)
        self.assertEqual(report["results"]["strict_review"]["raw_replay_images"], 2)
        self.assertEqual(report["results"]["strict_review"]["model_inferred_images"], 0)
        self.assertEqual(report["comparisons"]["strict_review"]["label_changes"], 0)
        factory = Mock(side_effect=AssertionError("GPU must not be loaded"))
        run_holdout(self.root, self.config, self.sample, self.training, self.lexicon,
                    self.split, variants, self.output, resume=True, client_factory=factory)
        factory.assert_not_called()
        saved = json.loads(self.output.read_text(encoding="utf-8"))
        saved["results"]["strict_review"]["records"][0]["raw_source_variant"] = "strict_review"
        _write_report(self.output, saved)
        with self.assertRaisesRegex(ValueError, "raw replay source differs"):
            run_holdout(self.root, self.config, self.sample, self.training, self.lexicon,
                        self.split, variants, self.output, resume=True, prepare_only=True)

    def test_changed_prompts_do_not_reuse_raw_responses(self):
        client = FakeClient([response(), response("完好"), response(), response("完好")])
        variants = _variants(["baseline", "all_labels"], self.config)
        report = run_holdout(self.root, self.config, self.sample, self.training, self.lexicon,
                             self.split, variants, self.output, client_factory=lambda: client)
        self.assertEqual(client.calls, 4)
        self.assertEqual(report["results"]["all_labels"]["raw_replay_images"], 0)

    def test_raw_replay_recomputes_strict_selection_instead_of_copying_answer(self):
        self.sample[0]["questionCategory"] = "轨道"
        self.sample[0]["bridgeName"] = ""
        first = json.loads(response())
        first["confidence"] = 0.3
        review = json.loads(response("完好"))
        review["confidence"] = 0.2
        client = FakeClient([json.dumps(first), json.dumps(review), response("完好")])
        variants = _variants(["baseline", "strict_review"], self.config)
        report = run_holdout(self.root, self.config, self.sample, self.training, self.lexicon,
                             self.split, variants, self.output, client_factory=lambda: client)
        self.assertEqual(client.calls, 3)
        self.assertEqual(report["results"]["baseline"]["records"][0]["prediction"]["defectType"], "完好")
        self.assertEqual(report["results"]["strict_review"]["records"][0]["prediction"]["defectType"], "裂缝")
        self.assertEqual(report["comparisons"]["strict_review"]["gained_correct"], 1)

    def test_review_permissions_match_deployment_and_inputs_exclude_targets(self):
        bridge, track = map(_input_item, self.sample)
        self.assertFalse(_allow_review(bridge, self.config))
        self.assertTrue(_allow_review(track, self.config))
        self.assertTrue(_allow_review(bridge, {**self.config, "bridge_review_enabled": True,
                                             "calibration_review": False, "bridge_review_scenes": []}))
        for key in ("defectType", "defectDescription", "ratingScale(1-5)"):
            self.assertNotIn(key, bridge)

    def test_prompt_coverage_reports_rare_excluded_training_labels(self):
        lexicon = copy.deepcopy(self.lexicon)
        labels = [f"罕见病害{index}" for index in range(13)]
        lexicon["categories"]["桥梁"]["allowed_labels"].extend(labels)
        lexicon["categories"]["桥梁"]["label_specs"].update({
            label: {"count": 0} for label in labels})
        sample = [{**self.sample[0], "defectType": labels[-1]}]
        limited = _label_coverage(sample, lexicon, self.config)
        full = _label_coverage(sample, lexicon, {**self.config, "label_scope": "all"})
        self.assertEqual(limited["桥梁"]["included_targets"], 0)
        self.assertEqual(limited["桥梁"]["excluded_label_counts"], {labels[-1]: 1})
        self.assertEqual(full["桥梁"]["included_targets"], 1)

    def test_comparison_aligns_ids_and_counts_gains_and_losses(self):
        report = self.run_fixture(client_factory=lambda: FakeClient([response(), response("完好")]))
        base = report["results"]["baseline"]["records"]
        candidate = copy.deepcopy(base)
        base[0]["prediction"]["defectType"] = "完好"
        candidate[1]["prediction"]["defectType"] = "裂缝"
        comparison = _paired_comparison(base, list(reversed(candidate)))
        self.assertEqual(comparison["paired_sample_size"], 2)
        self.assertEqual(comparison["gained_correct"], 1)
        self.assertEqual(comparison["lost_correct"], 1)
        self.assertEqual(comparison["label_changes"], 2)

    def test_gpu_guard_refuses_busy_gpu_without_loading_torch(self):
        busy = subprocess.CompletedProcess([], 0, stdout="500, 12227, 99\n")
        with patch("evaluate_holdout.subprocess.run", return_value=busy):
            with self.assertRaisesRegex(RuntimeError, "GPU is occupied"):
                _gpu_preflight(10240)
        idle = subprocess.CompletedProcess([], 0, stdout="11000, 12227, 0\n")
        with patch("evaluate_holdout.subprocess.run", return_value=idle):
            _gpu_preflight(10240)
        with self.assertRaises(ValueError):
            _gpu_preflight(0)

    def test_invalid_sample_size_and_duplicate_images_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            _make_holdout(self.sample, 0, 17, "留出桥")
        with self.assertRaisesRegex(ValueError, "duplicate image"):
            _make_holdout(self.sample + self.sample, 2, 17, "留出桥")
        with self.assertRaisesRegex(ValueError, "Max images"):
            self.run_fixture(prepare_only=True, max_images=0)


if __name__ == "__main__":
    unittest.main()
