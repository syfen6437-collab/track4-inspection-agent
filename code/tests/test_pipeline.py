from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from track4_agent.model import (
    build_checklist_prompt, candidate_labels, checklist_prediction, crop_montage,
    needs_review, sanitize_prediction,
)
from track4_agent.inference import infer_item, run_inference
from evaluate_holdout import _make_holdout
from track4_agent.validation import validate_result


LEXICON = {
    "categories": {
        "桥梁": {
            "allowed_labels": ["完好", "裂缝"],
            "label_specs": {
                "完好": {"count": 10, "rating_distribution": {"": 10}, "default_rating": ""},
                "裂缝": {"count": 3, "rating_distribution": {"2": 3}, "default_rating": "2"},
            },
        }
    }
}


class PipelineTests(unittest.TestCase):

    def test_negated_healthy_evidence_does_not_trigger_review(self) -> None:
        self.assertFalse(needs_review({"defectType": "完好", "defectDescription": "无裂缝、无渗水", "confidence": 1.0}, 0.5))
        self.assertTrue(needs_review({"defectType": "完好", "defectDescription": "局部可见裂缝", "confidence": 1.0}, 0.5))
    class FakeClient:
        def __init__(self, responses: list[str]) -> None:
            self.responses = iter(responses)
            self.prompts: list[str] = []
            self.image_counts: list[int] = []

        def generate(self, images, prompt: str) -> str:
            self.prompts.append(prompt)
            self.image_counts.append(len(images))
            return next(self.responses)

    def test_crop_montage_contains_four_tiles(self) -> None:
        montage = crop_montage(Image.new("RGB", (800, 600), "gray"), 128)
        self.assertEqual(montage.size, (256, 256))

    def test_holdout_split_removes_entire_bridge_and_sampled_images(self) -> None:
        rows = []
        for bridge in ("留出桥（左幅）", "留出桥（右幅）", "其他桥（左幅）", "其他桥（右幅）"):
            for index in range(4):
                rows.append({
                    "image": f"{bridge}-{index}.jpg",
                    "questionCategory": "桥梁",
                    "bridgeName": bridge,
                    "filename": f"image-{index}.jpg",
                    "defectType": "完好" if index < 2 else "裂缝",
                })
        for index in range(8):
            rows.append({
                "image": f"track-{index}.jpg",
                "questionCategory": "轨道",
                "bridgeName": "",
                "filename": f"track-{index}.jpg",
                "defectType": "完好" if index < 4 else "裂缝",
            })

        sample, training, report = _make_holdout(rows, 10, 17, "留出桥")
        sampled_images = {row["image"] for row in sample}
        training_images = {row["image"] for row in training}

        self.assertEqual(report["bridge_group_image_count"], 8)
        self.assertTrue(all(not row["bridgeName"].startswith("留出桥") for row in training if row["questionCategory"] == "桥梁"))
        self.assertFalse(sampled_images & training_images)
        self.assertEqual(len(sample), 10)

    def test_sanitize_prediction_normalizes_rating(self) -> None:
        prediction, valid = sanitize_prediction(
            '{"defectType":"裂缝","defectDescription":"梁底可见细裂缝",'
            '"ratingScale":"5","confidence":0.8,"evidence":"细裂缝"}',
            LEXICON,
            "桥梁",
        )
        self.assertTrue(valid)
        self.assertEqual(prediction["defectType"], "裂缝")
        self.assertEqual(prediction["ratingScale"], "2")

    def test_full_label_scope_exposes_rare_legal_labels(self) -> None:
        lexicon = {
            "categories": {
                "桥梁": {
                    "allowed_labels": ["完好", "伸缩缝病害", "剥落"],
                    "label_specs": {
                        "完好": {"count": 10, "rating_distribution": {}, "default_rating": ""},
                        "伸缩缝病害": {"count": 1, "rating_distribution": {"2": 1}, "default_rating": "2"},
                        "剥落": {"count": 1, "rating_distribution": {"2": 1}, "default_rating": "2"},
                    },
                    "scene_label_counts": {"aerial": {"完好": 10}},
                }
            }
        }
        prompt = build_checklist_prompt(
            {"questionCategory": "桥梁", "filename": "DJI_0001.JPG"},
            lexicon,
            label_scope="all",
        )
        self.assertIn("伸缩缝病害", prompt)
        self.assertIn("剥落", prompt)

    def test_sanitize_prediction_fills_missing_nonhealthy_rating(self) -> None:
        prediction, valid = sanitize_prediction(
            '{"defectType":"裂缝","defectDescription":"梁底可见细裂缝",'
            '"ratingScale":"","confidence":0.8,"evidence":"细裂缝"}',
            LEXICON,
            "桥梁",
        )
        self.assertTrue(valid)
        self.assertEqual(prediction["ratingScale"], "2")

    def test_checklist_prediction_maps_selected_atomic_labels(self) -> None:
        lexicon = {
            "categories": {
                "桥梁": {
                    "allowed_labels": ["完好", "裂缝", "裂缝、破损"],
                    "label_specs": {
                        "完好": {"count": 10, "rating_distribution": {"": 10}, "default_rating": ""},
                        "裂缝": {"count": 3, "rating_distribution": {"2": 3}, "default_rating": "2"},
                        "裂缝、破损": {"count": 1, "rating_distribution": {"3": 1}, "default_rating": "3"},
                    },
                    "scene_label_counts": {"bottom": {"裂缝": 2, "裂缝、破损": 1}},
                }
            }
        }
        prediction, valid = checklist_prediction(
            '{"selected":["裂缝","破损"],"description":"梁底可见裂缝和破损","confidence":0.8}',
            lexicon,
            "桥梁",
            "bottom",
        )
        self.assertTrue(valid)
        self.assertEqual(prediction["defectType"], "裂缝、破损")

    def test_presence_prediction_is_not_reused_as_main_prompt_context(self) -> None:
        client = self.FakeClient([
            '{"has_defect":false,"coarse_type":"完好","confidence":1.0}',
            '{"defectType":"裂缝","defectDescription":"可见细裂缝",'
            '"ratingScale":"2","confidence":0.8,"evidence":"细裂缝"}',
        ])
        item = {
            "id": "test.jpg",
            "questionCategory": "桥梁",
            "bridgeName": "示例桥",
            "filename": "test.jpg",
        }
        config = {
            "global_max_side": 128,
            "bridge_max_side": 128,
            "bridge_use_crops": False,
            "two_stage_bridge": True,
            "presence_context_policy": "none",
            "confidence_threshold": 0.55,
            "bridge_review_enabled": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "test.jpg"
            Image.new("RGB", (128, 96), "gray").save(image_path)
            result, records = infer_item(client, item, LEXICON, config, image_path)

        self.assertEqual(result["defectType"], "裂缝")
        self.assertEqual(len(records), 2)
        self.assertNotIn("初筛模型观察到的粗类别", client.prompts[1])

    def test_strict_review_does_not_replace_with_low_confidence_valid_json(self) -> None:
        client = self.FakeClient([
            '{"defectType":"完好","defectDescription":"无明显病害",'
            '"ratingScale":"","confidence":0.8,"evidence":"未见异常"}',
            '{"defectType":"渗水/泛碱","defectDescription":"疑似水痕",'
            '"ratingScale":"2","confidence":0.2,"evidence":"颜色变化"}',
        ])
        item = {"id": "test.jpg", "questionCategory": "桥梁", "bridgeName": "示例桥", "filename": "支座.JPG"}
        config = {
            "global_max_side": 128, "bridge_max_side": 128, "bridge_use_crops": False,
            "crop_size": 64, "confidence_threshold": 0.55, "bridge_review_enabled": True,
            "review_selection": "strict", "bridge_review_scenes": ["support"],
        }
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "test.jpg"
            Image.new("RGB", (128, 96), "gray").save(image_path)
            result, records = infer_item(client, item, LEXICON, config, image_path,
                                         raw_dir=Path(directory))
        self.assertEqual(result["defectType"], "完好")
        self.assertEqual(len(records), 2)

    def test_aerial_healthy_prediction_can_trigger_crop_review(self) -> None:
        client = self.FakeClient([
            '{"has_defect":false,"coarse_type":"完好","confidence":1.0}',
            '{"defectType":"完好","defectDescription":"无明显病害",'
            '"ratingScale":"","confidence":0.99,"evidence":"未见明显异常"}',
            '{"defectType":"裂缝","defectDescription":"局部可见细裂缝",'
            '"ratingScale":"2","confidence":0.8,"evidence":"细裂缝"}',
        ])
        item = {
            "id": "bridge/DJI_0001.JPG",
            "questionCategory": "桥梁",
            "bridgeName": "示例桥",
            "filename": "DJI_0001.JPG",
        }
        config = {
            "global_max_side": 128,
            "bridge_max_side": 128,
            "bridge_use_crops": False,
            "two_stage_bridge": True,
            "presence_context_policy": "none",
            "confidence_threshold": 0.55,
            "bridge_review_enabled": True,
            "review_aerial_healthy": True,
            "bridge_aerial_review_crops": True,
            "aerial_crop_size": 64,
            "crop_size": 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            image_path = raw_dir / "初赛测试集" / item["id"]
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (128, 96), "gray").save(image_path)
            result, records = infer_item(
                client,
                item,
                LEXICON,
                config,
                image_path,
                raw_dir=raw_dir,
            )

        self.assertEqual(result["defectType"], "裂缝")
        self.assertEqual(len(records), 3)
        self.assertNotIn("初筛模型观察到的粗类别", client.prompts[1])
        self.assertEqual(client.image_counts[-1], 5)

    def test_candidate_resume_rejects_configuration_changes_and_preserves_submission(self) -> None:
        response = '{"defectType":"完好","defectDescription":"未见明显病害","ratingScale":"","confidence":0.9}'
        config = {"global_max_side": 64, "bridge_use_crops": False, "confidence_threshold": 0.5,
                  "checkpoint_every": 1, "deadline_hours": 6.0, "bridge_review_enabled": False}
        manifest = [{"id": "bridge/a.jpg", "image": "初赛测试集/bridge/a.jpg", "filename": "a.jpg",
                     "questionCategory": "桥梁", "bridgeName": "bridge", "defectLocation": ""}]
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            image_path = workspace / "data" / "raw" / manifest[0]["image"]
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (64, 64)).save(image_path)
            submitted_path = workspace / "result" / "result.json"
            submitted_path.parent.mkdir()
            submitted_path.write_bytes(b"preserve submitted result")
            artifact_dir = workspace / "runs" / "candidate"
            run_inference(workspace, self.FakeClient([response]), config, manifest, LEXICON,
                          artifact_dir=artifact_dir)
            summary = run_inference(workspace, self.FakeClient([]), config, manifest, LEXICON,
                                    artifact_dir=artifact_dir, resume=True)
            self.assertEqual(summary["processed_this_run"], 0)
            with self.assertRaisesRegex(ValueError, "Checkpoint"):
                run_inference(workspace, self.FakeClient([]), {**config, "global_max_side": 128},
                              manifest, LEXICON, artifact_dir=artifact_dir, resume=True)
            self.assertEqual(submitted_path.read_bytes(), b"preserve submitted result")

    def test_validate_result_accepts_exact_schema(self) -> None:
        manifest = [
            {
                "questionCategory": "桥梁",
                "bridgeName": "示例桥",
                "defectLocation": "",
                "filename": "a.jpg",
            }
        ]
        result = [
            {
                "questionCategory": "桥梁",
                "bridgeName": "示例桥",
                "defectLocation": "",
                "filename": "a.jpg",
                "defectType": "完好",
                "defectDescription": "未见明显病害",
                "ratingScale(1-5)": "",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "result").mkdir()
            (workspace / "logs").mkdir()
            path = workspace / "result" / "result.json"
            path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            report = validate_result(workspace, manifest, LEXICON, path)
            self.assertTrue(report["valid"])


if __name__ == "__main__":
    unittest.main()
