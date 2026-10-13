from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import review_track_candidate as runner
from track4_agent.track_consensus import adapter_decision
from track4_agent.vision_review import digest_json, sha256_file, strict_prediction


class TrackReviewResumeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.out = self.root / "runs" / "review"
        self.out.mkdir(parents=True)
        self.base_path = self.root / "result.json"
        self.candidate_path = self.root / "candidate.json"
        self.adapter_path = self.root / "adapter.json"
        self.output, self.audit = self.out / "result.json", self.out / "audit.json"
        self.manifest = [{"id": f"track/{name}", "filename": name, "image": f"track/{name}",
                          "questionCategory": "轨道", "bridgeName": "", "defectLocation": ""}
                         for name in ("a.jpg", "b.jpg")]
        self.lexicon = {"categories": {"轨道": {"allowed_labels": ["完好", "破损", "支座锈蚀"],
                           "label_specs": {l: {"default_rating": ""} for l in ("完好", "破损", "支座锈蚀")}}}}
        self.base = [{"questionCategory": "轨道", "bridgeName": "", "defectLocation": "", "filename": i["filename"],
                      "defectType": "破损", "defectDescription": "模型原始描述", "ratingScale(1-5)": ""}
                     for i in self.manifest]
        self.candidate = [{**r, "defectType": "支座锈蚀"} for r in self.base]
        for path, rows in ((self.base_path, self.base), (self.candidate_path, self.candidate)):
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        score = {"label": "支座锈蚀", "confidence": 0.9}
        decisions = [adapter_decision(row, score, score) for row in self.base]
        self.adapter_path.write_text(json.dumps({"base_result_sha256": sha256_file(self.base_path),
            "training_manifest_sha256": digest_json([]), "threshold": 0.55, "records": decisions}), encoding="utf-8")
        self.config = {"gpu_max_memory_mib": 1024, "gpu_reserve_mib": 1024}

    def review(self, index):
        raw = json.dumps({"defectType": "支座锈蚀", "defectDescription": "支座表面锈蚀", "ratingScale": "",
                          "confidence": 0.9, "evidence": "支座锈蚀"}, ensure_ascii=False)
        item = self.manifest[index]
        return {"id": item["id"], "filename": item["filename"], "vision_candidate": "支座锈蚀",
                "vision_confidence": 0.9, "raw": raw, "attempts": [raw], "valid": True,
                "prediction": strict_prediction(raw, self.lexicon, item)}

    def run_cli(self, responses=(), flags=(), context=None):
        argv = ["review", "--config", str(self.root / "config.json"), "--base-result", str(self.base_path),
                "--candidate", str(self.candidate_path), "--adapter-audit", str(self.adapter_path),
                "--output", str(self.output), "--audit", str(self.audit), *flags]
        factory = Mock(return_value=object())
        generated = Mock(side_effect=list(responses))
        with patch.object(runner, "ROOT", self.root), patch.object(sys, "argv", argv), \
                patch.object(runner, "load_assets", return_value=([], self.manifest, self.lexicon)), \
                patch.object(runner, "load_config", return_value=self.config), \
                patch.object(runner, "review_context", return_value=context or {"binding": "fixture"}), \
                patch.object(runner, "_gpu_preflight"), patch.object(runner, "load_client", factory), \
                patch.object(runner, "review_image", generated):
            runner.main()
        return factory, generated

    def test_interruption_resumes_only_missing_and_complete_replay_avoids_gpu(self):
        original = self.base_path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "interrupt"):
            self.run_cli([self.review(0), RuntimeError("interrupt")])
        state = json.loads((self.out / "audit.checkpoint.json").read_text(encoding="utf-8"))
        self.assertEqual(len(state["records"]), 1)
        self.assertFalse(self.output.exists())
        factory, generated = self.run_cli([self.review(1)], ["--resume"])
        self.assertEqual(generated.call_count, 1)
        factory, generated = self.run_cli(flags=["--resume", "--revalidate-only"])
        factory.assert_not_called()
        generated.assert_not_called()
        self.assertEqual(self.base_path.read_bytes(), original)
        self.assertEqual(json.loads(self.audit.read_text(encoding="utf-8"))["accepted_count"], 2)

    def test_changed_context_or_edited_cache_is_rejected_before_gpu(self):
        self.run_cli([self.review(0), self.review(1)])
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_cli(flags=["--resume"], context={"binding": "changed model"})
        path = self.out / "audit.checkpoint.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved["records"][0]["prediction"]["confidence"] = 1.0
        path.write_text(json.dumps(saved), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.run_cli(flags=["--resume"])

    def test_incomplete_revalidation_never_loads_model(self):
        with self.assertRaisesRegex(RuntimeError, "interrupt"):
            self.run_cli([self.review(0), RuntimeError("interrupt")])
        with self.assertRaisesRegex(ValueError, "complete checkpoints"):
            self.run_cli(flags=["--resume", "--revalidate-only"])


if __name__ == "__main__":
    unittest.main()
