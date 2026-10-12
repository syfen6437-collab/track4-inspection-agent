from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_inference_threads as benchmark


class ThreadBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "runs" / "threads" / "report.json"
        self.train = []
        for category, filename in (("桥梁", "bridge.jpg"), ("轨道", "track.jpg")):
            path = self.root / "data" / "raw" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture, decoded by neither fake client nor mocked inference")
            self.train.append({"image": filename, "filename": filename,
                               "questionCategory": category, "bridgeName": "", "defectLocation": "",
                               "defectType": "truth must not enter inference metadata"})
        self.threads = []
        fake_torch = SimpleNamespace(get_num_threads=lambda: 16,
                                     set_num_threads=self.threads.append,
                                     cuda=SimpleNamespace(synchronize=Mock()))
        self.client = SimpleNamespace(torch=fake_torch)
        for name, value in (("ROOT", self.root), ("load_assets", Mock(return_value=(self.train, [], {}))),
                             ("load_config", Mock(return_value={"model_id": "fixture"})),
                             ("load_client", Mock(return_value=self.client)),
                             ("_gpu_preflight", Mock()),
                             ("infer_item", Mock(return_value=({"defectType": "model generated"},
                                                               [{"raw": "fixed model response"}])))):
            replacement = patch.object(benchmark, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)

    def run_main(self, *extra):
        with patch.object(sys, "argv", ["benchmark", "--output", str(self.output), *extra]):
            benchmark.main()

    def test_abba_measurements_keep_category_images_fixed_and_exclude_targets(self):
        self.run_main()
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(self.threads, [16, 4, 4, 16, 16, 4, 4, 16])
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["output_consistent"])
        self.assertEqual(len(report["records"]), 8)
        self.assertEqual([row["category"] for row in report["records"]], ["桥梁"] * 4 + ["轨道"] * 4)
        for call in benchmark.infer_item.call_args_list:
            self.assertNotIn("defectType", call.args[1])
            self.assertFalse(call.kwargs["allow_review"])

    def test_inconsistent_model_outputs_are_reported_not_rewritten(self):
        benchmark.infer_item.side_effect = [({"defectType": str(index)}, [{"raw": str(index)}])
                                           for index in range(8)]
        self.run_main()
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertFalse(report["output_consistent"])
        self.assertEqual(report["records"][0]["prediction"]["defectType"], "0")
        self.assertEqual(report["records"][-1]["prediction"]["defectType"], "7")

    def test_existing_output_is_preserved_before_gpu_load(self):
        self.output.parent.mkdir(parents=True)
        self.output.write_bytes(b"preserve previous diagnostic")
        with self.assertRaises(SystemExit):
            self.run_main()
        self.assertEqual(self.output.read_bytes(), b"preserve previous diagnostic")
        benchmark.load_client.assert_not_called()

    def test_formal_output_and_invalid_threads_are_rejected(self):
        self.output = self.root / "result" / "result.json"
        with self.assertRaisesRegex(ValueError, "inside workspace/runs"):
            self.run_main()
        benchmark.load_client.assert_not_called()
        self.output = self.root / "runs" / "threads" / "report.json"
        with self.assertRaises(SystemExit):
            self.run_main("--threads", "0")
        benchmark.load_client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
