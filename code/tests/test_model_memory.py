from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track4_agent.model import ModelClient
from evaluate_holdout import _required_gpu_memory


class ModelMemoryTests(unittest.TestCase):
    def client(self, budget=None, reserve=1536):
        client = ModelClient.__new__(ModelClient)
        client.gpu_max_memory_mib = budget
        client.gpu_reserve_mib = reserve
        return client

    def test_default_allocation_is_unchanged(self):
        self.assertEqual(self.client().memory_plan(), {0: "11GiB", "cpu": "22GiB"})
        self.assertEqual(_required_gpu_memory({}), 10240)

    def test_capped_weight_budget_preserves_activation_reserve(self):
        self.assertEqual(self.client(6144).memory_plan(8000), {0: 6144 * 1024**2, "cpu": "22GiB"})
        with self.assertRaisesRegex(RuntimeError, "including reserve"):
            self.client(6144).memory_plan(7679)
        self.assertEqual(_required_gpu_memory({"gpu_max_memory_mib": 6144}), 7680)

    def test_cli_minimum_cannot_bypass_weight_budget_or_default_guard(self):
        self.assertEqual(_required_gpu_memory({}, 500), 10240)
        self.assertEqual(_required_gpu_memory({"gpu_max_memory_mib": 6144}, 500), 7680)
        self.assertEqual(_required_gpu_memory({"gpu_max_memory_mib": 6144}, 11000), 11000)

    def test_invalid_memory_values_are_rejected(self):
        for value in (0, -1, True, "6144", 6144.0):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.client(value).memory_plan()
                with self.assertRaises(ValueError):
                    self.client(6144, value).memory_plan()
        with self.assertRaises(ValueError):
            _required_gpu_memory({}, 0)


if __name__ == "__main__":
    unittest.main()
