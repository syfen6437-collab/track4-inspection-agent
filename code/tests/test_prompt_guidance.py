from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_holdout import _holdout_lexicon, _input_item, _label_coverage, _variants
from track4_agent.inference import infer_item
from track4_agent.model import (
    build_presence_prompt, build_prompt, build_track_atomic_prompt,
    track_atomic_prediction, training_description_prompt,
)
from evaluate_track_prompt import select_track_sample


class PromptGuidanceTests(unittest.TestCase):
    def setUp(self):
        self.atoms = ["完好", "破损", "渗水泛碱", "钢筋锈蚀", "裂缝(混凝土裂缝)"]
        labels = self.atoms + ["钢筋锈蚀、破损", "钢筋锈蚀、破损、渗水泛碱"]
        self.lexicon = {"categories": {"轨道": {
            "allowed_labels": labels,
            "label_specs": {x: {"count": 1, "descriptions": [], "rating_distribution": {"": 1},
                                "default_rating": ""} for x in labels},
        }, "桥梁": {
            "allowed_labels": ["完好", "已处治病害（修补）", "渗水/泛碱"],
            "label_specs": {x: {"count": 1, "descriptions": [d], "rating_distribution": {"": 1},
                                "default_rating": ""} for x, d in (
                ("完好", "表面完整"), ("已处治病害（修补）", "矩形孔洞封堵块"),
                ("渗水/泛碱", "接缝处水渍析出物"))},
        }}}
        self.item = {"questionCategory": "轨道", "filename": "photo.jpg", "bridgeName": "",
                     "image": "photo.jpg", "id": "photo.jpg", "defectLocation": ""}

    def response(self, selected):
        return json.dumps({"selected": selected, "defectDescription": "钢筋锈蚀及破损",
                           "ratingScale": "", "confidence": 0.9}, ensure_ascii=False)

    def test_atomic_prompt_uses_atoms_not_long_combinations(self):
        prompt = build_track_atomic_prompt(self.item, self.lexicon)
        self.assertIn("分别检查每一项", prompt)
        self.assertNotIn("钢筋锈蚀、破损、渗水泛碱", prompt)
        for atom in self.atoms:
            self.assertIn(atom, prompt)

    def test_combination_order_is_normalized_without_losing_atoms(self):
        raw = self.response(["破损", "渗水泛碱", "钢筋锈蚀"])
        prediction, valid = track_atomic_prediction(raw, self.lexicon)
        self.assertTrue(valid)
        self.assertEqual(prediction["defectType"], "钢筋锈蚀、破损、渗水泛碱")
        self.assertEqual(prediction["dropped_atoms"], [])
        self.assertEqual(prediction["raw"], raw)

    def test_unsupported_combination_never_adds_a_defect(self):
        prediction, valid = track_atomic_prediction(self.response(["钢筋锈蚀", "裂缝(混凝土裂缝)"]), self.lexicon)
        self.assertTrue(valid)
        self.assertIn(prediction["defectType"], ["钢筋锈蚀", "裂缝(混凝土裂缝)"])
        self.assertEqual(len(prediction["dropped_atoms"]), 1)
        self.assertNotIn("破损", prediction["defectType"])

    def test_healthy_is_mutually_exclusive_and_duplicates_are_ignored(self):
        prediction, valid = track_atomic_prediction(self.response(["完好", "破损", "破损"]), self.lexicon)
        self.assertTrue(valid)
        self.assertEqual(prediction["defectType"], "破损")
        self.assertEqual(prediction["selected_atoms"], ["破损"])

    def test_invalid_atom_output_is_not_reported_as_valid(self):
        for selected in ([], "破损", ["不存在的标签"], [None], ["破损", "不存在的标签"]):
            with self.subTest(selected=selected):
                prediction, valid = track_atomic_prediction(self.response(selected), self.lexicon)
                self.assertFalse(valid)
                self.assertIn(prediction["defectType"], self.lexicon["categories"]["轨道"]["allowed_labels"])
        self.assertFalse(track_atomic_prediction("not json", self.lexicon)[1])

    def test_description_guidance_is_opt_in_and_bridge_only(self):
        bridge = {**self.item, "questionCategory": "桥梁"}
        self.assertNotIn("标签释义示例", build_prompt(bridge, self.lexicon))
        self.assertIn("矩形孔洞封堵块", build_prompt(bridge, self.lexicon, training_examples=True))
        self.assertIn("矩形孔洞封堵块", build_presence_prompt(bridge, lexicon=self.lexicon, training_examples=True))
        self.assertEqual(training_description_prompt(self.item, self.lexicon), "")

    def test_heldout_targets_do_not_enter_guidance(self):
        bridge = {**self.item, "questionCategory": "桥梁", "defectType": "渗水/泛碱",
                  "defectDescription": "HOLDOUT_SECRET_DESCRIPTION", "ratingScale(1-5)": "3"}
        fitting = [{**bridge, "defectDescription": "FITTING_DESCRIPTION"}]
        lexicon = _holdout_lexicon(fitting, self.lexicon)
        prompt = build_prompt(_input_item(bridge), lexicon, training_examples=True)
        self.assertIn("FITTING_DESCRIPTION", prompt)
        self.assertNotIn("HOLDOUT_SECRET_DESCRIPTION", prompt)
        self.assertNotIn("矩形孔洞封堵块", prompt)

    def test_atom_inference_has_only_formal_seven_fields_and_raw_provenance(self):
        class Client:
            def generate(client, images, prompt):
                self.assertIn('"selected"', prompt)
                return self.response(["钢筋锈蚀", "破损"])
        config = {"confidence_threshold": 0.55, "track_atomic_checklist": True}
        with patch("track4_agent.inference._first_pass_images", return_value=[]):
            prediction, raw = infer_item(Client(), self.item, self.lexicon, config, Path("photo.jpg"), allow_review=False)
        self.assertEqual(len(prediction), 7)
        self.assertEqual(prediction["defectType"], "钢筋锈蚀、破损")
        self.assertTrue(raw[0]["valid"])
        self.assertEqual(raw[0]["prediction"]["selected_atoms"], ["破损", "钢筋锈蚀"])

    def test_variant_and_coverage_use_full_legal_track_ontology(self):
        variants = _variants(["baseline", "training_evidence"], {})
        self.assertNotIn("track_atomic_checklist", variants["baseline"])
        self.assertTrue(variants["training_evidence"]["track_atomic_checklist"])
        row = {**self.item, "defectType": "钢筋锈蚀、破损、渗水泛碱"}
        self.assertEqual(_label_coverage([row], self.lexicon, variants["training_evidence"])["轨道"]["included_targets"], 1)

    def test_track_definitions_are_opt_in_and_never_change_bridge_prompts(self):
        plain = build_prompt(self.item, self.lexicon)
        improved = build_prompt(self.item, self.lexicon, track_definitions=True)
        self.assertNotIn("轨道病害判断标准", plain)
        self.assertIn("支座钢板", improved)
        self.assertIn("金属的旧漆", improved)
        bridge = {**self.item, "questionCategory": "桥梁"}
        self.assertEqual(build_prompt(bridge, self.lexicon), build_prompt(bridge, self.lexicon, track_definitions=True))

    def test_track_prompt_sampling_is_label_blind_and_excludes_prior_diagnostic(self):
        rows = [{"image": str(i), "questionCategory": "轨道", "defectType": "a"} for i in range(20)]
        selected = select_track_sample(rows, 8, 17, ["1", "2"])
        changed = select_track_sample([{**r, "defectType": "b"} for r in rows], 8, 17, ["1", "2"])
        self.assertEqual([r["image"] for r in selected], [r["image"] for r in changed])
        self.assertFalse({r["image"] for r in selected} & {"1", "2"})


if __name__ == "__main__":
    unittest.main()
