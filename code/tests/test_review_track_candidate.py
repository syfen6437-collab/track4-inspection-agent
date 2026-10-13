from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review_track_candidate import _validate_candidate


class ReviewTrackCandidateTests(unittest.TestCase):
    def setUp(self):
        self.manifest = [
            {"filename": "bridge.jpg", "questionCategory": "桥梁", "bridgeName": "B", "defectLocation": ""},
            {"filename": "track.jpg", "questionCategory": "轨道", "bridgeName": "", "defectLocation": ""},
        ]
        self.lexicon = {"categories": {"桥梁": {"allowed_labels": ["完好"]},
                                        "轨道": {"allowed_labels": ["完好", "破损"]}}}
        self.base = [
            {"questionCategory": "桥梁", "bridgeName": "B", "defectLocation": "", "filename": "bridge.jpg",
             "defectType": "完好", "defectDescription": "模型", "ratingScale(1-5)": ""},
            {"questionCategory": "轨道", "bridgeName": "", "defectLocation": "", "filename": "track.jpg",
             "defectType": "完好", "defectDescription": "模型", "ratingScale(1-5)": ""},
        ]

    def test_bridge_changes_are_rejected_but_track_label_changes_are_allowed(self):
        candidate = [dict(self.base[0]), dict(self.base[1])]
        candidate[1]["defectType"] = "破损"
        _validate_candidate(self.base, candidate, self.manifest, self.lexicon)
        candidate[0]["defectType"] = "完好"
        candidate[0]["defectDescription"] = "changed"
        with self.assertRaisesRegex(ValueError, "Bridge row changed"):
            _validate_candidate(self.base, candidate, self.manifest, self.lexicon)

    def test_illegal_candidate_label_is_rejected(self):
        candidate = [dict(row) for row in self.base]
        candidate[1]["defectType"] = "非法"
        with self.assertRaisesRegex(ValueError, "illegal"):
            _validate_candidate(self.base, candidate, self.manifest, self.lexicon)


if __name__ == "__main__":
    unittest.main()
