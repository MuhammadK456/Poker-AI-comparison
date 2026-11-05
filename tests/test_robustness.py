import json
from pathlib import Path
import tempfile
import unittest

from poker_ai.evaluation_report import audit_logs
from poker_ai.robustness import make_variant, worker
from poker_ai.train import ROOT


class RobustnessTests(unittest.TestCase):
    def test_frozen_variants_and_deterministic_complete_pairs(self):
        spec = json.loads((ROOT / "configs/robustness-development.json").read_text())
        spec["pairs"] = 2
        baseline = json.loads((ROOT / "configs/traditional-v1.json").read_text())["config"]
        self.assertEqual(len(spec["opponents"]), 3)
        for variant in spec["opponents"].values():
            agent = make_variant(variant, None, 0, baseline)
            for key, value in variant.items():
                if key != "kind":
                    self.assertEqual(getattr(agent.config, key), value)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            one = worker(root, "one", None, spec, baseline)
            two = worker(root, "two", None, spec, baseline)
            self.assertEqual(one["groups"], two["groups"])
            self.assertEqual((root / "validation-one.csv").read_bytes(), (root / "validation-two.csv").read_bytes())
            audit_logs(root, ["one", "two"], list(spec["opponents"]), 2)
