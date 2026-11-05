import csv
from pathlib import Path
import tempfile
import unittest

from poker_ai.final_eval import bootstrap_intervals, worker
from poker_ai.randomness import derived_seed
from poker_ai.traditional import TraditionalConfig


class FinalEvaluationTests(unittest.TestCase):
    def test_batched_bootstrap_determinism_and_nested_intervals(self):
        values = [-100, -50, 0, 50, 100]
        one = bootstrap_intervals(values, 17, 103, (.95, .9833333333333333))
        self.assertEqual(one, bootstrap_intervals(values, 17, 103, (.95, .9833333333333333)))
        narrow, wide = one["0.95"], one["0.9833333333333333"]
        self.assertLessEqual(wide[0], narrow[0])
        self.assertGreaterEqual(wide[1], narrow[1])
        self.assertEqual(bootstrap_intervals([25, 25], 17, 35)["0.95"], [25, 25])

    def test_measurement_does_not_change_deterministic_games(self):
        spec = {"pairs": 2, "seed": 17, "deal_namespace": "final_runner_development_fixture"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for label in ("one", "two"):
                worker(root, label, "random", None, spec, TraditionalConfig())
            self.assertEqual((root / "part-one-random.csv").read_bytes(), (root / "part-two-random.csv").read_bytes())
            with (root / "part-one-random.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            for deal in (0, 1):
                pair = rows[2*deal:2*deal+2]
                self.assertEqual({row["hero_seat"] for row in pair}, {"0", "1"})
                self.assertEqual({int(row["deck_seed"]) for row in pair},
                                 {derived_seed(17, spec["deal_namespace"], deal)})
