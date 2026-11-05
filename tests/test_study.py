import json
from pathlib import Path
import tempfile
import unittest

import torch

from poker_ai.diagnose import describe_action
from poker_ai.dqn import DQNConfig, DoubleDQN
from poker_ai.environment import Action, FULL_DECK, GameConfig, PokerHand
from poker_ai.features import FeatureEncoder
from poker_ai.study import aggregate, validate_spec
from poker_ai.train import ROOT, run_pilot
from test_learning import FixedEquity


class StudyTests(unittest.TestCase):
    def test_diagnostics_recognize_deduplicated_allin(self):
        hand = PokerHand(FULL_DECK, GameConfig((5, 200)))
        hand.step(Action.MIN_RAISE)
        obs = hand.observe()
        values = FeatureEncoder(FixedEquity()).encode(obs)
        values[117] = 0.2
        row = describe_action(obs, Action.MIN_RAISE, values, [0, 0, 2, 0, 0, 0])
        self.assertTrue(row["effective_allin_raise"])
        self.assertTrue(row["allin_below_50_equity"])
        self.assertTrue(row["selected_q_outside_return_bounds"])
        self.assertEqual(row["action"], "raise")

    def test_periodic_validation_leaves_training_identical(self):
        config = DQNConfig(batch_size=8, replay_capacity=64, warmup=16, target_every=8)
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a", Path(temp) / "b"
            run_pilot(a, 64, 33, 2, config, validation_seed=99, threads=1)
            run_pilot(b, 64, 33, 2, config, validation_every=16, validation_seed=99, threads=1)
            self.assertEqual((a / "training-hands.csv").read_bytes(), (b / "training-hands.csv").read_bytes())
            one, two = DoubleDQN.load_for_inference(a / "final.pt"), DoubleDQN.load_for_inference(b / "final.pt")
            for key, value in one.online.state_dict().items():
                torch.testing.assert_close(value, two.online.state_dict()[key], rtol=0, atol=0)
            rows = [json.loads(line) for line in (b / "learning-curve.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[0]["decisions"], 0)
            self.assertGreaterEqual(rows[-1]["decisions"], 64)
            self.assertEqual(rows[-1]["checkpoint"], "final.pt")
            self.assertTrue(all((b / row["checkpoint"]).exists() for row in rows))

    def test_seed_summary_does_not_pool_hands_and_requires_same_deals(self):
        spec = json.loads((ROOT / "configs" / "dqn-three-seed.json").read_text())
        validate_spec(spec)
        with self.assertRaises(ValueError):
            validate_spec({**spec, "seeds": [101, 101]})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for seed, value in ((101, 10), (202, 30), (303, 50)):
                folder = root / f"seed-{seed}"
                folder.mkdir()
                payload = {"validation_seed": 99, "actual_decisions": 250000, "training_seconds": 1,
                           "peak_process_rss_bytes": 100,
                           "validation": {"final": {"passive": {"bb_per_100": value, "ci95": [value-1, value+1],
                                                               "difference_vs_traditional": value, "difference_ci95": [value-1, value+1]}}}}
                (folder / "summary.json").write_text(json.dumps(payload))
            summary = aggregate(root, [101, 202, 303])
            self.assertEqual(summary["total_actual_decisions"], 750000)
            self.assertEqual(summary["results"]["passive"]["mean"], 30)
            self.assertEqual(summary["results"]["passive"]["sample_sd"], 20)
            payload["validation_seed"] = 98
            (root / "seed-303" / "summary.json").write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                aggregate(root, [101, 202, 303])
