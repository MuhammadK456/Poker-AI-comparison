from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

from poker_ai.demo import markdown, replay
from poker_ai.dqn import DoubleDQN
from poker_ai.equity import EquityConfig
from poker_ai.traditional import TraditionalConfig


class DemoTests(unittest.TestCase):
    def test_demo_is_reproducible_and_preserves_action_and_chip_records(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.pt"
            DoubleDQN(seed=33).save(path, 0, {"equity": asdict(EquityConfig()),
                "traditional_snapshot": {"config": asdict(TraditionalConfig())}})
            one, two = replay(path, 1, 27), replay(path, 1, 27)
            self.assertEqual(one, two)
            self.assertEqual([hand["ml_seat"] for hand in one["hands"]], [0, 1])
            for hand in one["hands"]:
                self.assertEqual(sum(hand["result"]["final_stacks"]), 400)
                self.assertEqual(sum(hand["result"]["net_chips"]), 0)
                self.assertEqual(len(hand["decisions"]), len(hand["result"]["history"]))
                for row in hand["decisions"]:
                    self.assertEqual(row["slot"], row["action"]["slot"])
                    self.assertEqual(row["observation"]["seat"], row["action"]["seat"])
            self.assertIn("not a win probability", markdown(one))
