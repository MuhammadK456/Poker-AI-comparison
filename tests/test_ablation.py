import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from poker_ai.dqn import DQNAgent, DQNConfig, DoubleDQN
from poker_ai.environment import FULL_DECK, PokerHand
from poker_ai.features import FeatureEncoder
from poker_ai.train import run_pilot
from test_learning import FixedEquity


class NoEquityTests(unittest.TestCase):
    def test_only_equity_changes_and_calculator_is_never_called(self):
        class ForbiddenEquity:
            def from_observation(self, obs):
                raise AssertionError("ablated encoder consulted equity")
        obs = PokerHand(FULL_DECK).observe()
        original = FeatureEncoder(FixedEquity()).encode(obs)
        ablated = FeatureEncoder(ForbiddenEquity(), include_equity=False).encode(obs)
        self.assertNotEqual(original[117], 0)
        original[117] = 0
        np.testing.assert_array_equal(original, ablated)

    def test_checkpoint_preserves_feature_setting_and_rejects_mismatch(self):
        torch.set_num_threads(1)
        learner = DoubleDQN(include_equity=False)
        regular = DoubleDQN()
        for name, value in learner.online.state_dict().items():
            torch.testing.assert_close(value, regular.online.state_dict()[name], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            learner.save(path, 0, {})
            restored = DoubleDQN.load_for_inference(path)
            self.assertFalse(restored.include_equity)
            with self.assertRaisesRegex(ValueError, "does not match"):
                DQNAgent(restored, FeatureEncoder(FixedEquity()))
            encoder = FeatureEncoder(FixedEquity(), include_equity=False)
            obs = PokerHand(FULL_DECK).observe()
            self.assertEqual(DQNAgent(learner, encoder).act(obs), DQNAgent(restored, encoder).act(obs))
            saved = torch.load(path, weights_only=True)
            saved.pop("include_equity")
            torch.save(saved, path)
            self.assertTrue(DoubleDQN.load_for_inference(path).include_equity)

    def test_training_validation_and_reload_use_zero_equity(self):
        from unittest.mock import patch
        original = FeatureEncoder.encode
        seen = []
        def checked(encoder, obs):
            result = original(encoder, obs)
            if not encoder.include_equity:
                self.assertEqual(result[117], 0)
                seen.append(1)
            return result
        config = DQNConfig(batch_size=8, replay_capacity=64, warmup=16, target_every=8)
        with tempfile.TemporaryDirectory() as temp, patch.object(FeatureEncoder, "encode", checked):
            root = Path(temp) / "run"
            run_pilot(root, 32, 33, 2, config, validation_every=16, threads=1, include_equity=False)
            self.assertGreater(len(seen), 32)
            self.assertFalse(DoubleDQN.load_for_inference(root / "final.pt").include_equity)
            self.assertFalse(json.loads((root / "run-config.json").read_text())["include_equity"])
