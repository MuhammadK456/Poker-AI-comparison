from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from poker_ai.agents import PassiveAgent
from poker_ai.dqn import DQNAgent, DQNConfig, DoubleDQN, double_dqn_targets, masked_argmax
from poker_ai.environment import Action, FULL_DECK, PokerHand, PublicAction, shuffled_deck
from poker_ai.equity import EquityCalculator, EquityEstimate
from poker_ai.features import CARD_INDEX, FEATURE_SIZE, FeatureEncoder
from poker_ai.learning_env import LearnerEpisode
from poker_ai.randomness import derived_seed
from poker_ai.replay import ReplayBuffer
from poker_ai.train import run_pilot

torch.set_num_threads(2)


class FixedEquity:
    def from_observation(self, obs):
        return EquityEstimate(50, 0, 50, "monte_carlo")


class FoldAgent:
    def act(self, obs):
        return Action.FOLD if obs.call_amount else Action.CHECK_CALL


def feature(mask=(False, True, False, False, False, False)):
    values = np.zeros(FEATURE_SIZE, dtype=np.float32)
    values[157:] = mask
    return values


class FeatureTests(unittest.TestCase):
    def test_schema_positions_and_relative_stacks(self):
        obs = PokerHand(FULL_DECK).observe()
        values = FeatureEncoder(FixedEquity()).encode(obs)
        self.assertEqual(values.shape, (163,))
        self.assertEqual(values.dtype, np.float32)
        self.assertEqual(values[:52].sum(), 2)
        for card in obs.hole_cards:
            self.assertEqual(values[CARD_INDEX[card]], 1)
        self.assertEqual(values[52:104].sum(), 0)
        np.testing.assert_array_equal(values[104:109], [1, 0, 0, 0, 1])
        np.testing.assert_allclose(values[109:117], [3/200, 1/200, 199/200, 198/200, 1/200, 2/200, 198/200, 0.25])
        self.assertEqual(values[117], 0.5)
        self.assertEqual(values[118], 1)
        np.testing.assert_array_equal(values[157:], obs.action_mask)

    def test_history_counts_actor_relative_and_last_action_amount(self):
        obs = PokerHand(FULL_DECK).observe()
        history = (PublicAction(1, 0, "raise", 3, 4, Action.MIN_RAISE),
                   PublicAction(0, 0, "call", 2, None, Action.CHECK_CALL),
                   PublicAction(0, 1, "bet", 4, 4, Action.POT))
        encoded = FeatureEncoder(FixedEquity()).encode(replace(obs, history=history))
        self.assertEqual(encoded[125 + 2], 1)  # own preflop raise
        self.assertEqual(encoded[125 + 4 + 1], 1)  # opponent preflop call
        self.assertEqual(encoded[125 + 8 + 4 + 2], 1)  # opponent flop bet
        self.assertEqual(encoded[118 + 4], 1)
        self.assertAlmostEqual(encoded[124], 4 / 200)

    def test_hidden_cards_do_not_affect_features(self):
        original = list(FULL_DECK)
        other = original.copy()
        other[0], other[20] = other[20], other[0]
        other[2], other[21] = other[21], other[2]
        other[5], other[30] = other[30], other[5]
        encoder = FeatureEncoder(EquityCalculator())
        np.testing.assert_array_equal(encoder.encode(PokerHand(tuple(original)).observe()),
                                      encoder.encode(PokerHand(tuple(other)).observe()))


class LearnerTransitionTests(unittest.TestCase):
    def test_next_observation_is_learner_not_opponent(self):
        episode = LearnerEpisode(FULL_DECK, 1, PassiveAgent())
        self.assertEqual(episode.observation.seat, 1)
        step = episode.step(int(Action.CHECK_CALL))
        self.assertFalse(step.terminal)
        self.assertEqual(step.reward, 0)
        self.assertEqual((step.observation.seat, step.observation.street), (1, 1))
        # Opponent closes preflop, then checks first on flop before learner acts.
        self.assertEqual([(a.seat, a.street, a.kind) for a in step.observation.history],
                         [(1, 0, "call"), (0, 0, "check"), (0, 1, "check")])

    def test_blind_loss_and_no_terminal_bootstrap_observation(self):
        episode = LearnerEpisode(FULL_DECK, 1, PassiveAgent())
        step = episode.step(int(Action.FOLD))
        self.assertTrue(step.terminal)
        self.assertIsNone(step.observation)
        self.assertEqual(step.reward, -1 / 200)
        with self.assertRaises(ValueError):
            episode.step(int(Action.CHECK_CALL))

    def test_opponent_fold_credits_prior_learner_action(self):
        episode = LearnerEpisode(FULL_DECK, 1, FoldAgent())
        step = episode.step(int(Action.MIN_RAISE))
        self.assertTrue(step.terminal)
        self.assertEqual(step.reward, 2 / 200)

    def test_no_decision_hand_does_not_invent_a_transition(self):
        episode = LearnerEpisode(FULL_DECK, 0, FoldAgent())
        self.assertIsNone(episode.observation)
        self.assertEqual(episode.result().net_chips, (1, -1))

    def test_only_terminal_step_has_hand_reward(self):
        episode = LearnerEpisode(shuffled_deck(24), 0, PassiveAgent())
        rewards = []
        while episode.observation is not None:
            step = episode.step(int(Action.CHECK_CALL))
            rewards.append(step.reward)
        self.assertEqual(rewards[:-1], [0] * (len(rewards) - 1))
        self.assertEqual(sum(rewards), episode.result().net_chips[0] / 200)


class ReplayTests(unittest.TestCase):
    def test_ring_bound_copy_sampling_and_memory(self):
        replay = ReplayBuffer(3)
        values = feature()
        for i in range(5):
            values[0] = i
            replay.add(values, values[157:], 1, 0, values, values[157:], False)
        values[0] = 999
        self.assertEqual((replay.size, replay.cursor), (3, 2))
        self.assertEqual(set(replay.observations[:, 0]), {2, 3, 4})
        batch = replay.sample(3, np.random.default_rng(1))
        self.assertEqual(set(batch["observations"][:, 0]), {2, 3, 4})
        batch["observations"][:] = 999
        self.assertNotIn(999, replay.observations)
        self.assertEqual(replay.nbytes, 3 * (2*163*4 + 2*6 + 8 + 4 + 1))

    def test_reject_illegal_actions_and_malformed_terminal(self):
        replay = ReplayBuffer(3)
        values, zeros = feature(), np.zeros(163, np.float32)
        with self.assertRaises(ValueError):
            replay.add(values, values[157:], 0, 0, values, values[157:], False)
        with self.assertRaises(ValueError):
            replay.add(values, values[157:], 1, 0.5, values, values[157:], False)
        with self.assertRaises(ValueError):
            replay.add(values, values[157:], 1, 0.5, values, values[157:], True)
        with self.assertRaises(ValueError):
            replay.add(values, values[157:], 1, 0, zeros, zeros[157:], False)
        replay.add(values, values[157:], 1, -0.005, zeros, zeros[157:], True)
        self.assertTrue(replay.terminals[0])


class DQNTests(unittest.TestCase):
    def test_masked_argmax_and_legal_exploration(self):
        values = torch.tensor([[999.0, 2, 3, 4, 5, 6]])
        mask = torch.tensor([[False, True, True, False, False, False]])
        self.assertEqual(int(masked_argmax(values, mask)[0]), 2)
        with self.assertRaises(ValueError):
            masked_argmax(values, torch.zeros_like(mask))
        learner = DoubleDQN()
        rng = random.Random(2)
        choices = {learner.choose(feature(mask[0].numpy()), mask[0].numpy(), 1, rng) for _ in range(100)}
        self.assertEqual(choices, {1, 2})
        self.assertIn(learner.choose(feature(mask[0].numpy()), mask[0].numpy(), 0, rng), {1, 2})

    def test_double_dqn_online_selects_target_evaluates_and_terminals_skip(self):
        class Constant(nn.Module):
            def __init__(self, values):
                super().__init__()
                self.values = torch.tensor([values], dtype=torch.float32)

            def forward(self, states):
                self.assert_finite(states)
                return self.values.expand(len(states), -1)

            @staticmethod
            def assert_finite(states):
                if not torch.isfinite(states).all():
                    raise RuntimeError("terminal rows were incorrectly evaluated")

        online = Constant([999, 2, 3, 0, 0, 0])
        target = Constant([999, 10, 0.7, 0, 0, 0])
        states = torch.zeros((2, 163))
        states[1] = torch.nan
        masks = torch.tensor([[False, True, True, False, False, False], [False]*6])
        results = double_dqn_targets(online, target, states, masks, torch.tensor([0.0, -0.5]), torch.tensor([False, True]))
        torch.testing.assert_close(results, torch.tensor([0.7, -0.5]))

    def test_network_size_schedule_target_sync_and_synthetic_fit(self):
        config = DQNConfig(learning_rate=0.01, batch_size=8, replay_capacity=32, warmup=8, target_every=2)
        learner = DoubleDQN(config)
        self.assertEqual(sum(p.numel() for p in learner.online.parameters()), 38278)
        self.assertEqual(config.epsilon(0), 1)
        self.assertAlmostEqual(config.epsilon(100_000), 0.05)
        self.assertAlmostEqual(config.epsilon(200_000), 0.05)
        replay = ReplayBuffer(16)
        zeros = np.zeros(163, np.float32)
        for i in range(16):
            values = feature()
            values[i % 2] = 1
            replay.add(values, values[157:], 1, 0.75 if i % 2 else -0.25, zeros, zeros[157:], True)
        batch = replay.sample(16, np.random.default_rng(10))
        initial_target = {k: v.clone() for k, v in learner.target.state_dict().items()}
        first = learner.update(batch)["loss"]
        for key, value in learner.target.state_dict().items():
            torch.testing.assert_close(value, initial_target[key])
        learner.update(batch)
        for key, value in learner.target.state_dict().items():
            torch.testing.assert_close(value, learner.online.state_dict()[key])
        for _ in range(198):
            final = learner.update(batch)["loss"]
        self.assertLess(final, 0.001)
        self.assertLess(final, first / 10)
        self.assertTrue(all(p.grad is None for p in learner.target.parameters()))

    def test_checkpoint_roundtrip_and_frozen_inference(self):
        learner = DoubleDQN()
        encoder = FeatureEncoder(FixedEquity())
        obs = PokerHand(FULL_DECK).observe()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.pt"
            learner.save(path, 0, {"test": True})
            restored = DoubleDQN.load_for_inference(path)
            expected = DQNAgent(learner, encoder).act(obs)
            actual = DQNAgent(restored, encoder).act(obs)
            self.assertEqual(expected, actual)
            self.assertEqual(restored.gradient_updates, 0)
            for key, value in learner.online.state_dict().items():
                torch.testing.assert_close(value, restored.online.state_dict()[key])
            with self.assertRaises(FileExistsError):
                learner.save(path, 0, {})


class PilotTests(unittest.TestCase):
    def test_small_end_to_end_pilot_reproduces_training_and_validation(self):
        config = DQNConfig(batch_size=8, replay_capacity=64, warmup=16, target_every=8)
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a", Path(temp) / "b"
            first = run_pilot(a, decisions=64, seed=17, validation_pairs=2, config=config)
            second = run_pilot(b, decisions=64, seed=17, validation_pairs=2, config=config)
            self.assertGreater(first["gradient_updates"], 0)
            self.assertGreaterEqual(first["actual_decisions"], 64)
            self.assertEqual(first["hands"] - first["no_decision_hands"], first["terminal_transitions"])
            self.assertEqual(first["validation"], second["validation"])
            self.assertEqual((a / "training-hands.csv").read_bytes(), (b / "training-hands.csv").read_bytes())
            self.assertEqual((a / "validation-final.csv").read_bytes(), (b / "validation-final.csv").read_bytes())
            one, two = DoubleDQN.load_for_inference(a / "final.pt"), DoubleDQN.load_for_inference(b / "final.pt")
            for key, value in one.online.state_dict().items():
                torch.testing.assert_close(value, two.online.state_dict()[key], rtol=0, atol=0)
            self.assertNotEqual(derived_seed(17, "dqn_training_deal", 0), derived_seed(17, "dqn_validation_deal", 0))
