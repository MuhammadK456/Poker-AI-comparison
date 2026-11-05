from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
import tempfile
import unittest

from poker_ai.environment import Action, FULL_DECK, GameConfig, PokerHand, PublicAction
from poker_ai.equity import EquityEstimate
from poker_ai.traditional import (
    LOOSE_CONFIG, POT_ODDS_CONFIG, TIGHT_CONFIG, ThresholdAgent, TraditionalAgent, TraditionalConfig,
)
from poker_ai.traditional_eval import paired_interval, run_development


class FixedEquity:
    def __init__(self, value):
        self.value = value

    def from_observation(self, observation):
        wins = round(self.value * 1000)
        return EquityEstimate(wins, 0, 1000 - wins, "monte_carlo")


def flop(button=False):
    hand = PokerHand(FULL_DECK)
    hand.step(Action.CHECK_CALL)
    hand.step(Action.CHECK_CALL)
    if button:
        hand.step(Action.CHECK_CALL)
    return hand


class TraditionalTests(unittest.TestCase):
    def decision(self, obs, equity, config=TraditionalConfig()):
        return TraditionalAgent(FixedEquity(equity), config).decide(obs)

    def test_call_threshold_boundary_and_reason_trace(self):
        obs = PokerHand(FULL_DECK).observe()
        below = self.decision(obs, 0.279)
        boundary = self.decision(obs, 0.28)
        self.assertEqual(below.slot, Action.FOLD)
        self.assertEqual(boundary.slot, Action.CHECK_CALL)
        self.assertAlmostEqual(boundary.pot_odds, 0.25)
        self.assertAlmostEqual(boundary.required_equity, 0.28)
        self.assertEqual(boundary.chips_added, 1)
        self.assertEqual(boundary.reason, "call_pot_odds")

    def test_free_check_is_never_fold(self):
        obs = flop().observe()
        decision = self.decision(obs, 0)
        self.assertEqual(decision.slot, Action.CHECK_CALL)
        self.assertEqual(decision.reason, "check_free")

    def test_position_and_repeated_raise_margins(self):
        obs = flop().observe()
        history = tuple(PublicAction(i, 1, "raise", 2, 4, Action.MIN_RAISE) for i in (0, 1))
        decision = self.decision(replace(obs, history=history), 0.5)
        self.assertEqual(decision.raises_this_street, 2)
        self.assertAlmostEqual(decision.required_equity, 0.07)
        self.assertAlmostEqual(decision.position_adjustment, 0.02)
        self.assertAlmostEqual(decision.aggression_adjustment, 0.02)
        previous_street = tuple(replace(a, street=0) for a in history)
        self.assertEqual(self.decision(replace(obs, history=previous_street), 0.5).aggression_adjustment, 0)
        river = self.decision(replace(obs, street=3), 0.5)
        self.assertEqual(river.base_margin, 0)
        self.assertEqual(river.position_adjustment, 0.02)

    def test_value_sizes_use_concrete_amounts_when_slots_deduplicated(self):
        obs = flop().observe()
        half = self.decision(obs, 0.75)
        pot = self.decision(obs, 0.9)
        self.assertEqual((half.slot, half.raise_to, half.reason), (Action.MIN_RAISE, 2, "value_half_pot"))
        self.assertEqual((pot.slot, pot.raise_to, pot.reason), (Action.POT, 4, "value_pot"))
        self.assertEqual(pot.chips_added, 4)

    def test_allin_guard_cannot_be_bypassed_by_min_raise_label(self):
        hand = PokerHand(FULL_DECK, GameConfig((5, 200)))
        hand.step(Action.MIN_RAISE)
        obs = hand.observe()
        rejected = self.decision(obs, 0.94)
        allowed = self.decision(obs, 0.96)
        self.assertEqual(rejected.slot, Action.CHECK_CALL)
        self.assertEqual((allowed.slot, allowed.raise_to), (Action.MIN_RAISE, 5))
        self.assertTrue(allowed.commits_effective_stack)

    def test_short_call_pot_odds_exclude_uncallable_excess(self):
        hand = PokerHand(FULL_DECK, GameConfig((80, 200)))
        hand.step(Action.ALL_IN)
        decision = self.decision(hand.observe(), 0.4)
        self.assertAlmostEqual(decision.pot_odds, 78 / 160)
        self.assertEqual(decision.callable_pot, 82)
        self.assertEqual(decision.slot, Action.FOLD)

    def test_bluffs_only_checked_to_postflop_and_reproducible(self):
        config = TraditionalConfig(bluff_probability=1)
        obs = flop(button=True).observe()
        decision = self.decision(obs, 0.2, config)
        self.assertEqual(decision.reason, "bluff_checked_to")
        self.assertIsNotNone(decision.bluff_draw)
        self.assertFalse(decision.commits_effective_stack)
        for other in (flop().observe(), PokerHand(FULL_DECK).observe(), replace(obs, history=())):
            self.assertIsNone(self.decision(other, 0.5, config).bluff_draw)
        agents = [TraditionalAgent(FixedEquity(0.2), TraditionalConfig(bluff_probability=0.5), seed=19) for _ in range(2)]
        sequences = [[agent.decide(obs) for _ in range(20)] for agent in agents]
        self.assertEqual(sequences[0], sequences[1])
        self.assertEqual({d.reason for d in sequences[0]}, {"check_free", "bluff_checked_to"})

    def test_bluff_cannot_commit_short_effective_stack(self):
        hand = PokerHand(FULL_DECK, GameConfig((5, 5)))
        hand.step(Action.MIN_RAISE)
        hand.step(Action.CHECK_CALL)
        hand.step(Action.CHECK_CALL)
        decision = self.decision(hand.observe(), 0.2, TraditionalConfig(bluff_probability=1))
        self.assertEqual(decision.slot, Action.CHECK_CALL)
        self.assertIsNone(decision.bluff_draw)

    def test_pot_odds_ablation_has_no_margin_or_aggression(self):
        obs = PokerHand(FULL_DECK).observe()
        self.assertEqual(self.decision(obs, 0.26, POT_ODDS_CONFIG).slot, Action.CHECK_CALL)
        result = self.decision(flop().observe(), 1, POT_ODDS_CONFIG)
        self.assertEqual(result.slot, Action.CHECK_CALL)
        self.assertEqual(result.required_equity, 0)

    def test_threshold_styles_differ_and_never_fold_free_check(self):
        obs = PokerHand(FULL_DECK).observe()
        tight = ThresholdAgent(FixedEquity(0.4), TIGHT_CONFIG)
        loose = ThresholdAgent(FixedEquity(0.4), LOOSE_CONFIG)
        self.assertEqual(tight.act(obs), Action.FOLD)
        self.assertEqual(loose.act(obs), Action.CHECK_CALL)
        self.assertEqual(tight.act(flop().observe()), Action.CHECK_CALL)

    def test_invalid_config_and_act_preserves_trace(self):
        for params in ({"bluff_probability": -1}, {"pot_threshold": 0.5}, {"base_margin": float("nan")}, {"value_betting": 1}):
            with self.assertRaises(ValueError):
                TraditionalConfig(**params)
        agent = TraditionalAgent(FixedEquity(0.75))
        obs = flop().observe()
        slot = agent.act(obs)
        self.assertEqual(slot, agent.last_decision.slot)
        json.dumps(asdict(agent.last_decision))


class DevelopmentTests(unittest.TestCase):
    def test_paired_bootstrap(self):
        self.assertEqual(paired_interval([3.0, 3.0, 3.0], 1), [3.0, 3.0])
        self.assertEqual(paired_interval([1, 2, 5], 12), paired_interval([1, 2, 5], 12))
        with self.assertRaises(ValueError):
            paired_interval([1], 0)

    def test_development_reproducibility_card_pairing_and_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a", Path(temp) / "b"
            first = run_development(a, pairs=2, seed=23)
            second = run_development(b, pairs=2, seed=23)
            self.assertEqual(first["total_hands"], 48)
            self.assertEqual(first["matchups"], second["matchups"])
            self.assertEqual(first["selected_policy"], second["selected_policy"])
            for file in ("hands.csv", "decisions.jsonl", "selected-policy.json"):
                self.assertEqual((a / file).read_bytes(), (b / file).read_bytes())
            with (a / "hands.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            for deal_id in ("0", "1"):
                group = [r for r in rows if r["deal_id"] == deal_id]
                self.assertEqual(len({r["deck_seed"] for r in group}), 1)
                self.assertEqual({r["hero_seat"] for r in group}, {"0", "1"})
            traces = [json.loads(line) for line in (a / "decisions.jsonl").read_text().splitlines()]
            # Replay one complete logged hand, including all actor choices.
            row = rows[0]
            from poker_ai.environment import shuffled_deck
            hand = PokerHand(shuffled_deck(int(row["deck_seed"])))
            for item in traces:
                if (item["opponent"], item["policy"], str(item["deal_id"]), str(item["hero_seat"])) == (row["opponent"], row["policy"], row["deal_id"], row["hero_seat"]):
                    self.assertEqual(item["actor"], hand.observe().seat)
                    hand.step(item["slot"])
            self.assertEqual(hand.result().net_chips[int(row["hero_seat"])], int(row["net_chips"]))
            for name, opponents in first["matchups"].items():
                for opponent, result in opponents.items():
                    net = sum(int(r["net_chips"]) for r in rows if r["policy"] == name and r["opponent"] == opponent)
                    self.assertEqual(result["bb_per_100"], 100 * net / (4 * 2))
            with self.assertRaises(FileExistsError):
                run_development(a, pairs=2)
