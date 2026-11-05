from dataclasses import FrozenInstanceError, replace
from itertools import combinations
import json
import random
import unittest

from pokerkit import StandardHighHand
from treys import Card

from poker_ai.environment import Action, FULL_DECK, PokerHand, shuffled_deck
from poker_ai.equity import (
    ALGORITHM, CARD_INTS, DEFAULT_TABLE, EquityCalculator, EquityConfig, EquityEstimate,
    PreflopTable, build_preflop_table, exact_river, monte_carlo,
    preflop_class, preflop_representatives, showdown_share,
)
from poker_ai.randomness import derived_seed


class EquityTests(unittest.TestCase):
    def test_card_conversion_all_52(self):
        self.assertEqual(len(set(CARD_INTS.values())), 52)
        for text, number in CARD_INTS.items():
            self.assertEqual(Card.int_to_str(number), text)

    def test_showdown_known_win_loss_and_tie(self):
        board = ("Qs", "Js", "Ts", "2c", "3d")
        self.assertEqual(showdown_share(("As", "Ks"), ("Ah", "Ad"), board), 1)
        self.assertEqual(showdown_share(("Ah", "Ad"), ("As", "Ks"), board), 0)
        board = ("As", "Ks", "Qs", "Js", "Ts")
        self.assertEqual(showdown_share(("2c", "3d"), ("Ah", "Ad"), board), 0.5)

    def test_1000_independent_evaluator_crosschecks(self):
        rng = random.Random(844)
        for _ in range(1000):
            cards = rng.sample(FULL_DECK, 9)
            hero, opponent, board = cards[:2], cards[2:4], cards[4:]
            h = StandardHighHand.from_game("".join(hero), "".join(board))
            o = StandardHighHand.from_game("".join(opponent), "".join(board))
            expected = 1.0 if h > o else 0.5 if h == o else 0.0
            self.assertEqual(showdown_share(hero, opponent, board), expected)

    def test_exact_nuts_and_board_tie(self):
        result = exact_river(("As", "Ks"), ("Qs", "Js", "Ts", "2c", "3d"))
        self.assertEqual((result.wins, result.ties, result.losses), (990, 0, 0))
        self.assertEqual(result.equity, 1)
        self.assertEqual(result.standard_error, 0)
        result = exact_river(("2c", "3d"), ("As", "Ks", "Qs", "Js", "Ts"))
        self.assertEqual((result.wins, result.ties, result.losses), (0, 990, 0))
        self.assertEqual(result.equity, 0.5)

    def test_full_river_enumeration_against_pokerkit(self):
        hero, board = ("Ah", "Td"), ("Ac", "9s", "8s", "7c", "2d")
        remaining = [c for c in FULL_DECK if c not in hero + board]
        h = StandardHighHand.from_game("".join(hero), "".join(board))
        wins = ties = 0
        for opponent in combinations(remaining, 2):
            o = StandardHighHand.from_game("".join(opponent), "".join(board))
            wins += h > o
            ties += h == o
        result = exact_river(hero, board)
        self.assertEqual((result.wins, result.ties, result.losses), (wins, ties, 990 - wins - ties))

    def test_monte_carlo_ties_and_convergence(self):
        tie = monte_carlo(("2c", "3d"), ("As", "Ks", "Qs", "Js", "Ts"), samples=32, seed=1)
        self.assertEqual((tie.wins, tie.ties, tie.losses), (0, 32, 0))
        hero, board = ("Ah", "Td"), ("Ac", "9s", "8s", "7c", "2d")
        exact = exact_river(hero, board)
        sampled = monte_carlo(hero, board, samples=20_000, seed=112)
        self.assertLess(abs(sampled.equity - exact.equity), 0.02)

    def test_monte_carlo_distinct_deals_and_complete_board(self):
        # Spy on the independent evaluator to verify sampling legality, not
        # just plausible final probabilities.
        from unittest.mock import patch
        from treys import Evaluator
        original = Evaluator.evaluate
        calls = []

        def checking(evaluator, hand, board):
            self.assertEqual(len(board), 5)
            self.assertEqual(len(hand), 2)
            self.assertEqual(len(set(board + hand)), 7)
            calls.append((tuple(board), tuple(hand)))
            return original(evaluator, hand, board)

        for board in ((), ("Qs", "Js", "2d"), ("Qs", "Js", "2d", "3c")):
            calls.clear()
            with patch.object(Evaluator, "evaluate", checking):
                monte_carlo(("As", "Ks"), board, samples=20, seed=2)
            for i in range(0, len(calls), 2):
                self.assertEqual(calls[i][0], calls[i + 1][0])
                self.assertFalse(set(calls[i][1]) & set(calls[i + 1][1]))

    def test_169_classes_cover_all_1326_hands(self):
        counts = {}
        for hand in combinations(FULL_DECK, 2):
            key = preflop_class(hand)
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(len(counts), 169)
        self.assertEqual(set(counts), set(preflop_representatives()))
        for key, count in counts.items():
            self.assertEqual(count, 6 if len(key) == 2 else 4 if key[-1] == "s" else 12)
        for key, hand in preflop_representatives().items():
            self.assertEqual(preflop_class(hand), key)
        self.assertEqual(preflop_class(("Kd", "Ad")), "AKs")
        self.assertEqual(preflop_class(("Ac", "Kh")), "AKo")

    def test_preflop_table_reproducible_and_validated(self):
        payload = build_preflop_table(samples=4, seed=9)
        self.assertEqual(payload, build_preflop_table(samples=4, seed=9))
        table = PreflopTable(payload)
        self.assertEqual(table.lookup(("Ah", "Kh")), table.lookup(("Ks", "As")))
        with self.assertRaises(ValueError):
            PreflopTable({**payload, "opponent_range": "custom"})
        with self.assertRaises(ValueError):
            PreflopTable({**payload, "entries": {}})
        with self.assertRaises(ValueError):
            PreflopTable([])
        payload["entries"]["AA"]["wins"] = -1
        with self.assertRaises(ValueError):
            PreflopTable(payload)

    def test_bundled_table_counts_and_sanity(self):
        payload = json.loads(DEFAULT_TABLE.read_text())
        self.assertEqual(payload["samples_per_class"], 10_000)
        table = PreflopTable(payload)
        aa = table.lookup(("As", "Ah"))
        seven_two = table.lookup(("7c", "2d"))
        self.assertEqual(aa.outcomes, 10_000)
        self.assertTrue(0.80 < aa.equity < 0.90)
        self.assertTrue(0.28 < seven_two.equity < 0.40)
        self.assertGreater(aa.equity, seven_two.equity)
        regenerated = monte_carlo(("As", "Ah"), (), samples=10_000,
                                  seed=derived_seed(payload["seed"], ALGORITHM, "preflop", "AA"))
        self.assertEqual((aa.wins, aa.ties, aa.losses),
                         (regenerated.wins, regenerated.ties, regenerated.losses))

    def test_routing_cache_bound_and_order_independence(self):
        calculator = EquityCalculator(EquityConfig(samples=64, seed=11, cache_size=1))
        hero, flop = ("As", "Ks"), ("Qs", "Js", "2d")
        self.assertEqual(calculator.evaluate(hero).method, "preflop_lookup")
        first = calculator.evaluate(hero, flop)
        self.assertEqual(first.method, "monte_carlo")
        self.assertEqual(first.outcomes, 64)
        self.assertIs(first, calculator.evaluate(hero[::-1], flop[::-1]))
        river = calculator.evaluate(hero, flop + ("3c", "4h"))
        self.assertEqual((river.method, river.outcomes), ("exact_river", 990))
        self.assertEqual(calculator.cache_entries, 1)
        # Eviction and recomputation must not change the sampled feature.
        self.assertEqual(first, calculator.evaluate(hero, flop))
        no_cache = EquityCalculator(EquityConfig(samples=64, seed=11, cache_size=0))
        self.assertEqual(first, no_cache.evaluate(hero, flop))
        self.assertEqual(no_cache.cache_entries, 0)

    def test_visible_information_only_and_no_rng_side_effects(self):
        calculator = EquityCalculator()
        hand = PokerHand(shuffled_deck(20))
        hand.step(Action.CHECK_CALL)
        hand.step(Action.CHECK_CALL)
        obs = hand.observe()
        random.seed(123)
        before = random.getstate()
        first = calculator.from_observation(obs)
        # Equity uses cards only; public betting context will be used by agents.
        changed = replace(obs, pot=999, stacks=(1, 2), history=())
        self.assertEqual(first, calculator.from_observation(changed))
        self.assertEqual(before, random.getstate())
        self.assertEqual(hand.observe(), obs)

    def test_invalid_cards_counts_and_streets(self):
        for hole, board in ((("As", "As"), ()), (("As",), ()),
                            (("As", "Kh"), ("As", "2d", "3c")),
                            (("As", "Kh"), ("2d",)), (("ZZ", "Kh"), ())):
            with self.assertRaises(ValueError):
                monte_carlo(hole, board)
        for samples in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                monte_carlo(("As", "Ks"), (), samples=samples)
        with self.assertRaises(ValueError):
            exact_river(("As", "Ks"), ("Qs", "Js", "2d"))
        with self.assertRaises(ValueError):
            showdown_share(("As", "Ks"), ("As", "Kh"), ("Qs", "Js", "Ts", "2d", "3c"))
        with self.assertRaises(ValueError):
            EquityConfig(cache_size=-1)

    def test_estimate_is_immutable_and_se_uses_ties(self):
        result = EquityEstimate(1, 2, 1, "monte_carlo")
        self.assertEqual(result.equity, 0.5)
        self.assertAlmostEqual(result.standard_error, (1 / 24)**0.5)
        self.assertIsNone(EquityEstimate(1, 0, 0, "monte_carlo").standard_error)
        with self.assertRaises(FrozenInstanceError):
            result.wins = 0
