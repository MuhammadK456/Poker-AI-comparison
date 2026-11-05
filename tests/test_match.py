import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from poker_ai.environment import PokerHand, shuffled_deck
from poker_ai.match import derived_seed, run_match


class MatchTests(unittest.TestCase):
    def test_paired_seats_preserve_actual_decks_and_cards(self):
        seen = []

        def capture(deck, config):
            hand = PokerHand(deck, config)
            seen.append((deck, tuple(tuple(map(repr, cards)) for cards in hand._state.hole_cards)))
            return hand

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            with patch("poker_ai.match.PokerHand", side_effect=capture):
                summary = run_match(10, 101, path)
            with (path / "hands.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(summary["hands"], 20)
            for i in range(0, 20, 2):
                self.assertEqual(seen[i], seen[i + 1])
                self.assertEqual(rows[i]["deal_id"], rows[i + 1]["deal_id"])
                self.assertEqual((rows[i]["random_seat"], rows[i + 1]["random_seat"]), ("0", "1"))
                self.assertEqual(seen[i][0], shuffled_deck(int(rows[i]["deck_seed"])))
            self.assertEqual(sum(int(r["random_net_chips"]) for r in rows), summary["random_net_chips"])

    def test_repeated_run_identical_logs_and_replayable_actions(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a", Path(temp) / "b"
            self.assertEqual(run_match(12, 42, a), run_match(12, 42, b))
            for file in a.iterdir():
                self.assertEqual(file.read_bytes(), (b / file.name).read_bytes())
            with (a / "hands.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            traces = [json.loads(line) for line in (a / "actions.jsonl").read_text().splitlines()]
            for row, trace in zip(rows, traces):
                hand = PokerHand(shuffled_deck(int(row["deck_seed"])))
                for action in trace["actions"]:
                    self.assertEqual(hand.observe().seat, action["seat"])
                    hand.step(action["slot"])
                self.assertEqual(hand.result().net_chips[int(row["random_seat"])], int(row["random_net_chips"]))
            with self.assertRaises(FileExistsError):
                run_match(1, 42, a)

    def test_seed_namespaces_and_bad_pair_count(self):
        self.assertNotEqual(derived_seed(1, "policy", 0), derived_seed(1, "smoke_deal", 0))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad"
            with self.assertRaises(ValueError):
                run_match(0, 42, path)
            self.assertFalse(path.exists())
