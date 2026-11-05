import unittest
import csv
import json
from pathlib import Path
import tempfile

from poker_ai.evaluate import precision_plan
from poker_ai.evaluation_report import audit_logs


class PrecisionTests(unittest.TestCase):
    def test_required_pairs_scale_with_variance_and_inverse_squared_width(self):
        result = precision_plan([-100, 100], (100, 200))
        self.assertAlmostEqual(result["pair_sd_bb_per_100"], 20000 ** 0.5)
        self.assertEqual(result["pairs_for_half_width"], {"100": 8, "200": 2})
        self.assertAlmostEqual(result["approx_half_width_at_10000_pairs"], 2.7718585822512662)

    def test_degenerate_and_insufficient_samples(self):
        self.assertEqual(precision_plan([0, 0])["approx_half_width_at_10000_pairs"], 0)
        with self.assertRaises(ValueError):
            precision_plan([1])

    def test_audit_rejects_broken_pairing_and_duplicate_hands(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def write(label, deck=42, duplicate=False):
                with (root / f"validation-{label}.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["opponent", "deal_id", "hero_seat", "deck_seed", "net_chips"])
                    writer.writerows([["passive", 0, 0, deck, 10], ["passive", 0, 1, deck, -2]])
                    if duplicate:
                        writer.writerow(["passive", 0, 0, deck, 10])
                (root / f"groups-{label}.json").write_text(json.dumps({"groups": {"passive": [200.0]}}))
            write("a")
            write("b")
            audit_logs(root, ["a", "b"], ["passive"], 1)
            write("b", deck=43)
            with self.assertRaisesRegex(ValueError, "share evaluation decks"):
                audit_logs(root, ["a", "b"], ["passive"], 1)
            write("b", duplicate=True)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                audit_logs(root, ["a", "b"], ["passive"], 1)
