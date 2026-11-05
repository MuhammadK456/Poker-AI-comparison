import csv
import json
from pathlib import Path
import tempfile
import unittest

from poker_ai.ablation_report import matched_groups


class AblationReportTests(unittest.TestCase):
    def test_comparison_uses_matched_deals_and_rejects_changed_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            reference, ablation = Path(temp) / "reference", Path(temp) / "ablation"
            for root, delta in ((reference, 0), (ablation, 2)):
                root.mkdir()
                config = {"pairs_per_matchup": 2, "validation_seed": 99, "deal_namespace": "test",
                          "traditional_snapshot": {}, "equity": {}, "threshold_opponents": {}}
                (root / "run-config.json").write_text(json.dumps(config))
                for label in ("traditional", "seed-101", "seed-202", "seed-303"):
                    shift = delta if label != "traditional" else 0
                    with (root / f"validation-{label}.csv").open("w", newline="") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(["opponent", "deal_id", "hero_seat", "deck_seed", "net_chips"])
                        for deal, chips in ((0, -100), (1, 100)):
                            for seat in (0, 1):
                                writer.writerow(["passive", deal, seat, deal+42, chips+shift])
                    groups = {"passive": [-5000+50*shift, 5000+50*shift]}
                    (root / f"groups-{label}.json").write_text(json.dumps({"groups": groups}))
            result = matched_groups(reference, ablation)["seed-101"]["passive"]
            self.assertEqual(result["difference_no_equity_minus_with_equity"], 100)
            self.assertEqual(result["ci95"], [100, 100])
            config["validation_seed"] = 100
            (ablation / "run-config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "matching validation_seed"):
                matched_groups(reference, ablation)
