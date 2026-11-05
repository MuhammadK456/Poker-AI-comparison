"""Export the completed development variance pilot and precision estimates."""

import argparse
import csv
import json
import math
from pathlib import Path
import shutil


def audit_logs(source, labels, opponents, pairs):
    """Verify complete seat pairs, shared decks, and independently rebuilt returns."""
    reference = None
    for label in labels:
        records = {}
        with (source / f"validation-{label}.csv").open() as handle:
            for row in csv.DictReader(handle):
                key = row["opponent"], int(row["deal_id"]), int(row["hero_seat"])
                if key in records:
                    raise ValueError("duplicate evaluation hand")
                records[key] = (int(row["deck_seed"]), int(row["net_chips"]))
        expected = {(opponent, deal, seat) for opponent in opponents
                    for deal in range(pairs) for seat in (0, 1)}
        if set(records) != expected:
            raise ValueError("missing or unexpected evaluation hands")
        decks = {key: value[0] for key, value in records.items()}
        if reference is not None and decks != reference:
            raise ValueError("policies did not share evaluation decks")
        reference = decks
        saved = json.loads((source / f"groups-{label}.json").read_text())["groups"]
        for opponent in opponents:
            actual = []
            for deal in range(pairs):
                one, two = records[opponent, deal, 0], records[opponent, deal, 1]
                if one[0] != two[0]:
                    raise ValueError("seat swaps did not share a deck")
                actual.append((one[1] + two[1]) * 25.0)
            if saved[opponent] != actual:
                raise ValueError("saved returns differ from hand logs")


def report(source, output):
    summary = json.loads((source / "summary.json").read_text())
    labels = ["seed-101", "seed-202", "seed-303", "traditional"]
    opponents = list(summary["results"]["traditional"]["matchups"])
    audit_logs(source, labels, opponents, summary["pairs_per_matchup"])
    output.mkdir(parents=True, exist_ok=False)
    for name in ("summary.json", "run-config.json"):
        shutil.copyfile(source / name, output / name)
    lines = ["# Larger development evaluation", "",
             f"Frozen models; {summary['pairs_per_matchup']:,} paired deals per matchup; "
             f"{summary['played_hands']:,} played hands. Evaluation wall time: {summary['wall_seconds']/60:.1f} minutes.", "",
             "Fresh development deals shared across all policies. No retraining, checkpoint selection or final test.", "",
             "All hand logs were checked for complete seat pairs, identical decks across policies, and exact agreement with saved paired returns.", "",
             "## Return (bb/100; paired-bootstrap 95% interval)", "",
             "| Opponent | Seed 101 | Seed 202 | Seed 303 | Traditional |",
             "|---|---:|---:|---:|---:|"]
    for opponent in opponents:
        cells = []
        for label in labels:
            row = summary["results"][label]["matchups"][opponent]
            cells.append(f"{row['bb_per_100']:+.1f} [{row['ci95'][0]:+.1f}, {row['ci95'][1]:+.1f}]")
        lines.append("| " + " | ".join([opponent, *cells]) + " |")
    lines += ["", "## Paired improvement over the traditional baseline", "",
              "Both policies face the same opponent on the same paired deals. These differences answer a different question from direct head-to-head returns.", "",
              "| Opponent | Seed 101 | Seed 202 | Seed 303 |", "|---|---:|---:|---:|"]
    for opponent in opponents:
        cells = []
        for label in labels[:3]:
            row = summary["results"][label]["matchups"][opponent]
            cells.append(f"{row['difference_vs_traditional']:+.1f} [{row['difference_ci95'][0]:+.1f}, {row['difference_ci95'][1]:+.1f}]")
        lines.append("| " + " | ".join([opponent, *cells]) + " |")
    lines += ["", "## Precision planning", "",
              "Normal approximation: required pairs = ceil((1.96 × paired-outcome SD / target half-width)²). "
              "This estimates interval width for a fixed policy; it is not statistical power, a guarantee, or a training-seed interval. "
              "It assumes the future matchup has similar variance. New opponent families need their own development variance estimates.", "",
              "| Seed | Direct vs traditional: half-width at 10k pairs | Pairs for ±100 bb/100 | Worst paired-difference requirement across current opponents |",
              "|---|---:|---:|---:|"]
    requirements = []
    for label in labels[:3]:
        matches = summary["results"][label]["matchups"]
        direct = matches["traditional"]["return_precision"]
        worst = max(row["difference_precision"]["pairs_for_half_width"]["100"] for row in matches.values())
        requirements.extend([worst, direct["pairs_for_half_width"]["100"]])
        lines.append(f"| {label} | {direct['approx_half_width_at_10000_pairs']:.1f} | {direct['pairs_for_half_width']['100']:,} | {worst:,} |")
    suggested = max(10000, math.ceil(max(requirements) / 1000) * 1000)
    lines += ["", f"For the existing matchup panel, retain at least **{suggested:,} pairs per matchup** "
              "for the ±100 planning target, rounded up to 1,000 and respecting the original 10,000-pair starting budget. "
              "This is a recommendation only; no final evaluation has been launched.", "",
              f"Linear same-panel runtime projection at {suggested:,} pairs: "
              f"{summary['wall_seconds'] * suggested / summary['pairs_per_matchup'] / 60:.1f} minutes, "
              "excluding additional opponents or ablations and allowing for hardware/load variation.", "",
              "The JSON includes alternative ±50/±200 targets. All intervals are pointwise, with no multiple-comparison correction. "
              "Whole deal pairs are resampled; training seeds are reported separately. Repeated development inspection prevents treating these results as final-test claims.", "",
              "Absolute-return and paired-difference intervals use separate bootstrap random streams. Against the deterministic traditional self-play reference, "
              "the underlying samples coincide, but finite bootstrap resampling can put endpoints on opposite sides of zero. "
              "Treat a boundary result as fragile; do not choose whichever interval gives a preferred conclusion.", "",
              "## Seat breakdown", "",
              "Descriptive returns by physical seat are in `seat-results.csv`; seat 0 is BB and seat 1 is SB/button. "
              "Their correlated hands must not replace whole-deal groups in uncertainty calculations.", "",
              "## Reproduce", "", "```sh",
              ".venv/bin/python -m poker_ai.evaluate --output artifacts/development-variance-new",
              ".venv/bin/python -m poker_ai.evaluation_report --source artifacts/development-variance-new --output docs/development-variance-new",
              "```", ""]
    with (output / "seat-results.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["policy", "opponent", "hero_seat", "hands", "bb_per_100"])
        for label in labels:
            totals = {}
            with (source / f"validation-{label}.csv").open() as handle:
                for row in csv.DictReader(handle):
                    key = row["opponent"], row["hero_seat"]
                    count, chips = totals.get(key, (0, 0))
                    totals[key] = count + 1, chips + int(row["net_chips"])
            for (opponent, seat), (count, chips) in totals.items():
                writer.writerow([label, opponent, seat, count, 50 * chips / count])
    (output / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report(args.source, args.output)
