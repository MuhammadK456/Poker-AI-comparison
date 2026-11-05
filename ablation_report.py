"""Summarize the single bounded equity ablation and configuration robustness check."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics

from .evaluation_report import audit_logs
from .randomness import derived_seed
from .traditional_eval import paired_interval


def matched_groups(reference, ablation):
    configs = [json.loads((path / "run-config.json").read_text()) for path in (reference, ablation)]
    for key in ("pairs_per_matchup", "validation_seed", "deal_namespace", "traditional_snapshot", "equity", "threshold_opponents"):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"ablation comparisons require matching {key}")
    opponents = list(json.loads((reference / "groups-traditional.json").read_text())["groups"])
    labels = ["traditional", "seed-101", "seed-202", "seed-303"]
    for source in (reference, ablation):
        audit_logs(source, labels, opponents, configs[0]["pairs_per_matchup"])
    # Check actual deck mappings across experiments, not just matching configuration.
    def decks(path):
        with path.open() as stream:
            return {(r["opponent"], r["deal_id"], r["hero_seat"]): r["deck_seed"] for r in csv.DictReader(stream)}
    if decks(reference / "validation-traditional.csv") != decks(ablation / "validation-traditional.csv"):
        raise ValueError("ablation evaluation decks differ")
    ref = {label: json.loads((reference / f"groups-{label}.json").read_text())["groups"] for label in labels}
    abl = {label: json.loads((ablation / f"groups-{label}.json").read_text())["groups"] for label in labels}
    if ref["traditional"] != abl["traditional"]:
        raise ValueError("traditional baseline changed between experiments")
    results = {}
    for label in labels[1:]:
        results[label] = {}
        for opponent in opponents:
            differences = [a-b for a, b in zip(abl[label][opponent], ref[label][opponent], strict=True)]
            results[label][opponent] = {
                "no_equity_bb_per_100": statistics.mean(abl[label][opponent]),
                "with_equity_bb_per_100": statistics.mean(ref[label][opponent]),
                "difference_no_equity_minus_with_equity": statistics.mean(differences),
                "ci95": paired_interval(differences, derived_seed(configs[0]["validation_seed"], "equity_ablation_ci", label, opponent)),
            }
    return results


def report(reference, ablation, robustness, training, output):
    differences = matched_groups(reference, ablation)
    robust = json.loads((robustness / "summary.json").read_text())
    study = json.loads((training / "study-summary.json").read_text())
    robust_config = json.loads((robustness / "run-config.json").read_text())
    audit_logs(robustness, list(robust["results"]), list(robust_config["opponents"]), robust["pairs_per_matchup"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "ablation-differences.json").write_text(json.dumps(differences, indent=2) + "\n")
    for source, prefix, names in ((reference, "reference", ["summary.json", "run-config.json"]),
                                  (ablation, "ablation", ["summary.json", "run-config.json"]),
                                  (robustness, "robustness", ["summary.json", "run-config.json"]),
                                  (training, "training", ["study-summary.json", "study-config.json"])):
        for name in names:
            shutil.copyfile(source / name, output / f"{prefix}-{name}")
    for seed in (101, 202, 303):
        for name in ("summary.json", "run-config.json", "learning-curve.jsonl", "training-progress.csv"):
            shutil.copyfile(training / f"seed-{seed}" / name, output / f"seed-{seed}-{name}")
    (output / "report-source-sha256.txt").write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() + "\n")
    lines = ["# Bounded equity ablation and opponent robustness", "",
             "Exactly one feature ablation, three seeds, and three opponent-parameter variations. "
             "All checkpoints are fixed final 250k-decision models. No retuning or final testing.", "",
             "## No-equity ablation", "",
             "The supplied equity input is held at zero during training and inference. Raw cards, pot odds, "
             "history, architecture and training budget are retained. Opponents still use their original equity calculations.", "",
             "The evaluation uses the same 2,000 paired development deals per opponent as the reference models. "
             "Values below are no-equity minus with-equity returns, in bb/100, with whole-pair bootstrap 95% intervals. "
             "Negative differences favor retaining the explicit equity feature.", "",
             "| Opponent | Seed 101 | Seed 202 | Seed 303 |", "|---|---:|---:|---:|"]
    for opponent in differences["seed-101"]:
        cells = []
        for seed in (101, 202, 303):
            row = differences[f"seed-{seed}"][opponent]
            cells.append(f"{row['difference_no_equity_minus_with_equity']:+.1f} [{row['ci95'][0]:+.1f}, {row['ci95'][1]:+.1f}]")
        lines.append("| " + " | ".join([opponent, *cells]) + " |")
    lines += ["", "Absolute returns and paired differences are saved in `ablation-differences.json`. "
              "Matching training seeds do not force identical training trajectories. These intervals reflect deal uncertainty "
              "for fixed model pairs, not uncertainty over training runs or universal feature importance.", "",
              "## Opponent-configuration robustness", "",
              "The original models and traditional baseline face two shifted threshold settings and a 15% eligible "
              "checked-to bluff variant of the existing traditional rules. These configurations were excluded from training; "
              "they are not new algorithm families or human opponents. The ablation models are not added to this panel.", "",
              "Return in bb/100 (95% whole-pair bootstrap interval), 2,000 paired deals per matchup:", "",
              "| Opponent | Seed 101 | Seed 202 | Seed 303 | Traditional |", "|---|---:|---:|---:|---:|"]
    for opponent in robust_config["opponents"]:
        cells = []
        for label in ("seed-101", "seed-202", "seed-303", "traditional"):
            row = robust["results"][label]["matchups"][opponent]
            cells.append(f"{row['bb_per_100']:+.1f} [{row['ci95'][0]:+.1f}, {row['ci95'][1]:+.1f}]")
        lines.append("| " + " | ".join([opponent, *cells]) + " |")
    lines += ["", "Paired improvements over the traditional baseline:", "",
              "| Opponent | Seed 101 | Seed 202 | Seed 303 |", "|---|---:|---:|---:|"]
    for opponent in robust_config["opponents"]:
        cells = []
        for seed in (101, 202, 303):
            row = robust["results"][f"seed-{seed}"]["matchups"][opponent]
            cells.append(f"{row['difference_vs_traditional']:+.1f} [{row['difference_ci95'][0]:+.1f}, {row['difference_ci95'][1]:+.1f}]")
        lines.append("| " + " | ".join([opponent, *cells]) + " |")
    lines += ["", "## Cost, checks and limits", "",
              f"Ablation training: {study['total_actual_decisions']:,} decisions, {study['wall_seconds']/60:.1f} minutes wall time. "
              f"Robustness: {robust['played_hands']:,} hands, {robust['wall_seconds']/60:.1f} minutes. "
              "Ablation evaluation adds 80,000 hands including a repeated traditional reference used to check unchanged behavior.", "",
              "Hand logs were audited for complete seat pairs, shared decks and exact return reconstruction. "
              "The original and ablation panels share actual decks and reproduce identical traditional-baseline returns. "
              "All intervals are pointwise, exploratory and unadjusted for multiple comparisons. Three training seeds provide limited replication. "
              "No best seed/checkpoint was selected. Final-test data remain untouched.", "",
              "## Scope boundary", "",
              "Development expansion stops here. Remaining required work is final evaluation and presentation/reproduction. "
              "The position ablation, new opponent family, opponent modeling, betting rollouts, self-play and game variants "
              "are outside required version-one scope. See `../bounded-completion-protocol.md`.", "",
              "## Reproduction", "", "```sh",
              ".venv/bin/python -m poker_ai.study --config configs/dqn-no-equity.json --output artifacts/no-equity-new",
              ".venv/bin/python -m poker_ai.evaluate --source artifacts/no-equity-new --output artifacts/no-equity-eval-new",
              ".venv/bin/python -m poker_ai.robustness --output artifacts/robustness-new",
              ".venv/bin/python -m poker_ai.ablation_report --reference artifacts/development-variance-20261104 --ablation artifacts/no-equity-eval-new --robustness artifacts/robustness-new --training artifacts/no-equity-new --output docs/bounded-experiments-new",
              "```", "",
              "Use fresh output directories. Saved no-equity checkpoints load with `DoubleDQN.load_for_inference`; "
              "construct their encoder with `FeatureEncoder(calculator, include_equity=model.include_equity)`. "
              "The agent rejects mismatched encoder settings. Existing checkpoints default to equity enabled. "
              "Checkpoints remain inference snapshots, not exact training-resume files.", ""]
    (output / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "ablation", "robustness", "training", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    report(args.reference, args.ablation, args.robustness, args.training, args.output)
