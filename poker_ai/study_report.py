"""Export static plots and a durable report for completed multi-seed training."""

import argparse
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts" / ".matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = ("#2563eb", "#d97706", "#059669")
LABELS = {"random": "Random", "passive": "Passive", "tight": "Tight",
          "loose_aggressive": "Loose/aggressive", "traditional": "Traditional"}


def report(source: Path, output: Path):
    summary = json.loads((source / "study-summary.json").read_text())
    spec = json.loads((source / "study-config.json").read_text())
    seeds = summary["seeds"]
    runs = {seed: json.loads((source / f"seed-{seed}" / "summary.json").read_text()) for seed in seeds}
    curves = {seed: [json.loads(line) for line in (source / f"seed-{seed}" / "learning-curve.jsonl").read_text().splitlines()] for seed in seeds}
    output.mkdir(parents=True, exist_ok=False)
    for name in ("study-summary.json", "study-config.json", "final-matchups.csv"):
        shutil.copyfile(source / name, output / name)
    for seed in seeds:
        for name in ("summary.json", "run-config.json", "learning-curve.jsonl", "training-progress.csv"):
            shutil.copyfile(source / f"seed-{seed}" / name, output / f"seed-{seed}-{name}")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    opponents = list(LABELS)
    baseline = runs[seeds[0]]["validation"]["traditional"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    for ax, opponent in zip(axes.flat, opponents):
        for seed, color in zip(seeds, COLORS):
            ax.plot([row["decisions"] / 1000 for row in curves[seed]],
                    [row["matchups"][opponent]["bb_per_100"] for row in curves[seed]],
                    marker="o", markersize=3, color=color, linewidth=1.6, label=f"Seed {seed}")
        ax.axhline(baseline[opponent]["bb_per_100"], color="#475569", linestyle="--", linewidth=1.2, label="Traditional baseline")
        ax.axhline(0, color="#cbd5e1", linewidth=0.8)
        ax.set_title(f"Against {LABELS[opponent]}", loc="left", weight="bold")
        ax.set_xlabel("Learner decisions (thousands)")
        ax.set_ylabel("Net bb / 100 hands")
        ax.grid(axis="y", alpha=0.15)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="upper left", frameon=False)
    axes.flat[-1].text(0.02, 0.5,
                       f"Fixed final budget: {spec['requested_decisions_per_seed']:,} decisions / seed\n"
                       f"Validation: {spec['validation_pairs']} paired deals / opponent\n"
                       "Same validation deals reused across checkpoints\n"
                       "Lines show individual runs, not a pooled estimate\n"
                       "No best-seed or best-checkpoint selection\n\n"
                       "Development results — final test still untouched",
                       transform=axes.flat[-1].transAxes, va="top", linespacing=1.7, fontsize=10)
    fig.suptitle("Double DQN: development learning curves", fontsize=19, weight="bold")
    fig.savefig(output / "learning-curves.png", dpi=170)
    fig.savefig(output / "learning-curves.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for opponent_index, opponent in enumerate(opponents):
        for index, (seed, color) in enumerate(zip(seeds, COLORS)):
            row = runs[seed]["validation"]["final"][opponent]
            y = opponent_index + (index - 1) * 0.17
            low, high = row["ci95"]
            ax.hlines(y, low, high, color=color, linewidth=2)
            ax.scatter(row["bb_per_100"], y, color=color, s=40,
                       label=f"Seed {seed}" if opponent_index == 0 else None, zorder=3)
        base = baseline[opponent]
        ax.hlines(opponent_index + 0.34, *base["ci95"], color="#334155", linewidth=2)
        ax.scatter(base["bb_per_100"], opponent_index + 0.34, color="#334155", marker="D", s=32,
                   label="Traditional baseline" if opponent_index == 0 else None, zorder=3)
    ax.set_yticks(range(len(opponents)), [LABELS[name] for name in opponents])
    ax.invert_yaxis()
    ax.axvline(0, color="#94a3b8", linewidth=1)
    ax.grid(axis="x", alpha=0.15)
    ax.set_xlabel("Net bb / 100 hands (whole-deal paired-bootstrap 95% intervals)")
    ax.set_title("Fixed final checkpoints: each seed shown separately", fontsize=17, weight="bold", loc="left", pad=42)
    ax.text(0, 1.035, "Development validation only. Hand intervals do not measure training-seed uncertainty.", transform=ax.transAxes, fontsize=10)
    ax.legend(loc="lower right", frameon=True)
    fig.savefig(output / "final-comparison.png", dpi=170)
    fig.savefig(output / "final-comparison.svg")
    plt.close(fig)

    rows = []
    for opponent in opponents:
        values = [runs[seed]["validation"]["final"][opponent]["bb_per_100"] for seed in seeds]
        rows.append("| " + " | ".join([LABELS[opponent], *(f"{v:+.1f}" for v in values),
                                       f"{baseline[opponent]['bb_per_100']:+.1f}"]) + " |")
    run_rows = []
    for seed in seeds:
        run = runs[seed]
        run_rows.append(f"| {seed} | {run['actual_decisions']:,} | {run['gradient_updates']:,} | "
                        f"{run['training_seconds']/60:.1f} | {run['peak_process_rss_bytes']/1e6:.1f} |")
    text = (
        "# Three-seed Double DQN development study\n\n"
        f"Completed {summary['total_actual_decisions']:,} learner decisions across seeds "
        + ", ".join(map(str, seeds)) + ". All runs started from scratch with the frozen configuration.\n\n"
        "Each run retained its fixed final checkpoint; intermediate checkpoints were used only for descriptive "
        "learning curves. No best seed was selected, no hyperparameters were changed mid-run, and no final-test deals were used.\n\n"
        "## Final validation returns (bb/100)\n\n"
        "| Opponent | " + " | ".join(f"Seed {seed}" for seed in seeds) + " | Traditional baseline |\n"
        "|---|" + "---:|" * (len(seeds) + 1) + "\n" + "\n".join(rows) + "\n\n"
        "![Final comparison](final-comparison.png)\n\n"
        f"Each matchup uses {spec['validation_pairs']} paired deals ({2*spec['validation_pairs']} hands). "
        "The same decks were used for every seed, checkpoint and traditional baseline. Intervals resample "
        "whole deal pairs; they describe deal uncertainty for a fixed policy, not variability across training runs. "
        "These are pointwise intervals without multiple-comparison adjustment. "
        "With only three seeds, cross-seed mean/SD/range are descriptive. Returns against simple bots are not a human-poker benchmark.\n\n"
        "## Learning curves\n\n![Learning curves](learning-curves.png)\n\n"
        "The horizontal axis is actual learner decisions, including any small overshoot needed to finish a hand. "
        "Validation is repeated on a small fixed set; the curves are not an independent final test. "
        "Repeated peaks must not be interpreted as best-checkpoint evidence.\n\n"
        "## Runtime\n\n| Seed | Decisions | Gradient updates | Training minutes | Peak process MB |\n"
        "|---|---:|---:|---:|---:|\n" + "\n".join(run_rows) + "\n\n"
        f"Whole-study wall time: {summary['wall_seconds']/60:.1f} minutes. Training times exclude validation. "
        "Three workers used one PyTorch thread each; process memory figures are separate high-water marks, "
        "not measured simultaneous total RAM. Checkpoints retain the same inference-only resume limitation as the pilot.\n\n"
        "## Reproduction and records\n\n"
        "```sh\n.venv/bin/python -m poker_ai.study --output artifacts/three-seed-new\n"
        ".venv/bin/python -m poker_ai.study_report --source artifacts/three-seed-new --output docs/three-seed-new\n```\n\n"
        "The saved study configuration, source hashes, per-seed summaries and learning curves accompany this report. "
        "Full training logs, validation hands and checkpoints remain in the study artifact directory. "
        f"Plots were generated with Matplotlib {matplotlib.__version__}.\n"
    )
    (output / "README.md").write_text(text)
    return output / "README.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(report(args.source, args.output))


if __name__ == "__main__":
    main()
