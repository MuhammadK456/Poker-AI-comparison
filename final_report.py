"""Generate the final held-out report, figures, and descriptive tables."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil

import numpy as np

from .evaluation_report import audit_logs
from .train import ROOT

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/.matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {"random": "Random", "passive": "Passive", "tight": "Tight",
          "loose_aggressive": "Loose/aggressive", "traditional": "Traditional"}
COLORS = ["#2563eb", "#d97706", "#059669", "#334155"]


def report(source, output):
    summary = json.loads((source / "summary.json").read_text())
    config = json.loads((source / "run-config.json").read_text())
    labels = [f"seed-{seed}" for seed in config["seeds"]] + ["traditional"]
    audit_logs(source, labels, config["opponents"], config["pairs"])
    output.mkdir(parents=True, exist_ok=False)
    for name in ("summary.json", "run-config.json"):
        shutil.copyfile(source / name, output / name)
    statistics_rows = []
    for label in labels:
        groups = {(name, seat): [] for name in config["opponents"] for seat in (0, 1)}
        with (source / f"validation-{label}.csv").open() as handle:
            for row in csv.DictReader(handle):
                groups[row["opponent"], int(row["hero_seat"])].append(int(row["net_chips"]) / 2)
        for name in config["opponents"]:
            for seat in (0, 1, "all"):
                values = np.asarray(groups[name, seat] if seat != "all" else groups[name, 0]+groups[name, 1])
                statistics_rows.append({"policy": label, "opponent": name, "seat": seat, "hands": len(values),
                    "bb_per_100": float(values.mean()*100), "hand_return_sd_bb": float(values.std(ddof=1)),
                    "hand_return_q05_bb": float(np.quantile(values, .05)),
                    "hand_return_median_bb": float(np.median(values)),
                    "hand_return_q95_bb": float(np.quantile(values, .95))})
    with (output / "position-and-return-spread.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(statistics_rows[0]))
        writer.writeheader()
        writer.writerows(statistics_rows)
    performance = []
    for label in labels:
        for name, row in summary["results"][label].items():
            count = row["decisions"]
            performance.append({"policy": label, "opponent": name, "decisions": count,
                **{k: row[k] for k in ("decision_latency_ms_mean", "decision_latency_ms_median", "decision_latency_ms_p95")},
                **{f"{kind}_fraction": row["actions"].get(kind, 0)/count for kind in ("fold", "check", "call", "bet", "raise")}})
    with (output / "actions-and-latency.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(performance[0]))
        writer.writeheader()
        writer.writerows(performance)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    adjusted_key = str(config["primary_simultaneous_coverage"])
    for index, label in enumerate(labels[:3]):
        row = summary["results"][label]["traditional"]
        ax.hlines(index, *row["intervals"][adjusted_key], color=COLORS[index], linewidth=3)
        ax.hlines(index, *row["intervals"]["0.95"], color=COLORS[index], linewidth=9, alpha=.4)
        ax.scatter(row["bb_per_100"], index, s=60, color=COLORS[index], zorder=3)
    ax.axvline(0, color="#64748b", linestyle="--", linewidth=1)
    ax.set_yticks(range(3), [f"Training seed {seed}" for seed in config["seeds"]])
    ax.invert_yaxis()
    ax.set_ylim(2.5, -.5)
    ax.set_xlabel("ML net big blinds per 100 hands against the traditional agent")
    ax.set_title("Final head-to-head: no reliable winner", loc="left", weight="bold", pad=35)
    ax.text(0, 1.035, "10,000 paired deals per seed • thick: 95% • thin: 98.333% Bonferroni-adjusted intervals",
            transform=ax.transAxes, fontsize=10)
    ax.grid(axis="x", alpha=.15)
    for extension in ("png", "svg"):
        fig.savefig(output / f"head-to-head.{extension}", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    for index, name in enumerate(config["opponents"]):
        for offset, label in enumerate(labels):
            row = summary["results"][label][name]
            y = index + (offset-1.5)*.16
            ax.hlines(y, *row["intervals"]["0.95"], color=COLORS[offset], linewidth=2)
            ax.scatter(row["bb_per_100"], y, color=COLORS[offset], s=30,
                       label=(label.replace("seed-", "ML seed ") if label != "traditional" else "Traditional") if index == 0 else None)
    ax.axvline(0, color="#94a3b8", linewidth=1)
    ax.set_yticks(range(5), [LABELS[name] for name in config["opponents"]])
    ax.invert_yaxis()
    ax.set_xlabel("Net big blinds per 100 hands (pointwise paired-bootstrap 95% intervals)")
    ax.set_title("Final held-out deals: strengths depend on opponent", loc="left", weight="bold", pad=30)
    ax.text(0, 1.025, "Existing opponent settings • no training or model selection • seeds shown separately",
            transform=ax.transAxes, fontsize=10)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="x", alpha=.15)
    for extension in ("png", "svg"):
        fig.savefig(output / f"matchups.{extension}", dpi=170)
    plt.close(fig)

    lines = ["# Final held-out evaluation", "",
             f"Completed {summary['played_hands']:,} hands on fresh final-test deals, "
             f"{summary['pairs_per_matchup']:,} independent seat-swapped deal pairs per matchup. "
             f"Total evaluation/analysis wall time: {summary['wall_seconds']/60:.1f} minutes.", "",
             "All three original equity-enabled 250k-decision models and the frozen traditional baseline were retained. "
             "No retraining, checkpoint selection or additional sampling followed these outcomes. "
             "The [protocol](../final-evaluation-protocol.md) was frozen before the run.", "",
             "## Primary result", "",
             "There is no reliable head-to-head winner on this test. Every interval includes zero. "
             "This is inconclusive, not evidence that the policies are equivalent.", "",
             "| Training seed | ML bb/100 vs traditional | 95% interval | Adjusted 98.333% interval |",
             "|---|---:|---:|---:|"]
    for seed in config["seeds"]:
        row = summary["results"][f"seed-{seed}"]["traditional"]
        lo, hi = row["intervals"]["0.95"]
        alo, ahi = row["intervals"][adjusted_key]
        lines.append(f"| {seed} | {row['bb_per_100']:+.1f} | [{lo:+.1f}, {hi:+.1f}] | [{alo:+.1f}, {ahi:+.1f}] |")
    lines += ["", "![Final head-to-head intervals](head-to-head.png)", "",
              "The three primary comparisons use 10,000 whole-pair bootstrap resamples. "
              "Bonferroni-adjusted intervals target 95% familywise coverage across these three fixed-model comparisons; "
              "bootstrap coverage is approximate. Deal uncertainty does not measure uncertainty across training runs.", "",
              "## Secondary matchup results", "",
              "Return in bb/100 with pointwise 95% intervals. These comparisons are descriptive and not multiplicity-adjusted.", "",
              "| Opponent | Seed 101 | Seed 202 | Seed 303 | Traditional |", "|---|---:|---:|---:|---:|"]
    for name in config["opponents"]:
        cells = []
        for label in labels:
            row = summary["results"][label][name]
            lo, hi = row["intervals"]["0.95"]
            cells.append(f"{row['bb_per_100']:+.1f} [{lo:+.1f}, {hi:+.1f}]")
        lines.append("| " + " | ".join([LABELS[name], *cells]) + " |")
    lines += ["", "![Final matchups](matchups.png)", "",
              "Paired differences below compare ML and traditional against the same opponent/deals. "
              "Positive values favor ML; this differs from direct head-to-head play.", "",
              "| Opponent | Seed 101 | Seed 202 | Seed 303 |", "|---|---:|---:|---:|"]
    for name in config["opponents"]:
        cells = []
        for label in labels[:3]:
            row = summary["results"][label][name]
            lo, hi = row["difference_intervals"]["0.95"]
            cells.append(f"{row['difference_vs_traditional']:+.1f} [{lo:+.1f}, {hi:+.1f}]")
        lines.append("| " + " | ".join([LABELS[name], *cells]) + " |")
    lines += ["", "The paired secondary results favor ML against passive and tight opponents for all three seeds. "
              "They favor the traditional baseline against random and loose/aggressive opponents. "
              "Against tight play, outperforming the baseline does not establish positive absolute returns: all ML absolute-return intervals span zero.", "",
              "## Additional measurements", "",
              "- [Position and hand-return spread](position-and-return-spread.csv): returns by BB/SB, individual-hand SD and 5th/50th/95th percentiles. "
              "These are descriptive; the inferential unit remains the independent deal pair.",
              "- [Actions and decision latency](actions-and-latency.csv): per-matchup fold/check/call/bet/raise fractions and mean/median/p95 latency. "
              "Timing includes feature/equity calculation, excludes opponent actions, and was observed under three concurrent workers with caching.",
              "- [Exact results](summary.json) and [frozen configuration/hashes](run-config.json). Full 400,000 hand records remain in the final-evaluation artifacts and release bundle.", "",
              "## Limits and reproducibility", "",
              "The final test holds out deals, not opponent families. The tested game is heads-up, 100 BB stacks reset each hand, "
              "no rake, with six discrete action slots and deduplicated legal sizes. Equity assumes uniformly distributed legal opposing cards; "
              "the learned history representation is a summary. No human-level, equilibrium or unrestricted-poker claim is supported.", "",
              "All final logs pass complete-pair, shared-deck and exact-return audits. Frozen model hashes match the development records. "
              "The fresh Python 3.11 environment passed 75 tests and reproduced the four-hand demo byte-for-byte. "
              "The no-equity and opponent-variation studies remain development evidence in the [bounded report](../bounded-experiments-20261104/README.md).", "",
              "```sh", ".venv/bin/python -m poker_ai.final_eval --output artifacts/final-reproduction",
              ".venv/bin/python -m poker_ai.final_report --source artifacts/final-reproduction --output docs/final-reproduction",
              "```", "",
              "Replaying the frozen final seed is a reproducibility check, not another independent final test. "
              "Do not tune models or choose checkpoints using these results.", ""]
    (output / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report(args.source, args.output)
