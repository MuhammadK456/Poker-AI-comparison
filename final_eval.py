"""Fixed-budget held-out-deal evaluation of the original three models."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from .dqn import DQNAgent, DoubleDQN
from .environment import PokerHand, shuffled_deck
from .equity import DEFAULT_TABLE, EquityCalculator, EquityConfig
from .evaluation_report import audit_logs
from .features import FeatureEncoder
from .match import play_hand
from .randomness import derived_seed
from .traditional import LOOSE_CONFIG, TIGHT_CONFIG, TraditionalAgent, TraditionalConfig
from .train import ROOT, make_opponent


def bootstrap_intervals(values, seed, resamples=2000, coverages=(0.95,)):
    """Memory-bounded whole-pair percentile bootstrap; deterministic per seed."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or resamples < 2 or not np.isfinite(values).all():
        raise ValueError("need finite paired outcomes and at least two resamples")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for start in range(0, resamples, 32):
        size = min(32, resamples-start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means[start:start+size] = values[indices].mean(axis=1)
    return {str(coverage): np.quantile(means, [(1-coverage)/2, (1+coverage)/2]).tolist()
            for coverage in coverages}


class MeasuredAgent:
    def __init__(self, agent, measurements):
        self.agent, self.measurements = agent, measurements

    def act(self, obs):
        start = time.perf_counter_ns()
        slot = self.agent.act(obs)
        self.measurements["latency_ns"].append(time.perf_counter_ns()-start)
        action = next(action for action in obs.legal_actions if action.slot == slot)
        kind = ("fold" if slot == 0 else ("call" if obs.call_amount else "check") if slot == 1
                else "raise" if max(obs.street_bets) else "bet")
        self.measurements["actions"][kind] += 1
        return action.slot


def worker(output, label, name, checkpoint, spec, baseline):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    learner = DoubleDQN.load_for_inference(Path(checkpoint)) if checkpoint else None
    if learner is not None and not learner.include_equity:
        raise ValueError("final main panel requires original equity-enabled models")
    calculator = EquityCalculator()
    encoder = FeatureEncoder(calculator)
    measurements = {"latency_ns": [], "actions": Counter()}
    values = []
    path = Path(output) / f"part-{label}-{name}.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["opponent", "deal_id", "deck_seed", "hero_seat", "net_chips"])
        writer.writeheader()
        for deal in range(spec["pairs"]):
            deck_seed = derived_seed(spec["seed"], spec["deal_namespace"], deal)
            deck = shuffled_deck(deck_seed)
            total = 0
            for seat in (0, 1):
                hero = (DQNAgent(learner, encoder) if learner else TraditionalAgent(calculator, baseline,
                        derived_seed(spec["seed"], "final_hero", deal, seat)))
                opponent = make_opponent(name, calculator,
                           derived_seed(spec["seed"], "final_opponent", name, deal, seat), baseline)
                measured = MeasuredAgent(hero, measurements)
                result = play_hand(PokerHand(deck), (measured, opponent) if seat == 0 else (opponent, measured))
                net = result.net_chips[seat]
                total += net
                writer.writerow(dict(opponent=name, deal_id=deal, deck_seed=deck_seed, hero_seat=seat, net_chips=net))
            values.append(total * 25.0)
    result = {"values": values, **measurements, "seconds": time.perf_counter()-start}
    (Path(output) / f"part-{label}-{name}.json").write_text(json.dumps(result) + "\n")
    return label, name


def run(output, source, spec):
    if spec["pairs"] < 2 or not 1 <= spec["workers"] <= 3:
        raise ValueError("invalid sample/worker budget")
    if spec["deal_namespace"] != "poker_final_test_v1":
        raise ValueError("use the isolated final-test namespace")
    checkpoints = {f"seed-{seed}": source / f"seed-{seed}/final.pt" for seed in spec["seeds"]}
    hashes = {label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in checkpoints.items()}
    baseline = TraditionalConfig(**json.loads((ROOT / "configs/traditional-v1.json").read_text())["config"])
    output.mkdir(parents=True, exist_ok=False)
    config = {**spec, "traditional_config": asdict(baseline), "equity": asdict(EquityConfig()),
              "threshold_opponents": {"tight": asdict(TIGHT_CONFIG), "loose_aggressive": asdict(LOOSE_CONFIG)},
              "checkpoint_sha256": hashes, "preflop_sha256": hashlib.sha256(DEFAULT_TABLE.read_bytes()).hexdigest(),
              "checkpoints": {label: str(path.resolve()) for label, path in checkpoints.items()},
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}}
    (output / "run-config.json").write_text(json.dumps(config, indent=2) + "\n")
    start = time.perf_counter()
    labels = [*checkpoints, "traditional"]
    with ProcessPoolExecutor(max_workers=spec["workers"]) as pool:
        futures = [pool.submit(worker, str(output), label, name, str(checkpoints[label]) if label in checkpoints else None, spec, baseline)
                   for name in spec["opponents"] for label in labels]
        for number, future in enumerate(as_completed(futures), 1):
            label, name = future.result()
            print(f"Completed {number}/{len(futures)}: {label} vs {name}", flush=True)
    evaluation_seconds = time.perf_counter()-start
    groups, measurements = {}, {}
    for label in labels:
        groups[label], measurements[label] = {}, {}
        with (output / f"validation-{label}.csv").open("w") as stream:
            for index, name in enumerate(spec["opponents"]):
                with (output / f"part-{label}-{name}.csv").open() as part:
                    header = next(part)
                    if index == 0:
                        stream.write(header)
                    shutil_copy_lines(part, stream)
                data = json.loads((output / f"part-{label}-{name}.json").read_text())
                groups[label][name] = data["values"]
                latency = np.asarray(data["latency_ns"]) / 1e6
                measurements[label][name] = {
                    "decisions": len(latency), "decision_latency_ms_mean": float(latency.mean()),
                    "decision_latency_ms_median": float(np.median(latency)),
                    "decision_latency_ms_p95": float(np.quantile(latency, .95)), "actions": data["actions"],
                }
        (output / f"groups-{label}.json").write_text(json.dumps({"groups": groups[label]}) + "\n")
    audit_logs(output, labels, spec["opponents"], spec["pairs"])
    results = {}
    for label in labels:
        results[label] = {}
        for name in spec["opponents"]:
            values = groups[label][name]
            differences = [a-b for a, b in zip(values, groups["traditional"][name], strict=True)]
            primary = label != "traditional" and name == "traditional"
            coverages = (.95, spec["primary_simultaneous_coverage"]) if primary else (.95,)
            count = spec["primary_bootstrap_resamples"] if primary else spec["secondary_bootstrap_resamples"]
            rng_seed = derived_seed(spec["seed"], "final_bootstrap", label, name)
            intervals = bootstrap_intervals(values, rng_seed, count, coverages)
            diff_intervals = bootstrap_intervals(differences, rng_seed, count, coverages)
            results[label][name] = {"bb_per_100": statistics.mean(values), "intervals": intervals,
                "difference_vs_traditional": statistics.mean(differences), "difference_intervals": diff_intervals,
                "pair_return_sd_bb_per_100": statistics.stdev(values), **measurements[label][name]}
    summary = {"pairs_per_matchup": spec["pairs"], "played_hands": spec["pairs"]*2*len(labels)*len(spec["opponents"]),
               "evaluation_seconds": evaluation_seconds, "wall_seconds": time.perf_counter()-start, "results": results}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Complete: {summary['played_hands']} final-test hands; audited", flush=True)
    return summary


def shutil_copy_lines(source, target):
    # Stream the logs rather than holding every hand in memory.
    for line in source:
        target.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/dqn-three-seed-20261104")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/final-evaluation.json")
    args = parser.parse_args()
    run(args.output, args.source, json.loads(args.config.read_text()))
