"""Small, frozen development comparison; never the final poker experiment."""

import argparse
from collections import Counter
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

from .agents import PassiveAgent, RandomAgent
from .environment import GameConfig, PokerHand, shuffled_deck
from .equity import DEFAULT_TABLE, EquityCalculator, EquityConfig
from .randomness import derived_seed
from .traditional import ThresholdAgent, ThresholdConfig, TraditionalAgent, TraditionalConfig


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "traditional-development.json"


def paired_interval(values: list[float], seed: int, resamples: int = 2000) -> list[float]:
    """Percentile interval resampling independent deal groups, never seat hands."""
    if len(values) < 2 or resamples < 2:
        raise ValueError("need at least two deal groups and resamples")
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(resamples))
    def quantile(p):
        index = (len(means) - 1) * p
        lower = int(index)
        upper = min(lower + 1, len(means) - 1)
        return means[lower] + (means[upper] - means[lower]) * (index - lower)
    return [quantile(0.025), quantile(0.975)]


def _opponent(spec, calculator, seed):
    if spec["kind"] == "random":
        return RandomAgent(seed)
    if spec["kind"] == "passive":
        return PassiveAgent()
    if spec["kind"] == "threshold":
        return ThresholdAgent(calculator, ThresholdConfig(**{k: v for k, v in spec.items() if k != "kind"}))
    raise ValueError("unknown opponent kind")


def run_development(output: Path, pairs: int = 250, seed: int = 20261104,
                    config_path: Path = DEFAULT_CONFIG) -> dict:
    if type(pairs) is not int or pairs < 2:
        raise ValueError("pairs must be an integer >= 2")
    spec = json.loads(config_path.read_text())
    if spec["schema_version"] != 1 or spec["candidates"] != ["no_bluff", "bluff_05"]:
        raise ValueError("unsupported development selection configuration")
    configs = {name: TraditionalConfig(**params) for name, params in spec["policies"].items()}
    calculator = EquityCalculator()
    for opponent in spec["opponents"].values():
        _opponent(opponent, calculator, seed)  # Validate before starting a run.
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in Path(__file__).parent.glob("*.py")}
    metadata = {"seed": seed, "pairs": pairs, "config": spec, "game": asdict(GameConfig()),
                "equity": asdict(EquityConfig()), "source_sha256": source_hashes,
                "preflop_table_sha256": hashlib.sha256(DEFAULT_TABLE.read_bytes()).hexdigest(),
                "versions": {"pokerkit": "0.7.7", "treys": "0.1.8"}}
    (output / "run-config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    returns = {name: {} for name in configs}
    reasons = {name: Counter() for name in configs}
    actions = {name: Counter() for name in configs}
    latencies = {name: [] for name in configs}
    start = time.perf_counter()
    with (output / "hands.csv").open("w", newline="") as stream, (output / "decisions.jsonl").open("w") as trace:
        writer = csv.DictWriter(stream, fieldnames=["opponent", "policy", "deal_id", "deck_seed",
                                                    "hero_seat", "net_chips", "decisions"])
        writer.writeheader()
        for opponent_name, opponent_spec in spec["opponents"].items():
            for name in configs:
                returns[name][opponent_name] = []
            for deal_id in range(pairs):
                deck_seed = derived_seed(seed, "traditional_development_deal", deal_id)
                deck = shuffled_deck(deck_seed)
                for name, config in configs.items():
                    group_chips = 0
                    for hero_seat in (0, 1):
                        hero = TraditionalAgent(calculator, config, derived_seed(seed, "traditional_policy", deal_id, hero_seat))
                        opponent = _opponent(opponent_spec, calculator,
                                             derived_seed(seed, "development_opponent", opponent_name, deal_id, hero_seat))
                        agents = (hero, opponent) if hero_seat == 0 else (opponent, hero)
                        hand = PokerHand(deck)
                        while not hand.terminal:
                            obs = hand.observe()
                            agent = agents[obs.seat]
                            before = time.perf_counter()
                            slot = agent.act(obs)
                            duration = 1000 * (time.perf_counter() - before)
                            if obs.seat == hero_seat:
                                decision = hero.last_decision
                                reasons[name][decision.reason] += 1
                                latencies[name].append(duration)
                            hand.step(slot)
                            trace.write(json.dumps({"opponent": opponent_name, "policy": name,
                                                    "deal_id": deal_id, "hero_seat": hero_seat,
                                                    "actor": obs.seat, "street": obs.street,
                                                    "pot": obs.pot, "call_amount": obs.call_amount,
                                                    "slot": slot,
                                                    "decision": asdict(agent.last_decision) if hasattr(agent, "last_decision") else None}) + "\n")
                        result = hand.result()
                        net = result.net_chips[hero_seat]
                        group_chips += net
                        for action in result.history:
                            if action.seat == hero_seat:
                                actions[name][action.kind] += 1
                        writer.writerow(dict(opponent=opponent_name, policy=name, deal_id=deal_id,
                                             deck_seed=deck_seed, hero_seat=hero_seat, net_chips=net,
                                             decisions=len(result.history)))
                    # Pair mean chips / 2-chip BB * 100 = pair sum * 25.
                    returns[name][opponent_name].append(group_chips * 25.0)
                if (deal_id + 1) % 50 == 0:
                    print(f"{opponent_name}: {deal_id + 1}/{pairs} paired deals for all policies", flush=True)
    results = {}
    for name in configs:
        results[name] = {}
        for opponent_name, values in returns[name].items():
            differences = [v - b for v, b in zip(values, returns["no_bluff"][opponent_name])]
            results[name][opponent_name] = {
                "bb_per_100": statistics.mean(values),
                "ci95": paired_interval(values, derived_seed(seed, "bootstrap_absolute", opponent_name)),
                "difference_vs_no_bluff": statistics.mean(differences),
                "difference_ci95": paired_interval(differences, derived_seed(seed, "bootstrap_difference", opponent_name)),
            }
    # Opponents share deal IDs. Average their paired differences per deal
    # before bootstrap, preserving cross-opponent correlation.
    improvements = [statistics.mean(returns["bluff_05"][opp][i] - returns["no_bluff"][opp][i]
                                     for opp in spec["opponents"]) for i in range(pairs)]
    interval = paired_interval(improvements, derived_seed(seed, "bootstrap_selection"))
    selected = "bluff_05" if interval[0] > 0 else "no_bluff"
    summary = {
        "purpose": "development comparison, not final held-out performance",
        "seed": seed, "pairs_per_matchup": pairs,
        "total_hands": pairs * 2 * len(configs) * len(spec["opponents"]),
        "matchups": results, "selected_policy": selected,
        "bluff_improvement_bb_per_100": statistics.mean(improvements), "bluff_improvement_ci95": interval,
        "selection_rule": spec["selection_rule"], "reason_counts": reasons, "action_counts": actions,
        "hero_latency_median_ms": {name: statistics.median(v) for name, v in latencies.items()},
        "latency_caveat": "Shared warm cache and fixed policy order; these are diagnostics, not fair speed comparisons.",
        "elapsed_seconds": time.perf_counter() - start,
        "limitations": "Small fixed development budget. Percentile intervals resample whole deal groups. "
                       "Tuning performance is optimistic; final held-out evaluation remains pending.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "selected-policy.json").write_text(json.dumps({
        "policy": selected, "config": asdict(configs[selected]), "development_seed": seed,
        "status": "frozen after predeclared development comparison; not final-test validated",
    }, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    print(json.dumps(run_development(args.output, args.pairs, args.seed), indent=2))


if __name__ == "__main__":
    main()
