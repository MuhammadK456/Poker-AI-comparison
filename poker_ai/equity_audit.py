"""Reproducible equity error study and local timing/integration measurements."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import time

from .agents import PassiveAgent, RandomAgent
from .environment import FULL_DECK, PokerHand, shuffled_deck
from .equity import DEFAULT_TABLE, EquityCalculator, EquityConfig, exact_river, monte_carlo
from .randomness import derived_seed


def audit(output: Path, states: int = 100, repeats: int = 10,
          seed: int = 20261103, hands: int = 1000) -> dict:
    if any(type(x) is not int or x <= 0 for x in (states, repeats, hands)):
        raise ValueError("states, repeats and hands must be positive integers")
    output.mkdir(parents=True, exist_ok=False)
    budgets = (128, 256, 1024)
    errors = {n: [] for n in budgets}
    timings = {n: [] for n in budgets}
    exact_times = []
    with (output / "river-errors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["state_id", "hole", "board", "samples",
                                                   "repeat", "exact", "estimate", "error", "milliseconds"])
        writer.writeheader()
        for state_id in range(states):
            rng = random.Random(derived_seed(seed, "equity_audit_state", state_id))
            cards = rng.sample(FULL_DECK, 7)
            hole, board = cards[:2], cards[2:]
            start = time.perf_counter()
            reference = exact_river(hole, board).equity
            exact_times.append(1000 * (time.perf_counter() - start))
            for budget in budgets:
                for repeat in range(repeats):
                    start = time.perf_counter()
                    result = monte_carlo(hole, board, samples=budget,
                                         seed=derived_seed(seed, "equity_audit_mc", state_id, budget, repeat))
                    milliseconds = 1000 * (time.perf_counter() - start)
                    error = result.equity - reference
                    errors[budget].append(error)
                    timings[budget].append(milliseconds)
                    writer.writerow(dict(state_id=state_id, hole=" ".join(hole), board=" ".join(board),
                                         samples=budget, repeat=repeat, exact=reference,
                                         estimate=result.equity, error=error, milliseconds=milliseconds))
            if (state_id + 1) % 20 == 0:
                print(f"Audited {state_id + 1}/{states} river states", flush=True)
    error_summary = {}
    for n in budgets:
        error_summary[n] = {
            "mae": statistics.mean(abs(e) for e in errors[n]),
            "rmse": statistics.mean(e * e for e in errors[n])**0.5,
            "bias": statistics.mean(errors[n]),
            "median_ms": statistics.median(timings[n]),
        }
    flop_timings = {}
    for n in budgets:
        elapsed = []
        for repeat in range(30):
            start = time.perf_counter()
            monte_carlo(("As", "Ks"), ("Qs", "Js", "2d"), samples=n,
                        seed=derived_seed(seed, "flop_timing", repeat))
            elapsed.append(1000 * (time.perf_counter() - start))
        flop_timings[n] = statistics.median(elapsed)

    # Exercise the calculator on legal, live observations from both seats.
    # Decisions deliberately remain random/passive: this tests integration,
    # not whether an equity-aware strategy is profitable.
    config = EquityConfig()
    calculator = EquityCalculator(config)
    decisions = 0
    methods = {}
    start = time.perf_counter()
    for hand_id in range(hands):
        hand = PokerHand(shuffled_deck(derived_seed(seed, "equity_integration_deal", hand_id)))
        random_agent = RandomAgent(derived_seed(seed, "equity_integration_policy", hand_id))
        agents = (random_agent, PassiveAgent()) if hand_id % 2 == 0 else (PassiveAgent(), random_agent)
        while not hand.terminal:
            observation = hand.observe()
            result = calculator.from_observation(observation)
            if not 0 <= result.equity <= 1:
                raise RuntimeError("invalid equity")
            methods[result.method] = methods.get(result.method, 0) + 1
            hand.step(agents[observation.seat].act(observation))
            decisions += 1
        if sum(hand.result().net_chips) != 0:
            raise RuntimeError("nonzero-sum settlement")
        if (hand_id + 1) % 200 == 0:
            print(f"Checked equity on {hand_id + 1}/{hands} live hands", flush=True)
    seconds = time.perf_counter() - start
    summary = {
        "seed": seed, "states": states, "repeats_per_state": repeats,
        "calculator_config": asdict(config),
        "preflop_table_sha256": hashlib.sha256(DEFAULT_TABLE.read_bytes()).hexdigest(),
        "opponent_range": "uniform_legal", "reference": "exact river enumeration, 990 combinations",
        "error_units": "pot-share probability, multiply by 100 for percentage points",
        "river": error_summary, "exact_river_median_ms": statistics.median(exact_times),
        "flop_median_ms": flop_timings,
        "integration": {"hands": hands, "decisions": decisions, "seconds": seconds,
                        "hands_per_second": hands / seconds, "methods": methods,
                        "cache_entries": calculator.cache_entries},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "treys": "0.1.8", "pokerkit": "0.7.7"},
        "limitations": "Uniform-range river errors do not measure range-model error or general flop accuracy. "
                       "Timings are local measurements, not training-time forecasts.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--hands", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(audit(args.output, args.states, args.repeats, args.seed, args.hands), indent=2))


if __name__ == "__main__":
    main()
