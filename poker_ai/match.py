"""Paired-deal smoke runner, not the final statistical experiment suite."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

from .agents import Agent, PassiveAgent, RandomAgent
from .environment import GameConfig, PokerHand, shuffled_deck
from .randomness import derived_seed


def play_hand(hand: PokerHand, agents: tuple[Agent, Agent]):
    while not hand.terminal:
        obs = hand.observe()
        hand.step(agents[obs.seat].act(obs))
    return hand.result()


def run_match(pairs: int, seed: int, output: Path) -> dict:
    if pairs <= 0:
        raise ValueError("pairs must be positive")
    # A new directory prevents silently overwriting another experiment.
    output.mkdir(parents=True, exist_ok=False)
    config = GameConfig()
    total = 0
    counts = {"random": {}, "passive": {}}
    with (output / "hands.csv").open("w", newline="") as csv_file, (output / "actions.jsonl").open("w") as traces:
        columns = ["deal_id", "deck_seed", "random_seat", "random_net_chips",
                   "passive_net_chips", "decision_count", "board"]
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for deal_id in range(pairs):
            deck_seed = derived_seed(seed, "smoke_deal", deal_id)
            deck = shuffled_deck(deck_seed)
            for random_seat in (0, 1):
                random_agent = RandomAgent(derived_seed(seed, "policy", deal_id, random_seat))
                agents = ((random_agent, PassiveAgent()) if random_seat == 0
                          else (PassiveAgent(), random_agent))
                result = play_hand(PokerHand(deck, config), agents)
                net = result.net_chips[random_seat]
                total += net
                writer.writerow(dict(deal_id=deal_id, deck_seed=deck_seed,
                                     random_seat=random_seat, random_net_chips=net,
                                     passive_net_chips=-net, decision_count=len(result.history),
                                     board=" ".join(result.board)))
                for action in result.history:
                    name = "random" if action.seat == random_seat else "passive"
                    counts[name][action.kind] = counts[name].get(action.kind, 0) + 1
                traces.write(json.dumps({"deal_id": deal_id, "random_seat": random_seat,
                                         "actions": [asdict(a) for a in result.history]}) + "\n")
    summary = {
        "purpose": "milestone-1 correctness smoke run, not strength evidence",
        "seed": seed, "pairs": pairs, "hands": 2 * pairs,
        "game": asdict(config), "pokerkit": "0.7.7",
        "seat_mapping": {"0": "big blind", "1": "small blind/button"},
        "random_net_chips": total, "passive_net_chips": -total,
        "random_bb_per_100": 100 * total / (2 * pairs * config.big_blind),
        "action_counts": counts,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs <= 0:
        parser.error("--pairs must be positive")
    print(json.dumps(run_match(args.pairs, args.seed, args.output), indent=2))


if __name__ == "__main__":
    main()
