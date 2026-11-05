"""Offline, readable replay of a saved model against the traditional agent."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from .dqn import DQNAgent, DoubleDQN
from .environment import PokerHand, shuffled_deck
from .equity import EquityCalculator, EquityConfig
from .features import FeatureEncoder
from .randomness import derived_seed
from .traditional import TraditionalAgent, TraditionalConfig
from .train import ROOT


def replay(checkpoint, pairs=2, seed=17):
    if pairs < 1:
        raise ValueError("pairs must be positive")
    torch.set_num_threads(1)
    model = DoubleDQN.load_for_inference(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    calculator = EquityCalculator(EquityConfig(**saved["metadata"]["equity"]))
    encoder = FeatureEncoder(calculator, include_equity=model.include_equity)
    baseline = TraditionalConfig(**saved["metadata"]["traditional_snapshot"]["config"])
    hands = []
    for deal in range(pairs):
        deck = shuffled_deck(derived_seed(seed, "poker_demo_v1", deal))
        for ml_seat in (0, 1):
            learner = DQNAgent(model, encoder)
            traditional = TraditionalAgent(calculator, baseline, derived_seed(seed, "demo_policy", deal, ml_seat))
            agents = (learner, traditional) if ml_seat == 0 else (traditional, learner)
            hand = PokerHand(deck)
            decisions = []
            while not hand.terminal:
                obs = hand.observe()
                slot = agents[obs.seat].act(obs)
                row = {"actor": "ML" if obs.seat == ml_seat else "Traditional", "observation": asdict(obs), "slot": int(slot)}
                if obs.seat == ml_seat:
                    features = encoder.encode(obs)
                    with torch.no_grad():
                        q = model.online(torch.from_numpy(features).unsqueeze(0))[0].tolist()
                    row["legal_q"] = {str(a.slot.value): q[a.slot.value] for a in obs.legal_actions}
                    row["supplied_equity"] = float(features[117]) if model.include_equity else None
                else:
                    row["rule_trace"] = asdict(traditional.last_decision)
                hand.step(slot)
                decisions.append(row)
            result = hand.result()
            for row, action in zip(decisions, result.history, strict=True):
                row["action"] = asdict(action)
            hands.append({"deal_id": deal, "ml_seat": ml_seat, "decisions": decisions,
                          "result": asdict(result), "ml_net_chips": result.net_chips[ml_seat]})
    return {"purpose": "Offline educational replay, not evaluation or best-hand selection",
            "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "training_seed": model.seed, "demo_seed": seed, "pairs": pairs, "include_equity": model.include_equity,
            "hands": hands}


def markdown(data):
    lines = ["# Poker AI replay", "",
             f"Illustrative model: training seed {data['training_seed']}; demo seed {data['demo_seed']}. "
             "Two seats start with 200 chips each, blinds 1/2. Seat 0 is BB, seat 1 is SB/button.", "",
             "This offline transcript displays each acting player's own cards. Those cards are never passed to the other agent. "
             "The same deal is replayed with agents swapping seats. Hands were not selected for favorable results.", "",
             "ML selects the highest estimated Q among legal actions. Q is an estimated terminal net return divided by 200, "
             "not a win probability or proof that the move is optimal. Equity is showdown pot share against a uniform legal hand.", ""]
    for hand in data["hands"]:
        lines += [f"## Deal {hand['deal_id']}, ML in seat {hand['ml_seat']}", "",
                  "| Street | Actor / seat | Own cards | Board | Pot / call | Action | Explanation |",
                  "|---|---|---|---|---:|---|---|"]
        for row in hand["decisions"]:
            obs, action = row["observation"], row["action"]
            move = action["kind"] + (f" to {action['raise_to']}" if action["raise_to"] is not None else "")
            if "rule_trace" in row:
                trace = row["rule_trace"]
                explanation = f"{trace['reason']}; equity {trace['equity']:.3f}, call threshold {trace['required_equity']:.3f}"
            else:
                explanation = f"highest legal Q = {row['legal_q'][str(row['slot'])]:+.3f}"
                if row["supplied_equity"] is not None:
                    explanation += f"; equity {row['supplied_equity']:.3f}"
            street = ("Preflop", "Flop", "Turn", "River")[obs["street"]]
            lines.append(f"| {street} | {row['actor']} / {obs['seat']} | {' '.join(obs['hole_cards'])} | "
                         f"{' '.join(obs['board']) or '—'} | {obs['pot']} / {obs['call_amount']} | {move} | {explanation} |")
        result = hand["result"]
        lines += ["", f"Final board: {' '.join(result['board']) or 'none'}. "
                  f"ML net: **{hand['ml_net_chips']:+d} chips**. Final stacks by seat: {result['final_stacks']}.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/dqn-three-seed-20261104/seed-101/final.pt")
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, help="Fresh output directory; otherwise print readable replay")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("output already exists")
    data = replay(args.checkpoint, args.pairs, args.seed)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "replay.json").write_text(json.dumps(data, indent=2) + "\n")
        (args.output / "README.md").write_text(markdown(data))
    else:
        print(markdown(data))
