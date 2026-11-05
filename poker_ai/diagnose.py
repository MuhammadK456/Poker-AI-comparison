"""Describe a frozen learned policy on validation deals, without tuning it."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import torch

from .dqn import DoubleDQN, masked_argmax
from .environment import Action, PokerHand, shuffled_deck
from .equity import EquityCalculator, EquityConfig
from .features import FeatureEncoder
from .randomness import derived_seed
from .traditional import TraditionalConfig
from .train import OPPONENT_MIXTURE, make_opponent


def describe_action(obs, slot, features, q_values):
    action = next(a for a in obs.legal_actions if a.slot == slot)
    kind = ("fold" if slot == Action.FOLD else
            ("call" if obs.call_amount else "check") if slot == Action.CHECK_CALL else
            "raise" if max(obs.street_bets) else "bet")
    commits = (action.raise_to is not None and
               action.raise_to >= min(obs.stacks[i] + obs.street_bets[i] for i in (0, 1)))
    equity = float(features[117])
    return {"street": obs.street, "seat": obs.seat, "equity": equity,
            "pot_odds": float(features[116]), "facing_bet": bool(obs.call_amount),
            "action": kind, "slot": int(slot), "raise_to": action.raise_to,
            "effective_allin_raise": bool(commits),
            "allin_below_50_equity": bool(commits and equity < 0.5),
            "raise_below_50_equity": bool(action.raise_to is not None and equity < 0.5),
            "selected_q": float(q_values[slot]),
            "selected_q_outside_return_bounds": abs(float(q_values[slot])) > 1.0,
            "equity_bin": "below_30" if equity < 0.3 else "30_to_50" if equity < 0.5 else "50_to_70" if equity < 0.7 else "at_least_70"}


def diagnose(checkpoint: Path, output: Path, pairs: int = 100, seed: int | None = None):
    if pairs < 1:
        raise ValueError("pairs must be positive")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    seed = saved["seed"] if seed is None else seed
    metadata = saved["metadata"]
    config = TraditionalConfig(**metadata["traditional_snapshot"]["config"])
    torch.set_num_threads(1)
    learner = DoubleDQN.load_for_inference(checkpoint)
    calculator = EquityCalculator(EquityConfig(**metadata["equity"]))
    encoder = FeatureEncoder(calculator, include_equity=learner.include_equity)
    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    with (output / "decisions.csv").open("w", newline="") as stream:
        writer = None
        for name in (*OPPONENT_MIXTURE, "traditional"):
            counts = Counter()
            by_street, by_equity = defaultdict(Counter), defaultdict(Counter)
            equity_totals = Counter()
            total_chips = 0
            for deal in range(pairs):
                deck = shuffled_deck(derived_seed(seed, "dqn_validation_deal", deal))
                for seat in (0, 1):
                    hand = PokerHand(deck)
                    opponent = make_opponent(name, calculator,
                                             derived_seed(seed, "validation_opponent", name, deal, seat), config)
                    while not hand.terminal:
                        obs = hand.observe()
                        if obs.seat == seat:
                            features = encoder.encode(obs)
                            with torch.no_grad():
                                values = learner.online(torch.from_numpy(features).unsqueeze(0))
                                slot = int(masked_argmax(values, torch.tensor([obs.action_mask]))[0])
                            diagnostic_features = features.copy()
                            # For diagnostics only; this estimate never enters the ablated network.
                            if not learner.include_equity:
                                diagnostic_features[117] = calculator.from_observation(obs).equity
                            row = {"opponent": name, "deal_id": deal,
                                   **describe_action(obs, slot, diagnostic_features, values[0].tolist())}
                            if writer is None:
                                writer = csv.DictWriter(stream, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                            counts["decisions"] += 1
                            counts[row["action"]] += 1
                            for field in ("facing_bet", "effective_allin_raise", "allin_below_50_equity",
                                          "raise_below_50_equity", "selected_q_outside_return_bounds"):
                                counts[field] += row[field]
                            by_street[str(obs.street)][row["action"]] += 1
                            by_equity[row["equity_bin"]][row["action"]] += 1
                            equity_totals[row["action"]] += row["equity"]
                        else:
                            slot = opponent.act(obs)
                        hand.step(slot)
                    total_chips += hand.result().net_chips[seat]
            summaries[name] = {"bb_per_100": 100 * total_chips / (pairs * 2 * 2),
                               "counts": counts, "by_street": by_street, "by_equity": by_equity,
                               "mean_equity_by_action": {k: value / counts[k] for k, value in equity_totals.items()}}
    result = {"checkpoint": str(checkpoint), "checkpoint_decisions": saved["decisions"],
              "validation_seed": seed, "pairs_per_opponent": pairs, "opponents": summaries,
              "limits": "Descriptive on-policy validation states, not counterfactual action EVs. "
                        "Weak-equity aggression can sometimes be a rational bluff; these counts alone do not prove an error."}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.checkpoint, args.output, args.pairs, args.seed), indent=2))


if __name__ == "__main__":
    main()
