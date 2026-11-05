"""CPU Double DQN development training with fixed validation and full-hand rewards."""

import argparse
from collections import Counter, deque
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import random
import resource
import statistics
import time

import numpy as np
import torch

from .agents import PassiveAgent, RandomAgent
from .dqn import DQNAgent, DQNConfig, DoubleDQN
from .environment import PokerHand, shuffled_deck
from .equity import DEFAULT_TABLE, EquityCalculator, EquityConfig
from .features import FEATURE_SIZE, FEATURE_VERSION, FeatureEncoder
from .learning_env import LearnerEpisode
from .match import play_hand
from .randomness import derived_seed
from .replay import ReplayBuffer
from .traditional import LOOSE_CONFIG, TIGHT_CONFIG, ThresholdAgent, TraditionalAgent, TraditionalConfig
from .traditional_eval import paired_interval

ROOT = Path(__file__).resolve().parents[1]
OPPONENT_MIXTURE = {"random": 0.10, "passive": 0.20, "tight": 0.35, "loose_aggressive": 0.35}


def make_opponent(name, calculator, seed, traditional_config):
    if name == "random":
        return RandomAgent(seed)
    if name == "passive":
        return PassiveAgent()
    if name in ("tight", "loose_aggressive"):
        return ThresholdAgent(calculator, TIGHT_CONFIG if name == "tight" else LOOSE_CONFIG)
    if name == "traditional":
        return TraditionalAgent(calculator, traditional_config, seed)
    raise ValueError("unknown opponent")


def validate(output, label, learner, pairs, seed, traditional_config):
    """learner=None measures the frozen traditional baseline on identical deals."""
    calculator = EquityCalculator()
    encoder = FeatureEncoder(calculator, include_equity=learner.include_equity if learner is not None else True)
    groups = {}
    with (output / f"validation-{label}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["opponent", "deal_id", "deck_seed", "hero_seat", "net_chips"])
        writer.writeheader()
        for name in (*OPPONENT_MIXTURE, "traditional"):
            values = []
            for deal_id in range(pairs):
                deck_seed = derived_seed(seed, "dqn_validation_deal", deal_id)
                deck = shuffled_deck(deck_seed)
                total = 0
                for seat in (0, 1):
                    hero = (DQNAgent(learner, encoder) if learner is not None else
                            TraditionalAgent(calculator, traditional_config,
                                             derived_seed(seed, "validation_hero", deal_id, seat)))
                    opponent = make_opponent(name, calculator,
                                             derived_seed(seed, "validation_opponent", name, deal_id, seat), traditional_config)
                    result = play_hand(PokerHand(deck), (hero, opponent) if seat == 0 else (opponent, hero))
                    net = result.net_chips[seat]
                    total += net
                    writer.writerow(dict(opponent=name, deal_id=deal_id, deck_seed=deck_seed,
                                         hero_seat=seat, net_chips=net))
                values.append(total * 25.0)
            groups[name] = values
    return groups


def summarize_validation(groups, reference, seed):
    summary = {}
    for name, values in groups.items():
        differences = [a - b for a, b in zip(values, reference[name])]
        summary[name] = {
            "bb_per_100": statistics.mean(values),
            "ci95": paired_interval(values, derived_seed(seed, "validation_absolute_ci", name)),
            "difference_vs_traditional": statistics.mean(differences),
            "difference_ci95": paired_interval(differences, derived_seed(seed, "validation_difference_ci", name)),
        }
    return summary


def peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak * 1024)


def run_pilot(output: Path, decisions: int = 25_000, seed: int = 101,
              validation_pairs: int = 100, config: DQNConfig = DQNConfig(),
              validation_every: int | None = None, validation_seed: int | None = None,
              threads: int = 2, include_equity: bool = True) -> dict:
    if type(include_equity) is not bool:
        raise ValueError("include_equity must be boolean")
    if type(decisions) is not int or decisions < config.warmup:
        raise ValueError("decision budget must reach warmup")
    if type(validation_pairs) is not int or validation_pairs < 2:
        raise ValueError("validation needs at least two paired deals")
    if validation_every is not None and (type(validation_every) is not int or validation_every <= 0):
        raise ValueError("validation_every must be a positive integer or None")
    if type(threads) is not int or not 1 <= threads <= 4:
        raise ValueError("threads must be between one and four")
    validation_seed = seed if validation_seed is None else validation_seed
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    snapshot = json.loads((ROOT / "configs" / "traditional-v1.json").read_text())
    traditional_config = TraditionalConfig(**snapshot["config"])
    metadata = {
        "seed": seed, "requested_decisions": decisions, "validation_pairs": validation_pairs,
        "include_equity": include_equity,
        "dqn": asdict(config), "equity": asdict(EquityConfig()), "feature_version": FEATURE_VERSION,
        "opponent_mixture": OPPONENT_MIXTURE,
        "threshold_opponents": {"tight": asdict(TIGHT_CONFIG), "loose_aggressive": asdict(LOOSE_CONFIG)},
        "traditional_snapshot": snapshot, "threads": threads, "device": "cpu",
        "validation_every": validation_every, "validation_seed": validation_seed,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
        "preflop_sha256": hashlib.sha256(DEFAULT_TABLE.read_bytes()).hexdigest(),
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                    "pokerkit": "0.7.7", "treys": "0.1.8", "platform": platform.platform()},
        "checkpoint_selection": "Fixed final budget; intermediate validation is descriptive, no best-checkpoint search",
    }
    (output / "run-config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    learner = DoubleDQN(config, seed, include_equity=include_equity)
    learner.save(output / "initial.pt", 0, metadata)
    print("Validating frozen traditional baseline and untrained network on separate deals", flush=True)
    baseline = validate(output, "traditional", None, validation_pairs, validation_seed, traditional_config)
    initial = validate(output, "initial", learner, validation_pairs, validation_seed, traditional_config)
    curve_path = output / "learning-curve.jsonl"
    with curve_path.open("w") as stream:
        stream.write(json.dumps({"decisions": 0, "checkpoint": "initial.pt",
                                 "matchups": summarize_validation(initial, baseline, validation_seed)}) + "\n")
    encoder = FeatureEncoder(EquityCalculator(), include_equity=include_equity)
    replay = ReplayBuffer(config.replay_capacity)
    exploration_rng = random.Random(derived_seed(seed, "dqn_exploration"))
    mixture_rng = random.Random(derived_seed(seed, "training_opponent_mixture"))
    replay_rng = np.random.default_rng(derived_seed(seed, "replay_sampling"))
    recent_losses = deque(maxlen=100)
    first_losses = []
    steps = hands = no_decision_hands = terminal_transitions = 0
    opponent_counts, action_counts = Counter(), Counter()
    start = time.perf_counter()
    validation_seconds = 0.0
    next_validation = validation_every
    with (output / "training-hands.csv").open("w", newline="") as hand_file, (output / "training-progress.csv").open("w", newline="") as progress_file:
        hand_writer = csv.DictWriter(hand_file, fieldnames=["hand_id", "opponent", "learner_seat", "decisions", "net_chips"])
        progress_writer = csv.DictWriter(progress_file, fieldnames=["decisions", "hands_completed", "epsilon", "updates", "mean_recent_loss", "elapsed_seconds", "peak_rss_bytes"])
        hand_writer.writeheader()
        progress_writer.writeheader()
        # Finish the final hand instead of falsely labelling a budget cutoff terminal.
        while steps < decisions:
            name = mixture_rng.choices(list(OPPONENT_MIXTURE), weights=list(OPPONENT_MIXTURE.values()))[0]
            opponent_counts[name] += 1
            seat = hands % 2
            opponent = make_opponent(name, encoder.calculator,
                                     derived_seed(seed, "training_opponent_policy", hands), traditional_config)
            episode = LearnerEpisode(shuffled_deck(derived_seed(seed, "dqn_training_deal", hands)), seat, opponent)
            observation = episode.observation
            hand_steps = 0
            if observation is None:
                no_decision_hands += 1
            else:
                features = encoder.encode(observation)
                while observation is not None:
                    mask = np.asarray(observation.action_mask, dtype=bool)
                    action = learner.choose(features, mask, config.epsilon(steps), exploration_rng)
                    action_counts[str(action)] += 1
                    outcome = episode.step(action)
                    if outcome.terminal:
                        next_features = np.zeros(FEATURE_SIZE, dtype=np.float32)
                        next_mask = np.zeros(6, dtype=bool)
                        terminal_transitions += 1
                    else:
                        next_features = encoder.encode(outcome.observation)
                        next_mask = np.asarray(outcome.observation.action_mask, dtype=bool)
                    replay.add(features, mask, action, outcome.reward, next_features, next_mask, outcome.terminal)
                    steps += 1
                    hand_steps += 1
                    if steps >= config.warmup and steps % config.update_every == 0:
                        metrics = learner.update(replay.sample(config.batch_size, replay_rng))
                        recent_losses.append(metrics["loss"])
                        if len(first_losses) < 100:
                            first_losses.append(metrics["loss"])
                    if steps % 1000 == 0:
                        progress_writer.writerow(dict(decisions=steps, hands_completed=hands,
                                                      epsilon=config.epsilon(steps), updates=learner.gradient_updates,
                                                      mean_recent_loss=statistics.mean(recent_losses) if recent_losses else None,
                                                      elapsed_seconds=time.perf_counter() - start - validation_seconds, peak_rss_bytes=peak_rss_bytes()))
                        progress_file.flush()
                        print(f"{steps} learner decisions; {learner.gradient_updates} updates; epsilon={config.epsilon(steps):.3f}", flush=True)
                    observation, features = outcome.observation, next_features
            hand_writer.writerow(dict(hand_id=hands, opponent=name, learner_seat=seat,
                                      decisions=hand_steps, net_chips=episode.result().net_chips[seat]))
            hands += 1
            if next_validation is not None and steps >= next_validation and steps < decisions:
                validation_start = time.perf_counter()
                label = f"step-{steps:06d}"
                learner.save(output / f"{label}.pt", steps, metadata)
                print(f"Validating checkpoint at {steps} decisions", flush=True)
                current = validate(output, label, learner, validation_pairs, validation_seed, traditional_config)
                with curve_path.open("a") as stream:
                    stream.write(json.dumps({"decisions": steps, "checkpoint": f"{label}.pt",
                                             "matchups": summarize_validation(current, baseline, validation_seed)}) + "\n")
                validation_seconds += time.perf_counter() - validation_start
                next_validation += validation_every
    training_seconds = time.perf_counter() - start - validation_seconds
    learner.save(output / "final.pt", steps, metadata)
    print("Validating the fixed final pilot checkpoint (epsilon=0, no updates)", flush=True)
    # Use the saved checkpoint to exercise the actual inference loading path.
    restored = DoubleDQN.load_for_inference(output / "final.pt")
    final = validate(output, "final", restored, validation_pairs, validation_seed, traditional_config)
    with curve_path.open("a") as stream:
        stream.write(json.dumps({"decisions": steps, "checkpoint": "final.pt",
                                 "matchups": summarize_validation(final, baseline, validation_seed)}) + "\n")
    validation = {
        "traditional": summarize_validation(baseline, baseline, validation_seed),
        "initial": summarize_validation(initial, baseline, validation_seed),
        "final": summarize_validation(final, baseline, validation_seed),
        "final_vs_initial": summarize_validation(final, initial, validation_seed),
    }
    # Name the reference accurately for this additional within-DQN comparison.
    for entry in validation["final_vs_initial"].values():
        entry["difference_vs_initial"] = entry.pop("difference_vs_traditional")
    summary = {
        "purpose": "single-seed development training; not final held-out evaluation",
        "include_equity": include_equity,
        "seed": seed, "requested_decisions": decisions, "actual_decisions": steps,
        "hands": hands, "no_decision_hands": no_decision_hands, "terminal_transitions": terminal_transitions,
        "gradient_updates": learner.gradient_updates, "epsilon_end_of_pilot": config.epsilon(steps),
        "parameters": sum(p.numel() for p in learner.online.parameters()),
        "replay_entries": replay.size, "replay_capacity": replay.capacity, "replay_allocated_bytes": replay.nbytes,
        "training_seconds": training_seconds, "decisions_per_second": steps / training_seconds,
        "intermediate_validation_seconds": validation_seconds,
        "validation_seed": validation_seed,
        "peak_process_rss_bytes": peak_rss_bytes(),
        "mean_first_100_update_loss": statistics.mean(first_losses) if first_losses else None,
        "mean_last_100_update_loss": statistics.mean(recent_losses) if recent_losses else None,
        "opponent_hand_counts": opponent_counts, "training_action_slot_counts": action_counts,
        "validation_pairs_per_opponent": validation_pairs, "validation": validation,
        "limitations": "Training throughput excludes validation; RSS is the entire process high-water mark. "
                       "Training loss is not playing strength. One seed and reused validation deals "
                       "do not establish generalization, convergence or superiority. No final-test deals were used.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "dqn-pilot.json")
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    if spec["opponent_mixture"] != OPPONENT_MIXTURE:
        parser.error("this runner supports the fixed four-opponent mixture")
    print(json.dumps(run_pilot(args.output, spec["requested_decisions"], spec["seed"],
                               spec["validation_pairs"], DQNConfig(**spec["dqn"]),
                               spec.get("validation_every"), spec.get("validation_seed"), spec["threads"],
                               spec.get("include_equity", True)), indent=2))


if __name__ == "__main__":
    main()
