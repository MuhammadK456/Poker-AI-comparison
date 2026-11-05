"""Small frozen development check of three opponent parameter variations."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch

from .dqn import DQNAgent, DoubleDQN
from .environment import PokerHand, shuffled_deck
from .equity import DEFAULT_TABLE, EquityCalculator, EquityConfig
from .evaluation_report import audit_logs
from .features import FeatureEncoder
from .match import play_hand
from .randomness import derived_seed
from .traditional import ThresholdAgent, ThresholdConfig, TraditionalAgent, TraditionalConfig
from .train import ROOT, summarize_validation


def make_variant(spec, calculator, seed, baseline):
    if spec["kind"] == "threshold":
        return ThresholdAgent(calculator, ThresholdConfig(**{k: v for k, v in spec.items() if k != "kind"}))
    if spec["kind"] == "traditional":
        return TraditionalAgent(calculator, TraditionalConfig(**{**baseline, **{k: v for k, v in spec.items() if k != "kind"}}), seed)
    raise ValueError("unsupported robustness opponent")


def worker(output, label, checkpoint, spec, baseline):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    learner = DoubleDQN.load_for_inference(Path(checkpoint)) if checkpoint else None
    calculator = EquityCalculator()
    encoder = FeatureEncoder(calculator, include_equity=learner.include_equity if learner else True)
    groups = {}
    with (Path(output) / f"validation-{label}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["opponent", "deal_id", "deck_seed", "hero_seat", "net_chips"])
        writer.writeheader()
        for name, opponent_spec in spec["opponents"].items():
            values = []
            for deal in range(spec["pairs"]):
                deck_seed = derived_seed(spec["seed"], "robustness_development_deal", deal)
                deck = shuffled_deck(deck_seed)
                total = 0
                for seat in (0, 1):
                    hero = (DQNAgent(learner, encoder) if learner else TraditionalAgent(calculator, TraditionalConfig(**baseline),
                            derived_seed(spec["seed"], "robustness_hero", deal, seat)))
                    opponent = make_variant(opponent_spec, calculator,
                               derived_seed(spec["seed"], "robustness_opponent", name, deal, seat), baseline)
                    result = play_hand(PokerHand(deck), (hero, opponent) if seat == 0 else (opponent, hero))
                    net = result.net_chips[seat]
                    total += net
                    writer.writerow(dict(opponent=name, deal_id=deal, deck_seed=deck_seed, hero_seat=seat, net_chips=net))
                values.append(total * 25.0)
            groups[name] = values
    result = {"groups": groups, "seconds": time.perf_counter() - start}
    (Path(output) / f"groups-{label}.json").write_text(json.dumps(result) + "\n")
    return result


def run(output, source, spec):
    if type(spec["pairs"]) is not int or spec["pairs"] < 2:
        raise ValueError("need at least two pairs")
    baseline = json.loads((ROOT / "configs/traditional-v1.json").read_text())["config"]
    for variant in spec["opponents"].values():
        make_variant(variant, None, 0, baseline)
    checkpoints = {f"seed-{seed}": source / f"seed-{seed}/final.pt" for seed in (101, 202, 303)}
    hashes = {label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in checkpoints.items()}
    output.mkdir(parents=True, exist_ok=False)
    config = {**spec, "traditional_config": baseline, "equity": asdict(EquityConfig()),
              "checkpoint_sha256": hashes, "preflop_sha256": hashlib.sha256(DEFAULT_TABLE.read_bytes()).hexdigest(),
              "checkpoints": {label: str(path.resolve()) for label, path in checkpoints.items()},
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}}
    (output / "run-config.json").write_text(json.dumps(config, indent=2) + "\n")
    start = time.perf_counter()
    results = {}
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(worker, str(output), label, str(path) if path else None, spec, baseline): label
                   for label, path in {"traditional": None, **checkpoints}.items()}
        for future in as_completed(futures):
            label = futures[future]
            results[label] = future.result()
            print(f"Completed {label}: {results[label]['seconds']:.1f}s", flush=True)
    audit_logs(output, list(results), list(spec["opponents"]), spec["pairs"])
    summary = {"pairs_per_matchup": spec["pairs"], "played_hands": spec["pairs"] * 2 * len(spec["opponents"]) * 4,
               "wall_seconds": time.perf_counter() - start, "results": {
                   label: {"matchups": summarize_validation(result["groups"], results["traditional"]["groups"], spec["seed"])}
                   for label, result in results.items()}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Complete: {summary['played_hands']} hands; logs audited", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/dqn-three-seed-20261104")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/robustness-development.json")
    args = parser.parse_args()
    run(args.output, args.source, json.loads(args.config.read_text()))
