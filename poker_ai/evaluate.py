"""Frozen-model development evaluation and approximate precision planning."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch

from .dqn import DoubleDQN
from .equity import DEFAULT_TABLE, EquityConfig
from .traditional import LOOSE_CONFIG, TIGHT_CONFIG, TraditionalConfig
from .train import ROOT, validate, summarize_validation


def precision_plan(values, half_widths=(50, 100, 200)):
    """Normal-approximation CI width planning; not a power/significance guarantee."""
    if len(values) < 2:
        raise ValueError("need independent paired-deal values")
    sd = statistics.stdev(values)
    return {
        "pair_sd_bb_per_100": sd,
        "approx_half_width_at_10000_pairs": 1.96 * sd / math.sqrt(10000),
        "pairs_for_half_width": {
            str(width): max(2, math.ceil((1.96 * sd / width) ** 2))
            for width in half_widths
        },
    }


def worker(output, label, checkpoint, pairs, seed, snapshot):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    model = DoubleDQN.load_for_inference(Path(checkpoint)) if checkpoint else None
    values = validate(Path(output), label, model, pairs, seed, TraditionalConfig(**snapshot["config"]))
    result = {"groups": values, "seconds": time.perf_counter() - start}
    (Path(output) / f"groups-{label}.json").write_text(json.dumps(result) + "\n")
    return result


def run(output, source, pairs=2000, seed=2026110402):
    if pairs < 2:
        raise ValueError("need at least two pairs")
    checkpoints = {f"seed-{s}": source / f"seed-{s}" / "final.pt" for s in (101, 202, 303)}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in checkpoints.items()}
    snapshot = json.loads((ROOT / "configs/traditional-v1.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "purpose": "Development variance pilot; no final-test deals, training or model selection",
        "pairs_per_matchup": pairs, "validation_seed": seed,
        "deal_namespace": "dqn_validation_deal", "workers": 3, "threads_per_worker": 1,
        "checkpoints": {name: str(path.resolve()) for name, path in checkpoints.items()},
        "checkpoint_sha256": hashes, "traditional_snapshot": snapshot,
        "equity": asdict(EquityConfig()), "preflop_sha256": hashlib.sha256(DEFAULT_TABLE.read_bytes()).hexdigest(),
        "threshold_opponents": {"tight": asdict(TIGHT_CONFIG), "loose_aggressive": asdict(LOOSE_CONFIG)},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
        "planning": "Normal-approximation 95% CI half-widths 50/100/200 bb/100; primary planning target 100. Not power analysis. Initial final budget 10000 pairs; no automatic final test.",
    }
    (output / "run-config.json").write_text(json.dumps(config, indent=2) + "\n")
    start = time.perf_counter()
    results = {}
    jobs = {"traditional": None, **checkpoints}
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(worker, str(output), label, str(path) if path else None, pairs, seed, snapshot): label
                   for label, path in jobs.items()}
        for future in as_completed(futures):
            label = futures[future]
            results[label] = future.result()
            print(f"Completed {label}: {results[label]['seconds']:.1f}s", flush=True)
    reference = results["traditional"]["groups"]
    summary = {"pairs_per_matchup": pairs, "played_hands": pairs * 2 * 5 * 4,
               "wall_seconds": time.perf_counter() - start, "results": {}}
    for label, result in results.items():
        matchups = summarize_validation(result["groups"], reference, seed)
        for name, entry in matchups.items():
            values = result["groups"][name]
            differences = [a-b for a, b in zip(values, reference[name], strict=True)]
            entry["return_precision"] = precision_plan(values)
            entry["difference_precision"] = precision_plan(differences)
        summary["results"][label] = {"evaluation_seconds": result["seconds"], "matchups": matchups}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Complete: {summary['played_hands']} hands, {summary['wall_seconds']:.1f}s", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/dqn-three-seed-20261104")
    parser.add_argument("--pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026110402)
    args = parser.parse_args()
    run(args.output, args.source, args.pairs, args.seed)


if __name__ == "__main__":
    main()
