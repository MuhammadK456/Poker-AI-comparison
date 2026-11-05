"""Run a fixed set of independent training seeds in bounded worker processes."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from .dqn import DQNConfig
from .train import OPPONENT_MIXTURE, ROOT


def validate_spec(spec):
    if type(spec.get("include_equity", True)) is not bool:
        raise ValueError("include_equity must be boolean")
    seeds = spec["seeds"]
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty, distinct integers")
    if type(spec["workers"]) is not int or not 1 <= spec["workers"] <= 3:
        raise ValueError("use between one and three workers")
    if spec["opponent_mixture"] != OPPONENT_MIXTURE:
        raise ValueError("unsupported opponent mixture")
    config = DQNConfig(**spec["dqn"])
    if spec["requested_decisions_per_seed"] < config.warmup:
        raise ValueError("budget must reach warmup")
    if spec["validation_pairs"] < 2 or spec["validation_every"] <= 0:
        raise ValueError("invalid validation budget")
    if not 1 <= spec["threads_per_worker"] <= 4:
        raise ValueError("invalid thread count")


def aggregate(output: Path, seeds: list[int]) -> dict:
    runs = {str(seed): json.loads((output / f"seed-{seed}" / "summary.json").read_text()) for seed in seeds}
    validation_seeds = {run["validation_seed"] for run in runs.values()}
    if len(validation_seeds) != 1:
        raise ValueError("seed comparisons need shared validation deals")
    results = {}
    for opponent in next(iter(runs.values()))["validation"]["final"]:
        returns = [run["validation"]["final"][opponent]["bb_per_100"] for run in runs.values()]
        results[opponent] = {"per_seed_bb_per_100": dict(zip(map(str, seeds), returns)),
                             "mean": statistics.mean(returns),
                             "sample_sd": statistics.stdev(returns) if len(returns) > 1 else None,
                             "min": min(returns), "max": max(returns)}
    with (output / "final-matchups.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", "opponent", "bb_per_100", "ci_low", "ci_high",
                                                   "difference_vs_traditional", "difference_ci_low", "difference_ci_high"])
        writer.writeheader()
        for seed, run in runs.items():
            for opponent, row in run["validation"]["final"].items():
                writer.writerow(dict(seed=seed, opponent=opponent, bb_per_100=row["bb_per_100"],
                                     ci_low=row["ci95"][0], ci_high=row["ci95"][1],
                                     difference_vs_traditional=row["difference_vs_traditional"],
                                     difference_ci_low=row["difference_ci95"][0], difference_ci_high=row["difference_ci95"][1]))
    return {"seeds": seeds, "validation_seed": next(iter(validation_seeds)),
            "total_actual_decisions": sum(run["actual_decisions"] for run in runs.values()),
            "training_seconds_per_seed": {seed: run["training_seconds"] for seed, run in runs.items()},
            "peak_rss_bytes_per_seed": {seed: run["peak_process_rss_bytes"] for seed, run in runs.items()},
            "results": results,
            "limits": "Three development runs, fixed final checkpoints. Per-seed hand intervals do not measure "
                      "training-seed uncertainty; cross-seed SD/range are descriptive with only three seeds. "
                      "Shared validation deals were reused along learning curves. No final-test evaluation or best-seed selection."}


def run_study(output: Path, spec: dict) -> dict:
    validate_spec(spec)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "study-config.json").write_text(json.dumps(spec, indent=2) + "\n")
    pending = list(spec["seeds"])
    active, completed, handles = {}, {}, []
    start = time.perf_counter()
    last_report = -30.0
    try:
        while pending or active:
            while pending and len(active) < spec["workers"]:
                seed = pending.pop(0)
                config = {"seed": seed, "requested_decisions": spec["requested_decisions_per_seed"],
                          "validation_pairs": spec["validation_pairs"], "validation_seed": spec["validation_seed"],
                          "validation_every": spec["validation_every"], "threads": spec["threads_per_worker"],
                          "dqn": spec["dqn"], "opponent_mixture": spec["opponent_mixture"]}
                config["include_equity"] = spec.get("include_equity", True)
                config_path = output / f"seed-{seed}-config.json"
                config_path.write_text(json.dumps(config, indent=2) + "\n")
                log = (output / f"seed-{seed}.log").open("w")
                handles.append(log)
                active[seed] = subprocess.Popen([sys.executable, "-m", "poker_ai.train", "--config", str(config_path),
                                                 "--output", str(output / f"seed-{seed}")],
                                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            for seed, process in list(active.items()):
                code = process.poll()
                if code is not None:
                    completed[seed] = code
                    del active[seed]
            elapsed = time.perf_counter() - start
            if elapsed - last_report >= 30 or not active:
                status = {}
                for seed in spec["seeds"]:
                    progress = output / f"seed-{seed}" / "training-progress.csv"
                    rows = []
                    if progress.exists():
                        with progress.open() as stream:
                            rows = list(csv.DictReader(stream))
                    status[str(seed)] = {"state": "complete" if completed.get(seed) == 0 else
                                        "failed" if seed in completed else "running" if seed in active else "pending",
                                        "decisions": int(rows[-1]["decisions"]) if rows else 0}
                (output / "status.json").write_text(json.dumps({"elapsed_seconds": elapsed, "runs": status}, indent=2) + "\n")
                print(json.dumps({"elapsed_seconds": round(elapsed), "runs": status}), flush=True)
                last_report = elapsed
            if active:
                time.sleep(1)
        failed = {seed: code for seed, code in completed.items() if code != 0}
        if failed:
            raise RuntimeError(f"training workers failed; inspect their logs: {failed}")
        summary = aggregate(output, spec["seeds"])
        summary["wall_seconds"] = time.perf_counter() - start
        (output / "study-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    finally:
        for process in active.values():
            process.terminate()
        for process in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "dqn-three-seed.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_study(args.output, json.loads(args.config.read_text())), indent=2))


if __name__ == "__main__":
    main()
