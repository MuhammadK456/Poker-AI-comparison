"""Build a local, checksummed source/results/model bundle; never publishes it."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def public_bytes(path):
    data = path.read_bytes()
    if path.suffix in (".py", ".md", ".json", ".csv", ".txt", ".svg"):
        data = data.replace((str(ROOT) + "/").encode(), b"")
    return data


def build(output):
    if output.exists():
        raise FileExistsError(output)
    selected = {ROOT / name for name in ("README.md", ".gitignore", "requirements.txt", "requirements-lock.txt")}
    for folder in ("poker_ai", "tests", "configs", "docs"):
        selected.update(path for path in (ROOT / folder).rglob("*") if path.is_file()
                        and "__pycache__" not in path.parts and not path.name.startswith(".")
                        and path.name != "project-handoff.md")
    for study in ("dqn-three-seed-20261104", "dqn-no-equity-20261104"):
        base = ROOT / "artifacts" / study
        selected.update(base / name for name in ("study-config.json", "study-summary.json", "final-matchups.csv"))
        for seed in (101, 202, 303):
            selected.update(base / f"seed-{seed}" / name for name in
                            ("final.pt", "summary.json", "run-config.json", "learning-curve.jsonl", "training-progress.csv"))
    for experiment in ("development-variance-20261104", "no-equity-evaluation-20261104",
                       "robustness-development-20261104", "final-evaluation-20261104"):
        base = ROOT / "artifacts" / experiment
        selected.update(base / name for name in ("summary.json", "run-config.json"))
        for label in ("seed-101", "seed-202", "seed-303", "traditional"):
            selected.update(base / name for name in (f"validation-{label}.csv", f"groups-{label}.json"))
    missing = [str(path.relative_to(ROOT)) for path in selected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"release inputs missing: {missing}")
    manifest = {str(path.relative_to(ROOT)): hashlib.sha256(public_bytes(path)).hexdigest() for path in sorted(selected)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(selected):
            archive.writestr("poker-ai-v1/" + str(path.relative_to(ROOT)), public_bytes(path))
        archive.writestr("poker-ai-v1/MANIFEST.sha256.json", json.dumps(manifest, indent=2) + "\n")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{checksum}  {output.name}\n")
    print(json.dumps({"archive": str(output), "files": len(manifest), "bytes": output.stat().st_size, "sha256": checksum}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    build(parser.parse_args().output)
