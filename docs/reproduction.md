# Reproduction

Verified on November 4, 2026 using Python 3.11 on the project's Apple Silicon Mac.
The lock file pins runtime dependencies and setuptools; pip itself is an install
tool, not a pinned model dependency. Other operating systems/Python versions
were not tested. The installed NumPy/PyTorch wheels require a compatible platform.

## Fresh environment check

A new isolated environment was created under `artifacts/reproduction-venv` and
installed from `requirements-lock.txt`, with no use of the existing environment's
site-packages. `pip check` passed. All 75 tests passed in 11.8 seconds during the
final run. The saved-model four-hand demo reproduced byte-identical JSON, and the
final report was rebuilt successfully in that environment.

The release candidate was then extracted into a different folder. All 75 tests
passed there in 9.4 seconds using the fresh interpreter. The relocated demo
matched every decision/payoff and model hash (its recorded checkpoint path
naturally changed). Rebuilt final numerical tables and PNG figures were
byte-identical to the originals. The archive and every manifest digest passed
verification. The final bundle differs from that candidate only in completion
documentation and the verification record; the executable source/models are unchanged.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m poker_ai.demo
```

Run these from the extracted project root. The demo needs the included original
checkpoint; it uses a separate demo seed and does not consume final-test deals.

## Bundle contents

The full model/results bundle contains:

- Source, tests, configuration snapshots, pinned requirements and reports/figures.
- All three original final checkpoints and all three no-equity final checkpoints.
- Training summaries, progress and learning curves for both three-seed studies.
- Canonical per-hand CSVs, grouped returns, summaries and configuration/hashes
  for the larger development evaluation, ablation evaluation, robustness panel
  and 400,000-hand final test.
- `MANIFEST.sha256.json`, mapping each bundled file to its SHA-256 digest.

It excludes virtual environments, caches, duplicated per-task final-test logs,
intermediate/initial checkpoints, full training-hand logs, historical research
handoff notes, and the first pilot checkpoint. Those omissions do not prevent
running the demo, auditing/reporting the included studies, replaying the final
test, or retraining. The complete original workspace retains its other artifacts.
Historical reports describe their original runs; not every early-stage artifact
is in the bundle. Exported metadata uses project-relative paths.

The Git repository contains source, configurations, tests, and reports. Model
checkpoints and raw hand logs are excluded by `.gitignore`; a source-only clone
must restore the models or train new ones as described in the root README.
The packaging command builds a local archive and does not publish a release.

## Audit and regenerate without replaying 400,000 hands

```sh
.venv/bin/python -m poker_ai.final_report --source artifacts/final-evaluation-20261104 --output docs/final-report-rebuilt
```

This validates complete paired records, deck matching across policies, and exact
agreement between raw hands and grouped returns before generating figures and
tables. Use a new output directory.

To replay the original frozen final experiment, follow the root README. It takes
about 32 minutes on the measured machine, with three workers; runtime can vary.
Replaying that seed is verification, not a new independent experiment. Do not
use its outcomes for further tuning while calling it a held-out test.

## Build and verify a bundle

```sh
.venv/bin/python -m poker_ai.release --output artifacts/release/poker-ai-v1.zip
```

The packager refuses existing outputs and writes a sibling `.sha256` file. After
extracting, verify the manifest from the extracted project root:

```python
import hashlib, json
from pathlib import Path

manifest = json.loads(Path("MANIFEST.sha256.json").read_text())
for name, expected in manifest.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
print(f"Verified {len(manifest)} files")
```

Release packaging does not modify or select models. Exact training continuation
is not supported: checkpoint files omit replay/RNG state. Reproduction training
starts from the recorded seeds and configurations.
