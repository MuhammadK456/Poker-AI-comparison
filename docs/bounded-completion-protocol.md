# Bounded robustness and equity ablation

Frozen November 4, 2026, before new training or robustness results. This phase
contains exactly one ablation and three additional opponent configurations.
The reference models and training budget remained fixed.

## One ablation

Retrain seeds 101/202/303 from scratch with the explicit equity feature held at
zero. Keep all 163 inputs and network initialization, architecture, optimizer,
opponent mixture, 250,000-decision budget and validation schedule identical to
the reference study. Raw cards, pot odds and history remain available, so the
experiment removes a supplied estimate, not all information about hand strength.
Opponent policies retain their original equity calculations. Save the feature
setting in checkpoints and reject mismatched inference encoders.

Evaluate the three final ablation models on the existing 2,000-pair development
panel (seed 2026110402). Compare each to its matching original training seed on
the same deal groups. Report all seeds, no model selection. These comparisons
are exploratory development evidence, not final tests or independent estimates
of training-seed uncertainty. Training trajectories diverge after actions differ;
matching seeds does not force equal experience or hands.

## Limited robustness check

Use exactly the three frozen settings in `configs/robustness-development.json`:
a tighter threshold opponent, a looser threshold opponent, and the existing
traditional rules with a 15% eligible checked-to bluff probability. These settings
were not training opponents. They are variations on existing rule families,
not a claim of generalization to new algorithms or human players.

Evaluate original DQN seeds and the frozen traditional baseline, 2,000 paired
deals per matchup, seed 2026110403: 48,000 hands. Do not extend the matrix with
ablation models; the ablation already has the original five-opponent panel.
This keeps the two questions separate and bounds compute. Report paired
differences and per-seed intervals, not a pooled winner. No retuning follows
unfavorable outcomes.

## Results

See the [combined report](bounded-experiments-20261104/README.md) and
[interpretation](bounded-experiments-20261104/findings.md).
No retuning or additional training followed the results.
