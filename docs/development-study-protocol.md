# Larger development study: frozen protocol

Prepared November 4, 2026, before examining the larger-run results.

## Pilot behavior inspection

The frozen 25,004-decision pilot was replayed on the same validation decks.
The diagnostic reproduced all five recorded matchup returns exactly. Against
the traditional opponent, it took 715 decisions: 139 raises, 220 bets, 114
calls, 230 checks and 12 folds. Of its 359 bets/raises, 185 used an equity
estimate below 0.50. It committed the effective stack through aggression 88
times, including 22 decisions below 0.50 equity. On 19 decisions the selected
Q estimate exceeded the theoretical [-1,1] terminal-return bounds in magnitude.

These are descriptive counts, not counterfactual EV estimates. Weak-equity
aggression can be a profitable bluff, and Q-function approximation need not
stay within payoff bounds. The pattern motivates examining longer learning,
not claiming that every such action is a bug. No hand-authored all-in guard,
new feature, opponent model, reward change or optimizer retuning was added
to the learned policy as a result of this inspection.

The saved [pilot diagnostic](pilot-behavior-20261104.json) includes actions by
street and equity bin. Raw per-decision CSV data remain under
`artifacts/pilot-behavior-20261104/`.

## Frozen training design

- Three fresh runs with training seeds 101, 202 and 303.
- 250,000 learner decisions per run, finishing the final hand.
- Same 163 features, 163→128→128→6 network and equity feature as the pilot.
- Same 50,000 replay capacity, 5,000 warmup, update every four decisions,
  Adam 0.0003, Huber loss, gradient clip 10, target copy every 1,000 updates.
- Same 10% random / 20% passive / 35% tight / 35% loose-aggressive mixture.
- Epsilon reaches 0.05 after 100,000 decisions and stays there.
- Three independent processes with one PyTorch thread each. This is a
  throughput setting, not a change to the algorithm or opponents.
- Checkpoints and validation after each 25,000-decision boundary, at the end
  of the current hand. Training resumes with its existing replay and RNGs.

The pilot checkpoint is not resumed because it lacks replay/RNG state.
New runs restart from their seeded initial networks. The inference-only
checkpoint limitation still applies to files written by this study.

## Validation and selection

Every run and checkpoint uses 100 paired deals per opponent, seed 20261104
in the explicit validation namespace. The same five opponent families as the
pilot are measured, including the frozen traditional bot. This provides
like-for-like comparisons across training seeds, but the set is small and
reused. It is development validation, not a final test or held-out opponent test.

Keep the final fixed-budget checkpoint for all three seeds. Do not select the
best checkpoint, the best seed or a later training budget after seeing the
curves. No extra runs are triggered by an unfavorable result. Paired bootstrap
intervals describe deal uncertainty for a fixed model. Report per-seed returns
and their descriptive mean/SD/range separately; do not treat all seed-hand
combinations as independent algorithm replications.

Final-test evaluation, feature ablations and unseen opponent families are
later work. Their absence is explicit, not hidden behind this study's plots.

## Validation of the runner

All 64 tests passed before the three runs were launched. The added tests cover
diagnostics of deduplicated all-in actions, aggregation without pooling seeds,
shared validation-seed enforcement, and periodic validation noninterference.
In the latter, identical short training runs with and without intermediate
validation produced exactly equal model weights and training-hand logs.

The study records configuration, separate worker logs, per-seed source hashes,
training progress, validation hands, checkpoint files and aggregate results.
The static report shows every learning curve and every final seed result.

## Completed run

The frozen study completed all three seeds, with 750,008 actual learner
decisions in 29.4 minutes. No retuning or extra training was performed.
See the [complete report](three-seed-20261104/README.md) for per-seed outcomes,
confidence intervals, learning curves, runtime and final behavior diagnostics.
