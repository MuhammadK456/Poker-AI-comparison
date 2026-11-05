# Development variance pilot

Prepared November 4, 2026, before inspecting this pilot's aggregate results.

Evaluate the existing fixed 250k-decision checkpoints for seeds 101/202/303
and the frozen no-bluff traditional baseline. No training, parameter changes,
best-seed selection or final-test evaluation is part of this milestone.

- Use 2,000 independent paired deals per policy/opponent, 80,000 played hands
  across four policies and five existing opponents.
- Use validation seed 2026110402 in `dqn_validation_deal`, distinct from the
  smaller development study. Share actual decks and opponent randomness
  across policies; swap policy seats while keeping physical-seat cards fixed.
- Run three evaluation processes, one Torch thread each. Log every hand.
- Retain all model results separately, with 2,000-resample whole-pair bootstrap
  intervals for returns and differences from the traditional baseline.
- Estimate precision using paired-outcome sample SD. Show normal-approximation
  required sample sizes for 95% interval half-widths of 50, 100 and 200 bb/100.
  The primary planning target is 100 bb/100; retain the original 10,000-pair
  final starting budget unless a larger size is needed. Round upward to 1,000.
- This is interval-width planning, not a power analysis or a significance
  stopping rule. No extra games are triggered by favorable/unfavorable results.
- Record runtime and project same-panel evaluation cost. New opponent families
  and feature ablations remain separate work; these estimates cannot guarantee
  precision for unseen matchups.

The runner saves checkpoint/source/table hashes and policy configuration before
playing. Tests cover the precision formula and existing match/learning invariants.
The report also exports descriptive position results. Final-test data remain
untouched until the full opponent/ablation protocol is frozen.
