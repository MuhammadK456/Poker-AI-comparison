# Frozen final evaluation protocol

Frozen November 4, 2026 before any final-test games. Core development is closed.

- Original equity-enabled DQN final 250k checkpoints, seeds 101/202/303; frozen
  no-bluff traditional baseline. No training, selection, or rule changes.
- Five existing opponent settings: random, passive, tight, loose/aggressive,
  traditional. New final deals, not unseen opponent-family generalization.
  Robustness and no-equity analyses remain explicitly exploratory development
  results; no new model/opponent matrix is added.
- 10,000 independent deal pairs per policy/opponent: 400,000 hands total.
  Seed 2026110404, namespace `poker_final_test_v1`. Same actual decks across
  policies; seat swaps keep physical-seat cards fixed. Dedicated policy RNGs.
- Primary outcomes: each original seed's direct return against traditional,
  bb/100. Report all three with 95% intervals and Bonferroni-adjusted 98.333%
  intervals for a 95% familywise target across the three primary comparisons.
  These are bootstrap approximations, not exact finite-sample guarantees.
  Use 10,000 whole-pair bootstrap resamples for primary comparisons.
- Secondary outcomes: all opponent returns and paired improvements over the
  traditional baseline, with pointwise 95% intervals from 2,000 resamples.
  Secondary comparisons are descriptive, without multiplicity adjustment.
- Report position returns, action frequencies, individual-hand return spread,
  decision latency (including feature/equity calculation), runtime and resource
  context. Latency is observed under three concurrent workers, with a bounded
  equity cache, not an isolated hardware benchmark. Timing never changes actions.
- Three training seeds remain separate. Deal intervals do not measure training
  uncertainty. No best seed or checkpoint is selected. Zero-crossing intervals
  mean inconclusive, not equivalent policies.
- Complete the fixed budget regardless of intermediate outcomes. No extra
  training or evaluation follows an unfavorable result. Audit every log for
  complete paired deals and exact return reconstruction. Preserve model hashes.

Implementation smoke tests use only separately seeded development fixtures,
not final-test seeds. After this run, finish reporting, replay demo and a fresh
environment reproduction check. Do not add algorithms or experiments.

## Completed

The fixed 400,000-hand run completed in 32.1 minutes, and every paired log
passed the audit. See the [final report](final-evaluation-20261104/README.md).
No reliable direct winner was found. No additional training or sampling followed.
