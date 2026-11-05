# Interpretation of ablation and robustness results

Completed November 4, 2026. No new algorithms, game variants, parameter search
or longer training budget were introduced. Existing reference models are unchanged.

## Effect of explicit equity

At 250,000 learner decisions, the no-equity variant had lower point estimates
in all 15 seed/opponent comparisons. Eleven pointwise paired 95% intervals
exclude zero; four remain inconclusive. These are exploratory comparisons,
not multiplicity-adjusted claims or 15 independent training replications.

The passive-opponent loss is the clearest repeated effect: removing equity
reduces returns by approximately 853, 490 and 624 bb/100 for seeds 101/202/303,
with all three paired intervals below zero. Direct performance against the
traditional bot also declines for each seed, though seed 303's interval is
close to zero at its upper end.

This shows the value of a supplied probability estimate within our fixed
architecture, opponents and compute budget. It does not establish that raw-card
learning is impossible or that equity is sufficient for strong poker play.
No-equity inputs retain cards and pot odds, and opposing agents retain equity.

The ablation completed 750,001 decisions in 12.7 minutes, with approximately
296–298 MB peak process memory per worker. Original training took 29.4 minutes,
but this is not an isolated equity-computation speed benchmark: trajectories,
hand counts, validation cost and runtime conditions differ.

## Robustness remains limited

The tighter and looser threshold settings and 15% eligible-bluff rule variant
were not training opponents. They reuse existing rule families; we make no
claim about new algorithms or human players.

Against the looser setting, all three original ML models underperform the
traditional baseline by approximately 318–409 bb/100, with negative paired
intervals. This reinforces the earlier weak exploitation of loose/aggressive
opponents. Tighter-opponent and bluff-variant comparisons are mostly inconclusive;
seed 303's positive tighter-opponent comparison is one exploratory exception.
There is no broad robustness win to claim, and no retuning follows these results.

## Checks and artifacts

- All 72 tests passed, including no-equity isolation, checkpoint-setting
  persistence, mismatch rejection, matched-deal analysis and deterministic variants.
- Original checkpoint SHA-256 hashes are unchanged. An original-checkpoint
  replay regression reproduces the historical paired returns on 20 sampled hands.
- All 128,000 new evaluation hands passed completeness, shared-deck/seat-pair,
  and exact-return reconstruction audits. The repeated traditional reference
  exactly reproduces its earlier per-deal outcomes.
- Full logs/checkpoints remain in `artifacts/dqn-no-equity-20261104`,
  `artifacts/no-equity-evaluation-20261104`, and
  `artifacts/robustness-development-20261104`. Durable summaries, configs and
  training curves accompany the combined report.
