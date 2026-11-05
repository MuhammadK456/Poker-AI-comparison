# Interpretation and next steps

The 2,000-pair development evaluation gives a clearer picture than the previous
100-pair validation. All three saved ML policies have positive paired improvements
over the traditional baseline against passive play. All three have negative paired
improvements against random and loose/aggressive opponents. Tight-opponent
differences remain inconclusive. These are pointwise development comparisons,
not final-test or multiple-comparison-adjusted claims.

Direct ML-versus-traditional returns are +21.6, -27.1 and -80.1 bb/100 for seeds
101/202/303. The first two intervals include zero. Seed 303 is a boundary case:
the absolute-return bootstrap upper endpoint is -0.2, while the separately
resampled difference interval upper endpoint is +0.6. Both describe the same
underlying values because traditional self-play paired returns are zero.
This finite-bootstrap sensitivity is a reason to avoid declaring a firm loss.

The results support a narrow interpretation: the learned policies exploit the
passive style well but do not exploit all simple opponents effectively. This
does not identify the cause as overfitting, missing features or insufficient
training. The smaller study's positive tight-opponent point estimates did not
persist on the new deals; this illustrates why the larger sample was needed.

Keep the original final evaluation starting budget of 10,000 paired deals.
For the current models and opponents, estimated direct-match 95% half-widths
at that size are about 36–38 bb/100, and the worst paired-comparison requirement
for a ±100 target is 4,104 pairs. The same-panel runtime projection is about
41 minutes. These are approximate precision/cost estimates, not power guarantees;
new opponent families may have different variances.

Next work remains opponent robustness and the no-equity feature ablation,
followed by a frozen final-test protocol. Do not extend training or pick a best
checkpoint based on this evaluation. Preserve the current agents as version one.
Keep final opponent settings/deals out of development tuning.

Validation: all 67 tests passed. All 80,000 hand records passed completeness,
seat-pair/deck-matching and exact paired-return reconstruction checks. No new
training occurred. Full raw logs remain in `artifacts/development-variance-20261104`.
