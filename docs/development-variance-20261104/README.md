# Larger development evaluation

Frozen models; 2,000 paired deals per matchup; 80,000 played hands. Evaluation wall time: 8.2 minutes.

Fresh development deals shared across all policies. No retraining, checkpoint selection or final test.

All hand logs were checked for complete seat pairs, identical decks across policies, and exact agreement with saved paired returns.

## Return (bb/100; paired-bootstrap 95% interval)

| Opponent | Seed 101 | Seed 202 | Seed 303 | Traditional |
|---|---:|---:|---:|---:|
| random | +106.8 [+7.0, +211.2] | +201.6 [+87.7, +319.2] | +138.0 [+33.4, +243.3] | +1054.4 [+943.3, +1177.8] |
| passive | +1254.8 [+1131.6, +1374.3] | +1009.1 [+882.0, +1128.9] | +1010.0 [+891.6, +1130.1] | +242.8 [+221.6, +265.1] |
| tight | -30.5 [-110.5, +50.8] | -20.8 [-103.8, +56.9] | -44.4 [-122.3, +33.9] | -49.6 [-87.9, -12.8] |
| loose_aggressive | -13.2 [-99.1, +70.1] | -10.9 [-95.8, +78.8] | -42.8 [-126.6, +43.0] | +220.2 [+176.9, +262.9] |
| traditional | +21.6 [-63.4, +103.8] | -27.1 [-115.1, +57.0] | -80.1 [-161.6, -0.2] | +0.0 [+0.0, +0.0] |

## Paired improvement over the traditional baseline

Both policies face the same opponent on the same paired deals. These differences answer a different question from direct head-to-head returns.

| Opponent | Seed 101 | Seed 202 | Seed 303 |
|---|---:|---:|---:|
| random | -947.5 [-1086.0, -804.2] | -852.8 [-994.7, -715.8] | -916.4 [-1051.5, -776.2] |
| passive | +1012.0 [+890.2, +1129.7] | +766.4 [+650.2, +888.8] | +767.2 [+653.6, +887.3] |
| tight | +19.1 [-62.3, +102.6] | +28.8 [-56.4, +113.8] | +5.2 [-78.4, +89.5] |
| loose_aggressive | -233.4 [-331.1, -139.2] | -231.1 [-322.2, -140.9] | -262.9 [-358.1, -170.3] |
| traditional | +21.6 [-65.2, +107.6] | -27.1 [-111.8, +58.1] | -80.1 [-161.9, +0.6] |

## Precision planning

Normal approximation: required pairs = ceil((1.96 × paired-outcome SD / target half-width)²). This estimates interval width for a fixed policy; it is not statistical power, a guarantee, or a training-seed interval. It assumes the future matchup has similar variance. New opponent families need their own development variance estimates.

| Seed | Direct vs traditional: half-width at 10k pairs | Pairs for ±100 bb/100 | Worst paired-difference requirement across current opponents |
|---|---:|---:|---:|
| seed-101 | 37.9 | 1,437 | 3,872 |
| seed-202 | 37.1 | 1,374 | 3,978 |
| seed-303 | 36.4 | 1,325 | 4,104 |

For the existing matchup panel, retain at least **10,000 pairs per matchup** for the ±100 planning target, rounded up to 1,000 and respecting the original 10,000-pair starting budget. This is a recommendation only; no final evaluation has been launched.

Linear same-panel runtime projection at 10,000 pairs: 40.9 minutes, excluding additional opponents or ablations and allowing for hardware/load variation.

The JSON includes alternative ±50/±200 targets. All intervals are pointwise, with no multiple-comparison correction. Whole deal pairs are resampled; training seeds are reported separately. Repeated development inspection prevents treating these results as final-test claims.

Absolute-return and paired-difference intervals use separate bootstrap random streams. Against the deterministic traditional self-play reference, the underlying samples coincide, but finite resampling can put endpoints on opposite sides of zero. Treat a boundary result as fragile; do not choose whichever interval gives a preferred conclusion.

See [interpretation and next steps](findings.md) for the development findings.

## Seat breakdown

Descriptive returns by physical seat are in `seat-results.csv`; seat 0 is BB and seat 1 is SB/button. Their correlated hands must not replace whole-deal groups in uncertainty calculations.

## Reproduce

```sh
.venv/bin/python -m poker_ai.evaluate --output artifacts/development-variance-new
.venv/bin/python -m poker_ai.evaluation_report --source artifacts/development-variance-new --output docs/development-variance-new
```
