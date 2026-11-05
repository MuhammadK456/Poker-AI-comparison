# Bounded equity ablation and opponent robustness

**Later completion:** version one has finished its [final held-out evaluation](../final-evaluation-20261104/README.md)
and [reproduction checks](../reproduction.md). This report preserves the development-stage experiment.

Exactly one feature ablation, three seeds, and three opponent-parameter variations. All checkpoints are fixed final 250k-decision models. No retuning or final testing.

See [interpretation](findings.md) for findings and validation.

## No-equity ablation

The supplied equity input is held at zero during training and inference. Raw cards, pot odds, history, architecture and training budget are retained. Opponents still use their original equity calculations.

The evaluation uses the same 2,000 paired development deals per opponent as the reference models. Values below are no-equity minus with-equity returns, in bb/100, with whole-pair bootstrap 95% intervals. Negative differences favor retaining the explicit equity feature.

| Opponent | Seed 101 | Seed 202 | Seed 303 |
|---|---:|---:|---:|
| random | -188.4 [-341.5, -37.2] | -335.0 [-477.5, -185.3] | -87.8 [-220.0, +45.9] |
| passive | -853.3 [-1023.1, -685.0] | -489.9 [-655.3, -314.4] | -624.5 [-787.1, -461.9] |
| tight | -33.3 [-135.4, +65.5] | -129.7 [-242.5, -13.0] | -40.4 [-150.9, +57.9] |
| loose_aggressive | -191.9 [-310.5, -77.2] | -296.4 [-411.7, -175.1] | -97.1 [-222.1, +25.0] |
| traditional | -256.6 [-365.6, -150.1] | -162.4 [-274.9, -47.3] | -121.9 [-229.6, -3.0] |

Absolute returns and paired differences are saved in `ablation-differences.json`. Matching training seeds do not force identical training trajectories. These intervals reflect deal uncertainty for fixed model pairs, not uncertainty over training runs or universal feature importance.

## Opponent-configuration robustness

The original models and traditional baseline face two shifted threshold settings and a 15% eligible checked-to bluff variant of the existing traditional rules. These configurations were excluded from training; they are not new algorithm families or human opponents. The ablation models are not added to this panel.

Return in bb/100 (95% whole-pair bootstrap interval), 2,000 paired deals per matchup:

| Opponent | Seed 101 | Seed 202 | Seed 303 | Traditional |
|---|---:|---:|---:|---:|
| tighter | -15.6 [-76.0, +43.5] | +6.2 [-56.4, +62.3] | +20.2 [-34.5, +73.1] | -44.3 [-78.0, -10.5] |
| looser | -3.8 [-95.5, +82.4] | +86.9 [-6.8, +178.7] | +40.0 [-49.5, +127.0] | +404.7 [+349.6, +460.7] |
| bluffing_rules | +48.8 [-34.9, +130.8] | -8.0 [-91.9, +75.5] | +21.8 [-61.4, +104.4] | +3.1 [-0.8, +8.1] |

Paired improvements over the traditional baseline:

| Opponent | Seed 101 | Seed 202 | Seed 303 |
|---|---:|---:|---:|
| tighter | +28.7 [-29.2, +87.8] | +50.5 [-7.7, +109.6] | +64.5 [+8.8, +122.0] |
| looser | -408.5 [-506.2, -308.9] | -317.8 [-414.4, -221.3] | -364.7 [-466.1, -263.7] |
| bluffing_rules | +45.6 [-40.6, +128.9] | -11.1 [-94.7, +70.3] | +18.7 [-63.9, +99.0] |

## Cost, checks and limits

Ablation training: 750,001 decisions, 12.7 minutes wall time. Robustness: 48,000 hands, 5.1 minutes. Ablation evaluation adds 80,000 hands including a repeated traditional reference used to check unchanged behavior.

Hand logs were audited for complete seat pairs, shared decks and exact return reconstruction. The original and ablation panels share actual decks and reproduce identical traditional-baseline returns. All intervals are pointwise, exploratory and unadjusted for multiple comparisons. Three training seeds provide limited replication. No best seed/checkpoint was selected. Final-test data remain untouched.

## Scope boundary

Development expansion stops here. Remaining required work is final evaluation and presentation/reproduction. The position ablation, new opponent family, opponent modeling, betting rollouts, self-play and game variants are outside required version-one scope. See `../bounded-completion-protocol.md`.

## Reproduction

```sh
.venv/bin/python -m poker_ai.study --config configs/dqn-no-equity.json --output artifacts/no-equity-new
.venv/bin/python -m poker_ai.evaluate --source artifacts/no-equity-new --output artifacts/no-equity-eval-new
.venv/bin/python -m poker_ai.robustness --output artifacts/robustness-new
.venv/bin/python -m poker_ai.ablation_report --reference artifacts/development-variance-20261104 --ablation artifacts/no-equity-eval-new --robustness artifacts/robustness-new --training artifacts/no-equity-new --output docs/bounded-experiments-new
```

Use fresh output directories. Saved no-equity checkpoints load with `DoubleDQN.load_for_inference`; construct their encoder with `FeatureEncoder(calculator, include_equity=model.include_equity)`. The agent rejects mismatched encoder settings. Existing checkpoints default to equity enabled. Checkpoints remain inference snapshots, not exact training-resume files.
