# ML pipeline and first Double DQN pilot

**Later milestone:** the [three-seed development study](three-seed-20261104/README.md)
has now completed 250,000 decisions per seed. This document preserves the
original pilot results and explains the shared pipeline.

Implemented November 4, 2026. The initial implementation uses PyTorch 2.14.1
and NumPy 2.4.6 on CPU, with two PyTorch threads. This is reinforcement learning
from simulated decisions and chip returns; it does not use human-game labels.

## Input: 163 visible values

`FeatureEncoder` produces a fresh float32 array. Version: `visible-163-v1`.
Own/opponent features and history counts are relative to the acting player.

The bounded no-equity ablation retains the same 163-value layout and sets index
117 to constant zero, without calling the learner's equity calculator. Other
features and opponent equity calculations remain unchanged. Checkpoints record
`include_equity`; old checkpoints default to `true`. For either model type use
`FeatureEncoder(calculator, include_equity=model.include_equity)` when constructing
an inference agent. Mismatched encoder/model settings raise an error. See the
[bounded protocol](bounded-completion-protocol.md) for scope and comparison rules.

| Zero-based indices | Contents |
|---|---|
| 0–51 | Own hole-card multi-hot vector |
| 52–103 | Revealed board-card multi-hot vector |
| 104–107 | Street one-hot |
| 108 | SB/button flag |
| 109–115 | Pot, call, own/opponent stacks, own/opponent street bets, effective remaining stack |
| 116 | Pot odds, excluding any uncallable excess |
| 117 | Shared showdown equity estimate |
| 118–123 | Last action: none/fold/check/call/bet/raise |
| 124 | Last action's chips added |
| 125–156 | Counts by street × relative player × check/call/bet-or-raise/fold |
| 157–162 | Six legal-action mask values |

Chip amounts are divided by the 200-chip starting stack. History counts remain
raw counts; probabilities stay in [0,1]. Card indices follow `FULL_DECK` in the
engine module. Hidden cards, future deck, deal seeds, opponent identities and
evaluation labels are absent. The feature vector does not encode full action
order and is not asserted to be a sufficient belief state for poker.

## A learning transition

The `LearnerEpisode` wrapper holds the engine and the opposing policy. It returns
observations only when the learner acts. After a learner action it plays all
intervening opponent actions, including a possible first action on the next
street, until the learner acts again or the hand ends.

Stored transition:

`own features, own legal mask, chosen slot, reward, next own features, next mask, terminal`

Nonterminal reward is zero. Terminal reward is the learner's net chip payoff
divided by its 200-chip starting stack, including blinds. Folding the small
blind therefore returns -0.005, not zero. Terminal next features and masks are
all zeros and never bootstrap. If an opponent folds before the learner makes
any decision, the result is logged but no fabricated action/transition is added.

The runner alternates learner seats, resets stacks each hand, and finishes the
last hand when the decision budget is reached. This avoids treating a training
budget cutoff as a poker terminal. Equity is computed when observations are
encoded and stored in replay, not recomputed for each optimizer update.

## Replay and updates

The replay buffer is a fixed-capacity ring of dense NumPy arrays. At capacity
50,000 it allocates **66,450,000 bytes** (~63.37 MiB), including both feature
arrays, masks, actions, rewards and terminal flags. It copies transitions on
insertion and samples uniformly without replacement within each minibatch.
Validation checks reject illegal actions, mismatched masks and malformed
terminal transitions.

The network is `163 → 128 → 128 → 6`, with ReLU hidden layers and 38,278
parameters. Adam uses learning rate 0.0003 and Huber loss. Batch size is 128;
updates begin after 5,000 learner decisions and run every four decisions.
The target network is copied from the online network every 1,000 updates.
Gradients are clipped to norm 10 as a numerical safeguard, and nonfinite losses
or gradients fail explicitly. Discount is 1 for finite hands.

Double DQN chooses the best **legal** next action with the online network and
evaluates that action with the target network. This is different from taking
the target network's own maximum. Terminal rows are excluded entirely from
next-state evaluation, avoiding invalid all-false-mask maxima and `0 × infinity`.

Exploration chooses uniformly among legal slots. Epsilon decays linearly from
1 to 0.05 over 100,000 learner decisions. Greedy selection also masks illegal
actions. The pilot therefore still explores heavily at 25,000 decisions;
greedy validation always uses epsilon=0 and no learning.

## Opponents and experiment separation

Each training hand independently selects random/passive/tight/loose-aggressive
opponents with probabilities 10%/20%/35%/35%. Threshold parameters are the
previous frozen development presets. The traditional agent is not a training
opponent in this first mixture. Neither the opponent identity nor its policy
parameters enter the learned observation.

Separate deterministic RNG streams govern deals, exploration, opponent choice,
opponent policy actions and replay sampling. Validation has its own deal
namespace. Initial and final validation use identical decks and swapped seats,
also matched to the frozen traditional baseline. The checkpoints are fixed
before the run; there is no search for a best-looking checkpoint.

The pilot uses 100 independent paired deals per opponent (200 hands), against
the four training families plus the traditional agent. This is a small
validation set, not a final test or a test of unseen opponent families. Return
and difference intervals resample whole paired deals using 2,000 bootstrap
replicates. Reusing validation deals for initial/final comparisons is deliberate.
Final-test data and multi-seed learning uncertainty remain separate future work.

## Running and loading

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m poker_ai.train --output artifacts/dqn-pilot-new
```

Use a new output directory. `configs/dqn-pilot.json` freezes the default seed,
budget and optimizer settings. `run-config.json` records actual settings,
dependency versions, feature version, opponent presets and source/table hashes.
The hand and progress CSVs separate completed-hand outcomes from learner-step
counts. The small pilot may exceed its requested decision count slightly to
finish the last hand; the summary reports both counts.

```python
from pathlib import Path
from poker_ai.dqn import DoubleDQN, DQNAgent
from poker_ai.features import FeatureEncoder
from poker_ai.equity import EquityCalculator

model = DoubleDQN.load_for_inference(Path("artifacts/dqn-pilot-101-20261104/final.pt"))
agent = DQNAgent(model, FeatureEncoder(EquityCalculator()))
# slot = agent.act(observation)
```

Use the encoder/equity settings recorded with the checkpoint. The loader checks
the schema and feature version and uses PyTorch's `weights_only=True` loading.
Saved files contain online/target weights, optimizer state, counters and run
metadata. They intentionally omit replay and RNG states. The exposed loader is
for inference, not exact training continuation; optimizer state is retained
for inspection but not restored by that loader.

## Correctness checks

All **61 project tests** passed before the pilot. New tests verify exact feature
positions and relative-seat history, hidden-card noninterference, learner-turn
alignment, blind losses, opponent-fold returns, no-decision hands, bounded replay,
legal masks in exploration and greedy action choice, Double DQN selection versus
evaluation, terminal bootstrap suppression, target-copy cadence, synthetic
reward fitting, checkpoint round trips and frozen inference. A short end-to-end
pilot repeated with the same seed reproduced weights and hand logs exactly.

The synthetic fitting test only checks optimization plumbing. Playing strength
must be assessed separately. Peak RSS is the high-water mark for the entire
process, while measured training time excludes validation. Neither metric is
a guarantee for larger runs or different machines.

## Recorded seed-101 pilot

The completed run used **25,004 learner decisions** (four extra decisions to
finish the final hand) across **11,408 hands**, with **5,002 gradient updates**.
There were 1,086 hands ending before the learner acted and 10,322 terminal
transitions, exactly accounting for the remaining hands. The observed opponent
counts were random 1,107; passive 2,266; tight 4,013; loose/aggressive 4,022.

Training took **101.72 seconds**, about **245.8 learner decisions/second**.
Peak process RSS was **302,809,088 bytes** (~303 MB / 288.78 MiB), below the
project's 2 GB pilot target. Replay held 25,004 entries in its 50,000-entry
allocation. These are measured pilot figures, not extrapolated training promises.
Final exploration epsilon was 0.762462; validation used zero exploration.

Mean Huber loss over the first 100 updates was 0.07172, versus 0.05213 over the
last 100. This decline alone is not evidence of better poker play: replay and
bootstrapped targets change during training. The independently evaluated policy
results are more informative.

### Greedy validation returns (bb/100)

Each cell uses the same 100 paired deals per opponent, with no learning during
evaluation. Parentheses in the final column contain the paired 95% bootstrap
interval for the final policy's return.

| Opponent | Traditional baseline | Untrained DQN | Final pilot DQN (95% interval) |
|---|---:|---:|---:|
| Random | +872.25 | +304.75 | -106.75 (-669.02, +474.71) |
| Passive | +224.50 | -4.50 | +1,372.50 (+881.96, +1,910.60) |
| Tight | -9.25 | -1,066.75 | -737.75 (-1,221.33, -250.97) |
| Loose/aggressive | +151.00 | -2,049.00 | -820.50 (-1,298.50, -343.98) |
| Traditional | 0.00 | -1,966.50 | -1,335.00 (-1,765.60, -922.43) |

Paired differences support improvement over the untrained network against
passive, loose/aggressive and traditional opponents in this small validation
sample. Differences against random and tight opponents remain inconclusive.
The trained policy nevertheless loses substantially against the tight,
loose/aggressive and traditional policies, and underperforms the traditional
baseline in most matchups. Strong exploitation of passive play is not broad
poker competence. No parameter retuning or extra training was performed after
viewing these results.

The pilot passes pipeline correctness, finite-update, throughput and memory
checks. It does **not** establish convergence or a competitive learned policy.
The next stage should inspect behavior and undertake the planned larger,
multi-seed development runs, while retaining this initial result and keeping
final-test deals untouched. Full experiments, ablations and training-run
uncertainty are still pending.

### Saved artifacts

- Durable [pilot summary](dqn-pilot-101-20261104.json), including all paired differences.
- [Actual configuration and source hashes](dqn-pilot-run-config-101-20261104.json).
- [Training progress](dqn-pilot-progress-101-20261104.csv).
- Initial/final models and full hand/validation logs:
  `artifacts/dqn-pilot-101-20261104/`.

The checkpoint files are local artifacts ignored by Git; their loading code
and exact run configuration are in the project. The final evaluation was run
through the checkpoint loader rather than only the in-memory training model.
