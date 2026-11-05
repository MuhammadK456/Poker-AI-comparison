# Traditional agent

Implemented November 4, 2026. This is an equity-and-rules baseline, not minimax,
CFR, betting rollouts or a learned strategy. Its purpose is an understandable
comparison partner for the future learned agent.

## Decision rules

The agent receives only the existing immutable observation. It requests equity
from the shared calculator, then applies these rules in order:

1. Facing a bet, compare equity with pot odds plus a caution margin. Fold below
   that threshold; otherwise consider a value raise or call. Free checking
   never triggers a fold.
2. At equity >= 0.85, try a pot-sized value bet/raise. Otherwise, or if that size
   fails the stack safeguard, try half-pot at equity >= 0.70.
3. Optionally bluff for half-pot when checked to on the button postflop, with
   equity below 0.70. The initial comparison uses 0% and 5% of eligible decisions.
4. If no raise is chosen, check/call.

The base calling margin is 0.03 before the river and 0 on the river. Add 0.02
out of position postflop and 0.02 after at least two raises on the current
street. Only the base margin disappears on the river; the explicitly modeled
position/aggression caution remains. These are chosen heuristics, not proven
optimal adjustments. The repeated-raise rule counts public raises by both
players and excludes opening bets and earlier streets.

Pot odds are `call / (callable_pot + call)`. Usually callable_pot equals the
displayed pot. When the opponent has bet more than a short stack can call,
subtract the uncallable excess that will be refunded. For example, facing a
200-chip shove with 78 chips left after posting a 2-chip blind, the contestable
pot after calling is 160, so the price is 78/160, not 78/280.

Pot odds alone omit future betting and a selective opponent calling range.
The equity feature still assumes a uniform opponent range. History changes
the calling margin, not the probability distribution over opponent hands.

## Sizes and stack safeguards

The policy chooses from the observation's existing legal concrete raises.
Fractional raise-to amounts use the same pot-after-call formula as the adapter.
It does not independently bypass the legal mask or invent extra bet sizes.

A bet/raise that commits either player's effective stack requires equity >=
0.95 for the traditional agent. This applies to the actual amount, regardless
of its slot: a deduplicated minimum or pot raise can still be an all-in.
Bluffs cannot commit the effective stack. The agent does not automatically
shove whenever equity exceeds 0.95; it still tries the normal value sizes.

The safeguard applies to raises. Calling an all-in still follows pot odds and
the caution margin; 0.95 is not a requirement to call an all-in. This avoids
confusing a conservative raising rule with the price of an existing bet.

## Reason traces and usage

`act(observation)` returns a legal action slot and stores `last_decision`.
The frozen decision records equity/method/sample count, pot odds, callable
pot, required calling equity, each margin, current-street raise count, remaining
effective stack, chosen raise-to/chips-added, whether the raise commits the
effective stack, and any eligible bluff draw. Draws use a policy RNG separate
from deck generation and equity estimation. Do not call `decide()` a second
time just to log; doing so can consume another bluff draw.

```python
from poker_ai.agents import PassiveAgent
from poker_ai.environment import PokerHand, shuffled_deck
from poker_ai.equity import EquityCalculator
from poker_ai.traditional import TraditionalAgent

hero = TraditionalAgent(EquityCalculator(), seed=101)  # default: no bluffs
agents = (hero, PassiveAgent())
hand = PokerHand(shuffled_deck(202))
while not hand.terminal:
    obs = hand.observe()
    slot = agents[obs.seat].act(obs)
    if obs.seat == 0:
        print(hero.last_decision)
    hand.step(slot)
print(hand.result().net_chips)
```

## Fixed opponents and ablation

| Opponent | Calling rule | Aggression rule |
|---|---|---|
| Random | Uniform choice among deduplicated legal actions | Same random choice |
| Passive | Always check/call | Never |
| Tight | Equity >= max(0.50, pot odds) | Half-pot at >= 0.75; effective all-in guard 0.95 |
| Loose/aggressive | Equity >= max(0.30, pot odds) | Half-pot at >= 0.58; effective all-in guard 0.90 |

Threshold opponents check freely, ignore position/history, and do not bluff.
They are fixed development styles, not a claim to reproduce human players.
The `pot_odds` ablation only checks/calls or folds on equity versus pot odds:
no margins, value betting or bluffs. Both variants and the ablation use the
same equity calculator settings.

## Development protocol

`configs/traditional-development.json` was saved before the run. Only bluff
frequency is compared (0 vs 5%); all other traditional thresholds are fixed.
The ablation is descriptive and cannot be selected as the main policy.

The fixed budget is 250 independent decks, each replayed with seats swapped,
for all three policies against all four opponents: 6,000 hands. The same
physical cards/decks are used across variants and opponents. Dealing, policy,
opponent and bootstrap seeds have separate namespaces. Seeds are explicitly
development seeds; no final-test deals are used.

The metric is net bb/100. Intervals use 2,000 percentile bootstrap samples of
whole paired-deal groups. For selecting bluff frequency, the per-deal paired
improvement is averaged equally across opponents *before* bootstrap, preserving
correlation from shared decks. Select 5% only if its 95% interval lies entirely
above zero; otherwise retain 0%. No extra hands are added after inspecting
results. The small development sample can leave close comparisons unresolved.

Outputs include hand CSVs, decision JSONL, source/table hashes, full configuration,
results and a frozen selected policy. Decision traces exclude raw hole cards;
they are experiment logs and must never become inputs to another agent.
The equity cache is shared and policy order is fixed, so recorded latency is
diagnostic, not a fair policy speed comparison.

## Validation and remaining work

### Recorded development result

The fixed run completed **6,000 hands in 21.48 seconds**, with legal actions
and chip-accounting checks active. The selected `no_bluff` policy had these
development returns (500 hands / 250 independent paired deals per opponent):

| Opponent | Net bb/100 | Paired 95% bootstrap interval |
|---|---:|---:|
| Random | +1,019.2 | +661.4 to +1,361.6 |
| Passive | +211.2 | +154.0 to +271.8 |
| Tight | -23.0 | -116.8 to +71.5 |
| Loose/aggressive | +203.6 | +104.6 to +315.2 |

Large returns against these simple opponents are not a human-poker benchmark.
The tight matchup is inconclusive, not evidence of equal strength or a
confirmed loss. The 5% bluff variant improved the equally weighted mean by
only **2.05 bb/100**, with paired interval **[-11.10, +17.65]**. The predeclared
rule therefore retains no bluffs; it does not establish that bluffing is harmful.

The pot-odds-only ablation returned exactly zero against passive play on the
paired deals (both check/call without betting), and lost against the tight and
loose/aggressive opponents in this run. Full results, including the ablation
and both candidate configurations, are preserved rather than just the winner.

The frozen baseline is `configs/traditional-v1.json`. `TraditionalConfig()`
currently matches this selected no-bluff configuration. To explicitly load the
snapshot in a future experiment:

```python
import json
from pathlib import Path
from poker_ai.traditional import TraditionalConfig

snapshot = json.loads(Path("configs/traditional-v1.json").read_text())
config = TraditionalConfig(**snapshot["config"])
# hero = TraditionalAgent(calculator, config, seed=101)
```

Durable records:

- [Complete results](traditional-development-20261104.json)
- [Run configuration and source/table hashes](traditional-development-run-config-20261104.json)
- Raw hands and decision traces: `artifacts/traditional-development-20261104/`

All 46 project tests pass. New cases cover threshold boundaries, concrete
deduplicated sizes, free checks, position/history margins, short-stack pot odds,
effective all-in guards, reproducible bluff opportunities, the pot-odds ablation,
threshold opponents, paired bootstrap, repeated-run logs and logged-hand replay.

The learned agent, 163-feature encoding, replay buffer and transition/reward
pipeline are still pending. Final held-out evaluation, broader robustness and
learning-seed uncertainty remain later work. Development performance must not
be reported as final generalization performance.
