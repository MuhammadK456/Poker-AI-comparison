# Equity calculator

Implemented November 3, 2026, using Treys 0.1.8 and the existing PokerKit
observation interface. No neural network or betting strategy is implemented
by this component.

## Meaning of the number

For each possible sampled showdown, count a win as 1, a tie as 0.5, and a loss
as 0. The average is the hero's expected share of the showdown pot. Both agents
will receive estimates computed under the same assumptions and sample budget.

The initial opponent range is uniform over hands that do not conflict with
visible cards. The calculator does not use betting history to narrow that
range. It assumes showdown, without simulating bets, folds or continuation
policies. More card samples reduce sampling noise, not errors from a wrong
opponent-range assumption.

## Street-specific computation

| Street | Method | Default work |
|---|---|---|
| Preflop | Lookup by pair/suited/offsuit class | 169 entries, each generated from 10,000 samples |
| Flop | Sample two opponent cards and two future board cards | 256 trials |
| Turn | Sample two opponent cards and one river card | 256 trials |
| River | Enumerate every legal opponent pair | 45 choose 2 = 990 hands |

Samples draw all unknown cards without replacement within each trial. Trials
are independent. Unseen burned cards are marginalized rather than revealed
to the calculator. Showdown ranks come from Treys; lower rank means a stronger
hand, which is the opposite of PokerKit's comparable hand ordering.
Treys' read-only rank lookup tables are constructed once per process and
reused, avoiding repeated setup cost on each decision.

Preflop's 1,326 concrete hands collapse to 169 classes because suits have no
intrinsic strength. There are 13 pairs, 78 suited rank combinations and 78
offsuit combinations. Each class uses one representative hand. This equivalence
holds for the uniform range used here; a custom range would need a new design.
The table is an approximation, not a table of exact preflop probabilities.

## API and reproducibility

```python
from poker_ai.equity import EquityCalculator

calculator = EquityCalculator()
estimate = calculator.evaluate(("As", "Ks"), ("Qs", "Js", "2d"))
print(estimate.equity, estimate.method, estimate.outcomes)
# With a live acting-player snapshot:
# estimate = calculator.from_observation(observation)
```

Results expose win/tie/loss counts, the method, equity and sampling standard
error. Standard error describes the sample mean of pot share, accounting for
ties; it is not uncertainty about the opponent's behavior. Exact river
enumeration has no Monte Carlo sampling error under the specified range.
An identical-outcome Monte Carlo sample can report zero sample standard error
without establishing certainty.

The calculator uses only hole cards and board cards. It never receives an
engine object, opponent cards, future deck, deal ID or deal seed. Input cards
are validated for notation, legal street length and uniqueness.

Postflop sampling seeds are derived from visible cards, the estimator seed,
sample count and algorithm version, using a separate seed namespace from
dealing and policy decisions. Card order within the hole or board group is
normalized. Identical visible states therefore receive identical features
regardless of evaluation order or cache eviction. To obtain independent
repeats for error studies, use different estimator seeds or `monte_carlo` seeds.

The per-calculator LRU cache holds at most 4,096 visible postflop states by
default. Setting `EquityConfig(cache_size=0)` disables caching without changing
results. The preflop table has only 169 entries and is stored separately. The
configuration and returned estimates are immutable. Cache limits are counts,
not a verified end-to-end training memory measurement.

## Validation

The full project suite currently has **33 passing tests**. Equity checks cover:

- All 52 card conversions and the 1,326-to-169 preflop class mapping.
- Known winning, losing and split showdowns; exact nut and shared-board ties.
- 1,000 random showdown comparisons against PokerKit.
- A full 990-opponent river enumeration independently evaluated with PokerKit.
- Sampled card legality, a Monte Carlo/exact comparison and standard-error math.
- Complete table metadata/count validation and reproducible generation.
- Deterministic routing, bounded caching, immutable results and no changes to
  the game's observation or global RNG.

The separate audit expands the historical 40-state river experiment to
100 states, 10 repeats each, at 128/256/1,024 samples. Raw observations are in
`artifacts/equity-audit-final-20261103/river-errors.csv`; the summary includes timing
and a 1,000-hand random/passive integration run. These river errors do not
establish flop/turn accuracy or performance against a nonuniform opponent.

### Recorded measurements

The final run passed all 33 tests in 4.156 seconds. The audit completed all
1,000 live hands and 6,156 decisions in 17.30 seconds, using 3,314 of the
4,096 available cache entries. It exercised preflop lookup 2,264 times,
Monte Carlo 2,891 times and exact river evaluation 1,001 times.

| Samples | River mean absolute error | River RMSE | Flop median time |
|---|---|---|---|
| 128 | 2.66 percentage points | 3.47 pp | 1.97 ms |
| 256 | 1.84 percentage points | 2.45 pp | 3.93 ms |
| 1,024 | 0.95 percentage points | 1.24 pp | 15.78 ms |

Exact river enumeration took a median 6.86 ms. Error and time columns describe
different experiments: errors use 100 uniformly sampled river states and 10
estimator repeats; flop timing uses 30 repeats of one fixed situation. These
are local timings, not general training-time forecasts. Reusing Treys' rank
tables removed setup overhead without changing sampled outcomes.

The small, durable results snapshot is [equity-audit-20261103.json](equity-audit-20261103.json).
It records configuration, runtime versions and the generated preflop table's
SHA-256 hash. Large raw logs remain in the ignored artifacts directory.
