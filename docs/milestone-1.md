# Engine adapter and reproducible matches

Implemented November 3, 2026. Runtime: Python 3.11; PokerKit 0.7.7.

## Decisions

The engine adapter uses PokerKit and the standard library. Probability
estimation and learning are implemented in separate modules.

Card automations are disabled. After creating the engine state, the adapter
replaces its undealt deck with an explicit 52-card permutation and advances
dealing/burning when no player action is pending. All other settlement
automations remain enabled. This ensures future cards do not depend on random
numbers consumed by agent policies. Full deck validation prevents duplicate or
missing cards. Fixed deck tests check hole and board consumption.

Action slots are stable; legal availability changes at each decision. Raises
use total street contribution, not incremental chips. Fractional raise-to is
`M + round_half_up(f * (P + C))`, where M is the highest street bet, P includes
current bets and C is the actor's actual call amount. Amounts are clamped to
engine bounds and validated against the engine. Duplicate raises keep the
first slot in this order: minimum, half-pot, pot, all-in. An all-in-sized move
can therefore use an earlier slot. An all-in call is always check/call.

Snapshots are frozen dataclasses containing immutable tuples, including public
history and legal actions. They contain no deal IDs, seeds, opponent identity,
opponent hole cards, future cards or engine object. Noninterference tests
change hidden opponent/future cards while holding visible information fixed.

Seed derivation uses SHA-256 over explicit namespaces. Each pair shares a deck;
each seat assignment receives a separate policy seed. Physical seats and cards
stay fixed; agent assignments rotate. Logs can recreate each hand from the
deck seed and chosen slots. They deliberately contain no wall-clock fields so
repeat runs can be compared byte for byte.

## Verification

Recorded result: **19 tests passed** in 3.479 seconds. The separate CLI run
completed all **1,000 pairs / 2,000 hands** with chip invariants intact. The
random agent returned -6,139 chips and the passive agent +6,139 chips; these
are sanity-opponent smoke results, not conclusions about the planned agents.

Run `python -m unittest discover -s tests -v` using the project environment.
The suite covers:

- Action order on all four streets and correct board lengths.
- Raise-to versus chips added, fractional pot sizing, masks and deduplication.
- Rejection of illegal actions without mutation; no free-check folds.
- Short all-ins that do not reopen betting; unequal-stack refunds and payouts.
- Forced ties, blind-sized stacks, deck validity and no duplicate dealt cards.
- Frozen observations and preflop/postflop hidden-card noninterference.
- Fixed deck behavior independent of global randomness.
- 1,000 passive and 1,000 random-versus-random hands with chip/mask checks.
- Actual paired deck/hole-card equality, swapped agents, reproducible logs,
  action-trace replay and refusal to overwrite existing run directories.

The separate 1,000-pair random-versus-passive CLI smoke run exercises 2,000
additional hands. Its output is local under `artifacts/milestone-1-smoke/`.
These checks validate this adapter's initial scope; they do not establish
learning performance, full engine correctness or end-to-end training speed.
