# Explaining the project

## A short introduction

This project compares handwritten and learned decision-making in heads-up poker.
Both agents get the same visible information, calculated equity estimate and
legal betting menu. The traditional agent uses explicit probability-based rules;
the learned agent uses a small Double DQN trained on simulated experience.
The experiment asks where each approach works and how confidently we can tell.

The final 400,000-hand test found no reliable direct winner. ML extracted more
value from a passive opponent, while the traditional bot did better against
random and loose/aggressive opponents. A separate development ablation found
that removing the supplied equity estimate usually hurt learning at our budget.
The result is a measured comparison, not a claim of professional poker strength.

## Follow one decision

1. PokerKit owns the game state. Our adapter passes only the acting player's
   cards, revealed board, public stacks/bets/history and legal actions to its agent.
2. The equity calculator estimates expected showdown pot share against uniformly
   distributed legal opponent cards. Preflop uses a 169-class lookup, flop/turn
   use 256 samples, and river enumerates 990 opponent combinations exactly.
3. The traditional bot compares equity with pot odds plus explicit margins,
   then applies conservative value-betting rules. It records its decision reason.
4. The learned bot encodes the same visible information into 163 numbers and
   predicts six Q-values with a 163→128→128→6 neural network. It chooses the
   highest-valued legal action. The action mask excludes illegal and duplicate sizes.
5. The engine executes the action and checks chip conservation. The agents
   never receive the full deck, opponent cards, deal seeds or opponent identity.

See the [four-hand replay](replay-example/README.md). It includes poor-looking
as well as successful actions; the fixed demo deals were not cherry-picked.

## How learning works

A transition runs from one learner decision to its next decision, including
the opponent's intervening turns. Intermediate rewards are zero. The final
reward is net chips divided by the starting 200-chip stack, including blinds.
The discount is one because a hand is a finite episode.

Transitions enter a 50,000-entry NumPy replay buffer. Every four learner
decisions after warmup, a minibatch updates the network with Huber loss and Adam.
Double DQN uses the online network to choose a legal next action and a separate
target network to evaluate it. Terminal transitions never bootstrap. Exploration
is also restricted to legal actions.

Training uses a fixed mix of random, passive, tight and loose/aggressive
opponents. The traditional agent is excluded from that training mixture.
Each of three seeds receives 250,000 learner decisions. We retain each final
checkpoint rather than reporting the best seed or a lucky intermediate peak.

## Why the evaluation is designed this way

Poker outcomes contain substantial card luck. Each deck is played twice,
swapping agents between seats while preserving the cards attached to each seat.
Policies being compared face the same deck groups. We bootstrap independent
whole deal pairs, not correlated individual seat-swapped hands.

The main metric is net big blinds per 100 hands, not hand-win percentage. Winning
many small pots can still lose money after a few large losses. Returns remain
accumulated even though starting stacks reset every hand.

Training, development and final deals have separate RNG namespaces. The final
protocol fixes sample sizes and checkpoints before play. The three primary
head-to-head intervals receive a Bonferroni adjustment. Secondary matchup
intervals are clearly labeled pointwise. Multiple training seeds and many
evaluation hands address different uncertainties; one cannot substitute for the other.

## Questions to be ready to answer

**Does equity tell the bot which bet is profitable?** No. It estimates pot share
if the hand reaches showdown against an assumed range. It does not model fold
probability, future bets or the changing strength implied by an opponent's actions.

**Why give ML a calculated equity feature?** The core comparison isolates
learned versus handwritten action selection using comparable inputs. The
no-equity ablation separately examines the value of that supplied estimate.

**Why does a Q-value sometimes exceed one?** The network output is unconstrained.
For this reward definition, true expected terminal values are bounded, so
out-of-range estimates expose approximation/calibration error. They are not
valid probabilities and do not prove that a selected action is good.

**Did ML beat the traditional bot?** Not reliably in direct final play. All
three adjusted intervals include zero. It did outperform the baseline against
some shared opponents and underperform against others. Describe those separately.

**Is this an equilibrium solver?** No. It learns against a fixed opponent
distribution. There is no exploitability bound, self-play equilibrium claim
or evidence about arbitrary human players.

**Why not keep tuning until ML wins?** That would change the project question
and bias the result. We completed the fixed-budget comparison, retained all
seeds and reported failures. Further work is outside this version's scope.

## Practical limits

The game has two players, reset 100-BB stacks, no rake, and discrete bet sizes.
The equity range is uniform; history features omit full action order. Three
training seeds give limited replication. Final deals are unseen, but opponent
settings are familiar. Checkpoints support inference, not exact training resume.
The project demonstrates simulation, RL engineering and statistical evaluation;
it does not establish human poker skill, financial performance or hiring outcomes.

## A factual portfolio description

“Built a reproducible heads-up poker comparison between an interpretable
equity-and-rules agent and masked Double DQN; trained three seeds, tested
400,000 held-out hands with paired uncertainty estimates, and evaluated an
equity-feature ablation under a fixed compute budget.”

Use that description only alongside an explanation you can give in your own
words. The [final report](final-evaluation-20261104/README.md), [ML guide](ml-pipeline.md)
and [traditional-agent guide](traditional-agent.md) provide the implementation detail.
