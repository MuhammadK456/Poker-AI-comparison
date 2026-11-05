"""Explainable, hand-authored policies. These are not betting-rollout solvers."""

from dataclasses import dataclass, fields
import math
import random

from .environment import Action, LegalAction, Observation
from .equity import EquityCalculator, EquityEstimate


@dataclass(frozen=True)
class TraditionalConfig:
    base_margin: float = 0.03
    position_margin: float = 0.02
    repeated_raise_margin: float = 0.02
    half_pot_threshold: float = 0.70
    pot_threshold: float = 0.85
    allin_threshold: float = 0.95
    bluff_probability: float = 0.0
    value_betting: bool = True

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "value_betting":
                if type(value) is not bool:
                    raise ValueError("value_betting must be boolean")
            elif type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{field.name} must be finite and in [0, 1]")
        if not self.half_pot_threshold <= self.pot_threshold <= self.allin_threshold:
            raise ValueError("thresholds must satisfy half-pot <= pot <= all-in")


@dataclass(frozen=True)
class Decision:
    slot: Action
    reason: str
    equity: float
    equity_method: str
    equity_outcomes: int
    pot_odds: float
    callable_pot: int
    required_equity: float
    base_margin: float
    position_adjustment: float
    aggression_adjustment: float
    raises_this_street: int
    effective_stack: int
    raise_to: int | None
    chips_added: int
    commits_effective_stack: bool
    bluff_probability: float
    bluff_draw: float | None


def _commits_stack(obs: Observation, amount: int) -> bool:
    # Check concrete amounts, not slots: a deduplicated MIN_RAISE can be all-in.
    return amount >= min(obs.stacks[i] + obs.street_bets[i] for i in (0, 1))


def _callable_pot(obs: Observation) -> int:
    # Exclude an opponent's uncallable excess, which will be refunded. This
    # matters in unequal-stack fixtures; otherwise a short call looks too cheap.
    excess = max(0, max(obs.street_bets) - obs.street_bets[obs.seat] - obs.stacks[obs.seat])
    return obs.pot - excess


def _pot_odds(obs: Observation) -> float:
    return obs.call_amount / (_callable_pot(obs) + obs.call_amount)


def _sized_raise(obs: Observation, fraction: float, equity: float,
                 allin_threshold: float) -> LegalAction | None:
    raises = [a for a in obs.legal_actions if a.raise_to is not None]
    if not raises:
        return None
    increment = ((obs.pot + obs.call_amount + 1) // 2 if fraction == 0.5
                 else obs.pot + obs.call_amount)
    target = max(obs.street_bets) + increment
    target = max(min(a.raise_to for a in raises), min(max(a.raise_to for a in raises), target))
    candidate = next((a for a in raises if a.raise_to == target), None)
    if candidate is not None and _commits_stack(obs, target) and equity < allin_threshold:
        return None
    return candidate


def _trace(obs: Observation, estimate: EquityEstimate, action: LegalAction, reason: str,
           required: float, base: float = 0, position: float = 0, aggression: float = 0,
           bluff_probability: float = 0, bluff_draw: float | None = None) -> Decision:
    added = (action.raise_to - obs.street_bets[obs.seat] if action.raise_to is not None
             else obs.call_amount if action.slot == Action.CHECK_CALL else 0)
    return Decision(
        slot=action.slot, reason=reason, equity=estimate.equity,
        equity_method=estimate.method, equity_outcomes=estimate.outcomes,
        pot_odds=_pot_odds(obs), callable_pot=_callable_pot(obs), required_equity=required,
        base_margin=base, position_adjustment=position, aggression_adjustment=aggression,
        raises_this_street=sum(a.street == obs.street and a.kind == "raise" for a in obs.history),
        effective_stack=min(obs.stacks), raise_to=action.raise_to, chips_added=added,
        commits_effective_stack=(action.raise_to is not None and _commits_stack(obs, action.raise_to)),
        bluff_probability=bluff_probability, bluff_draw=bluff_draw,
    )


class TraditionalAgent:
    """Monte Carlo equity plus explicit pot-odds/value/bluff rules.

    `last_decision` records the last action actually requested. Calling decide()
    also consumes any eligible bluff draw; do not call it again just to log.
    """

    def __init__(self, calculator: EquityCalculator, config: TraditionalConfig = TraditionalConfig(),
                 seed: int = 0):
        self.config = config
        self._calculator = calculator
        self._rng = random.Random(seed)
        self.last_decision: Decision | None = None

    def decide(self, obs: Observation) -> Decision:
        estimate = self._calculator.from_observation(obs)
        equity = estimate.equity
        raises = sum(a.street == obs.street and a.kind == "raise" for a in obs.history)
        base = self.config.base_margin if obs.street < 3 else 0.0
        position = self.config.position_margin if obs.street > 0 and not obs.is_button else 0.0
        aggression = self.config.repeated_raise_margin if raises >= 2 else 0.0
        pot_odds = _pot_odds(obs)
        required = min(1.0, pot_odds + base + position + aggression)
        by_slot = {a.slot: a for a in obs.legal_actions}
        action, reason = by_slot[Action.CHECK_CALL], "check_free" if obs.call_amount == 0 else "call_pot_odds"
        bluff_draw = None
        if obs.call_amount > 0 and equity < required:
            action, reason = by_slot[Action.FOLD], "fold_below_call_threshold"
        else:
            if self.config.value_betting:
                for threshold, fraction, label in (
                    (self.config.pot_threshold, 1.0, "value_pot"),
                    (self.config.half_pot_threshold, 0.5, "value_half_pot"),
                ):
                    if equity >= threshold:
                        candidate = _sized_raise(obs, fraction, equity, self.config.allin_threshold)
                        if candidate is not None:
                            action, reason = candidate, label
                            break
            # Bluff only after the opponent checks to the button postflop.
            checked_to = (obs.street > 0 and obs.is_button and obs.call_amount == 0
                          and obs.history and obs.history[-1].street == obs.street
                          and obs.history[-1].seat != obs.seat and obs.history[-1].kind == "check")
            if (action.raise_to is None and checked_to
                    and equity < self.config.half_pot_threshold and self.config.bluff_probability > 0):
                # Bluffs never commit the effective stack, even if a size was clamped.
                candidate = _sized_raise(obs, 0.5, 0.0, 1.0)
                if candidate is not None:
                    bluff_draw = self._rng.random()
                    if bluff_draw < self.config.bluff_probability:
                        action, reason = candidate, "bluff_checked_to"
        decision = _trace(obs, estimate, action, reason, required, base, position, aggression,
                          self.config.bluff_probability, bluff_draw)
        self.last_decision = decision
        return decision

    def act(self, observation: Observation) -> Action:
        return self.decide(observation).slot


POT_ODDS_CONFIG = TraditionalConfig(base_margin=0, position_margin=0,
                                   repeated_raise_margin=0, value_betting=False)


@dataclass(frozen=True)
class ThresholdConfig:
    call_threshold: float
    raise_threshold: float
    allin_threshold: float

    def __post_init__(self):
        values = (self.call_threshold, self.raise_threshold, self.allin_threshold)
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or not 0 <= values[0] <= values[1] <= values[2] <= 1):
            raise ValueError("thresholds must be finite and satisfy 0 <= call <= raise <= all-in <= 1")


TIGHT_CONFIG = ThresholdConfig(0.50, 0.75, 0.95)
LOOSE_CONFIG = ThresholdConfig(0.30, 0.58, 0.90)


class ThresholdAgent:
    """Fixed development opponent; no position/history adjustment or bluffing."""

    def __init__(self, calculator: EquityCalculator, config: ThresholdConfig):
        self._calculator = calculator
        self.config = config
        self.last_decision: Decision | None = None

    def act(self, obs: Observation) -> Action:
        estimate = self._calculator.from_observation(obs)
        required = max(self.config.call_threshold, _pot_odds(obs))
        by_slot = {a.slot: a for a in obs.legal_actions}
        action, reason = by_slot[Action.CHECK_CALL], "threshold_call" if obs.call_amount else "check_free"
        if obs.call_amount and estimate.equity < required:
            action, reason = by_slot[Action.FOLD], "threshold_fold"
        elif estimate.equity >= self.config.raise_threshold:
            candidate = _sized_raise(obs, 0.5, estimate.equity, self.config.allin_threshold)
            if candidate is not None:
                action, reason = candidate, "threshold_value"
        self.last_decision = _trace(obs, estimate, action, reason, required)
        return action.slot
