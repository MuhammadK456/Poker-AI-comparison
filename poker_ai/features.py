"""Versioned, actor-relative 163-value encoding of visible information only."""

import numpy as np

from .environment import FULL_DECK, Observation
from .equity import EquityCalculator

FEATURE_VERSION = "visible-163-v1"
FEATURE_SIZE = 163
CARD_INDEX = {card: i for i, card in enumerate(FULL_DECK)}
LAST_ACTIONS = ("none", "fold", "check", "call", "bet", "raise")


class FeatureEncoder:
    def __init__(self, calculator: EquityCalculator, starting_stack: int = 200, *, include_equity: bool = True):
        if type(starting_stack) is not int or starting_stack <= 0:
            raise ValueError("starting_stack must be a positive integer")
        self.calculator = calculator
        self.starting_stack = starting_stack
        if type(include_equity) is not bool:
            raise ValueError("include_equity must be boolean")
        self.include_equity = include_equity

    def encode(self, obs: Observation) -> np.ndarray:
        values = np.zeros(FEATURE_SIZE, dtype=np.float32)
        for card in obs.hole_cards:
            values[CARD_INDEX[card]] = 1
        for card in obs.board:
            values[52 + CARD_INDEX[card]] = 1
        values[104 + obs.street] = 1
        values[108] = obs.is_button
        own, other = obs.seat, 1 - obs.seat
        values[109:116] = np.array([
            obs.pot, obs.call_amount, obs.stacks[own], obs.stacks[other],
            obs.street_bets[own], obs.street_bets[other], min(obs.stacks),
        ]) / self.starting_stack
        excess = max(0, max(obs.street_bets) - obs.street_bets[own] - obs.stacks[own])
        values[116] = obs.call_amount / (obs.pot - excess + obs.call_amount)
        # A constant zero keeps architecture/initialization identical for the ablation.
        if self.include_equity:
            values[117] = self.calculator.from_observation(obs).equity
        last = obs.history[-1] if obs.history else None
        values[118 + LAST_ACTIONS.index(last.kind if last else "none")] = 1
        values[124] = last.chips_added / self.starting_stack if last else 0
        categories = {"check": 0, "call": 1, "bet": 2, "raise": 2, "fold": 3}
        for action in obs.history:
            relative_seat = 0 if action.seat == own else 1
            values[125 + action.street * 8 + relative_seat * 4 + categories[action.kind]] += 1
        values[157:163] = obs.action_mask
        if not np.isfinite(values).all():
            raise ValueError("nonfinite feature")
        return values
