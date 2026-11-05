"""Sanity opponents only; neither is the planned traditional or ML bot."""

import random
from typing import Protocol

from .environment import Action, Observation


class Agent(Protocol):
    def act(self, observation: Observation) -> Action: ...


class PassiveAgent:
    def act(self, observation: Observation) -> Action:
        return Action.CHECK_CALL


class RandomAgent:
    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    def act(self, observation: Observation) -> Action:
        return self._rng.choice(observation.legal_actions).slot
