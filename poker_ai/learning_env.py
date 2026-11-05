"""Advance a hand from one learner turn to its next turn (or settlement)."""

from dataclasses import dataclass

from .agents import Agent
from .environment import GameConfig, Observation, PokerHand


@dataclass(frozen=True)
class LearnerStep:
    observation: Observation | None
    reward: float
    terminal: bool


class LearnerEpisode:
    def __init__(self, deck: tuple[str, ...], learner_seat: int, opponent: Agent,
                 config: GameConfig = GameConfig()):
        if learner_seat not in (0, 1):
            raise ValueError("learner seat must be 0 or 1")
        self.learner_seat = learner_seat
        self._hand = PokerHand(deck, config)
        self._opponent = opponent
        self._reward_scale = config.starting_stacks[learner_seat]
        self._advance_opponent()

    def _advance_opponent(self):
        while not self._hand.terminal:
            obs = self._hand.observe()
            if obs.seat == self.learner_seat:
                break
            self._hand.step(self._opponent.act(obs))

    @property
    def observation(self) -> Observation | None:
        return None if self._hand.terminal else self._hand.observe()

    def step(self, slot: int) -> LearnerStep:
        if self._hand.terminal:
            raise ValueError("episode already ended")
        self._hand.step(slot)
        self._advance_opponent()
        terminal = self._hand.terminal
        reward = self.result().net_chips[self.learner_seat] / self._reward_scale if terminal else 0.0
        return LearnerStep(self.observation, reward, terminal)

    def result(self):
        return self._hand.result()
