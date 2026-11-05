"""Bounded dense replay; no per-transition Python dictionaries are retained."""

import numpy as np

from .features import FEATURE_SIZE


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self.size = self.cursor = 0
        self.observations = np.empty((capacity, FEATURE_SIZE), dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.masks = np.empty((capacity, 6), dtype=np.bool_)
        self.next_masks = np.empty_like(self.masks)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminals = np.empty(capacity, dtype=np.bool_)

    @property
    def nbytes(self) -> int:
        return sum(getattr(self, name).nbytes for name in self.array_names)

    array_names = ("observations", "next_observations", "masks", "next_masks", "actions", "rewards", "terminals")

    def add(self, observation, mask, action, reward, next_observation, next_mask, terminal):
        observation = np.asarray(observation, dtype=np.float32)
        next_observation = np.asarray(next_observation, dtype=np.float32)
        mask, next_mask = np.asarray(mask, dtype=bool), np.asarray(next_mask, dtype=bool)
        if observation.shape != (FEATURE_SIZE,) or next_observation.shape != (FEATURE_SIZE,):
            raise ValueError("incorrect feature shape")
        if mask.shape != (6,) or next_mask.shape != (6,):
            raise ValueError("incorrect mask shape")
        if not np.isfinite(observation).all() or not np.isfinite(next_observation).all() or not np.isfinite(reward):
            raise ValueError("nonfinite transition")
        if type(action) is not int or not 0 <= action < 6 or not mask[action]:
            raise ValueError("replay action must be legal")
        if not np.array_equal(observation[157:], mask) or not np.array_equal(next_observation[157:], next_mask):
            raise ValueError("feature mask differs from transition mask")
        if terminal:
            if next_mask.any() or next_observation.any():
                raise ValueError("terminal transition must have zero next features and mask")
        elif not next_mask.any() or reward != 0:
            raise ValueError("nonterminal transition requires legal next actions and zero reward")
        i = self.cursor
        self.observations[i], self.next_observations[i] = observation, next_observation
        self.masks[i], self.next_masks[i] = mask, next_mask
        self.actions[i], self.rewards[i], self.terminals[i] = action, reward, terminal
        self.cursor = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict:
        if not 0 < batch_size <= self.size:
            raise ValueError("batch exceeds replay size")
        indices = rng.choice(self.size, size=batch_size, replace=False)
        return {name: getattr(self, name)[indices] for name in self.array_names}
