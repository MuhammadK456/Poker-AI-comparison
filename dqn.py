"""Small CPU masked Double DQN and inference-only checkpoint loading."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from .features import FEATURE_SIZE, FEATURE_VERSION, FeatureEncoder


@dataclass(frozen=True)
class DQNConfig:
    learning_rate: float = 0.0003
    batch_size: int = 128
    replay_capacity: int = 50_000
    warmup: int = 5000
    update_every: int = 4
    target_every: int = 1000
    gamma: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 100_000
    gradient_clip: float = 10.0

    def __post_init__(self):
        for name in ("batch_size", "replay_capacity", "warmup", "update_every", "target_every", "epsilon_decay"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "gamma", "epsilon_start", "epsilon_end", "gradient_clip"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.batch_size > self.replay_capacity or self.warmup < self.batch_size:
            raise ValueError("batch must fit replay and warmup")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1 or not 0 <= self.gamma <= 1:
            raise ValueError("invalid epsilon or gamma")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning rate and gradient clip must be positive")

    def epsilon(self, decisions: int) -> float:
        fraction = min(1.0, max(0, decisions) / self.epsilon_decay)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)


class QNetwork(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(FEATURE_SIZE, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 6))


def masked_argmax(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape or mask.dtype != torch.bool or not mask.any(dim=1).all():
        raise ValueError("every nonterminal row needs a compatible legal mask")
    if not torch.isfinite(values).all():
        raise ValueError("nonfinite action values")
    return values.masked_fill(~mask, -torch.inf).argmax(dim=1)


@torch.no_grad()
def double_dqn_targets(online, target, next_observations, next_masks, rewards, terminals, gamma=1.0):
    # Do not evaluate terminal rows: their masks are empty, and 0 * -inf is NaN.
    targets = rewards.clone()
    live = ~terminals
    if live.any():
        actions = masked_argmax(online(next_observations[live]), next_masks[live])
        values = target(next_observations[live]).gather(1, actions[:, None]).squeeze(1)
        targets[live] += gamma * values
    if not torch.isfinite(targets).all():
        raise ValueError("nonfinite targets")
    return targets


class DoubleDQN:
    def __init__(self, config: DQNConfig = DQNConfig(), seed: int = 101, *, include_equity: bool = True):
        if type(include_equity) is not bool:
            raise ValueError("include_equity must be boolean")
        self.include_equity = include_equity
        self.config = config
        self.seed = seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.online, self.target = QNetwork(), QNetwork()
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.gradient_updates = 0

    @torch.no_grad()
    def choose(self, features: np.ndarray, mask, epsilon: float, rng: random.Random) -> int:
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (6,) or not mask.any():
            raise ValueError("need at least one legal action")
        if epsilon and rng.random() < epsilon:
            return int(rng.choice(np.flatnonzero(mask).tolist()))
        values = self.online(torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0))
        return int(masked_argmax(values, torch.from_numpy(mask).unsqueeze(0))[0])

    def update(self, batch: dict) -> dict:
        data = {name: torch.from_numpy(value) for name, value in batch.items()}
        targets = double_dqn_targets(self.online, self.target, data["next_observations"],
                                     data["next_masks"], data["rewards"], data["terminals"], self.config.gamma)
        predictions = self.online(data["observations"]).gather(1, data["actions"][:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite learning loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.gradient_updates += 1
        if self.gradient_updates % self.config.target_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return {"loss": float(loss.detach()), "gradient_norm": float(norm)}

    def save(self, path: Path, decisions: int, metadata: dict):
        if path.exists():
            raise FileExistsError(path)
        torch.save({"schema_version": 1, "feature_version": FEATURE_VERSION,
                    "include_equity": self.include_equity,
                    "config": asdict(self.config), "seed": self.seed,
                    "online": self.online.state_dict(), "target": self.target.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "gradient_updates": self.gradient_updates,
                    "decisions": decisions, "metadata": metadata,
                    "purpose": "inference and inspection; replay/RNG state omitted, not an exact training-resume checkpoint"}, path)

    @classmethod
    def load_for_inference(cls, path: Path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["schema_version"] != 1 or saved["feature_version"] != FEATURE_VERSION:
            raise ValueError("incompatible checkpoint")
        # Existing v1 checkpoints predate ablations and always include equity.
        learner = cls(DQNConfig(**saved["config"]), saved["seed"],
                      include_equity=saved.get("include_equity", True))
        learner.online.load_state_dict(saved["online"])
        learner.target.load_state_dict(saved["target"])
        learner.gradient_updates = saved["gradient_updates"]
        learner.online.eval()
        return learner


class DQNAgent:
    """Frozen greedy policy; no exploration, updates or replay insertion."""

    def __init__(self, learner: DoubleDQN, encoder: FeatureEncoder):
        if learner.include_equity != encoder.include_equity:
            raise ValueError("encoder equity setting does not match checkpoint")
        self.learner, self.encoder = learner, encoder
        self._rng = random.Random(0)  # Greedy choose() never consumes it.

    def act(self, observation) -> int:
        return self.learner.choose(self.encoder.encode(observation), observation.action_mask, 0.0, self._rng)
