"""Showdown pot share against a uniform legal opponent range.

Equity is win probability + half the tie probability, assuming showdown.
It is not the expected return of a bet and contains no model of folds.
"""

import argparse
from collections import OrderedDict
from dataclasses import asdict, dataclass
from itertools import combinations
import json
from math import sqrt
from pathlib import Path
import random

from treys import Card, Evaluator

from .environment import FULL_DECK, Observation
from .randomness import derived_seed


CARD_INTS = {card: Card.new(card) for card in FULL_DECK}
RANKS = "AKQJT98765432"
ALGORITHM = "uniform-showdown-v1"
DEFAULT_TABLE = Path(__file__).parent / "data" / "preflop-uniform-v1.json"
# Treys builds rank lookup tables at construction; evaluate() only reads them.
_EVALUATOR = Evaluator()


def _positive_int(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _cards(hole, board) -> tuple[tuple[str, ...], tuple[str, ...]]:
    hole, board = tuple(hole), tuple(board)
    if len(hole) != 2 or len(board) not in (0, 3, 4, 5):
        raise ValueError("need two hole cards and 0, 3, 4 or 5 board cards")
    cards = hole + board
    if any(not isinstance(c, str) or c not in CARD_INTS for c in cards):
        raise ValueError("cards must use rank/suit notation, e.g. As, Td, 2c")
    if len(set(cards)) != len(cards):
        raise ValueError("visible cards must be distinct")
    # Order within hole/board groups carries no equity information.
    return tuple(sorted(hole)), tuple(sorted(board))


@dataclass(frozen=True)
class EquityEstimate:
    wins: int
    ties: int
    losses: int
    method: str

    def __post_init__(self):
        counts = (self.wins, self.ties, self.losses)
        if any(type(x) is not int or x < 0 for x in counts) or sum(counts) <= 0:
            raise ValueError("outcome counts must be nonnegative integers with a positive total")
        if self.method not in ("monte_carlo", "exact_river", "preflop_lookup"):
            raise ValueError("unknown estimation method")

    @property
    def outcomes(self) -> int:
        return self.wins + self.ties + self.losses

    @property
    def equity(self) -> float:
        return (self.wins + 0.5 * self.ties) / self.outcomes

    @property
    def standard_error(self) -> float | None:
        """Sample SE of pot share, not uncertainty in the opponent model.

        An all-identical sample can yield zero SE without proving certainty.
        Exact enumeration has zero Monte Carlo sampling error.
        """
        if self.method == "exact_river":
            return 0.0
        n = self.outcomes
        if n < 2:
            return None
        variance = (self.wins + 0.25 * self.ties - n * self.equity**2) / (n - 1)
        return sqrt(max(0.0, variance) / n)


def showdown_share(hole, opponent, board) -> float:
    """Known-opponent evaluator for tests/analysis, not agent observations."""
    hole, board = _cards(hole, board)
    opponent, _ = _cards(opponent, board)
    if len(board) != 5 or set(hole) & set(opponent):
        raise ValueError("showdown requires a full board and nine distinct cards")
    b = [CARD_INTS[c] for c in board]
    hero_score = _EVALUATOR.evaluate([CARD_INTS[c] for c in hole], b)
    other_score = _EVALUATOR.evaluate([CARD_INTS[c] for c in opponent], b)
    return 1.0 if hero_score < other_score else 0.5 if hero_score == other_score else 0.0


def monte_carlo(hole, board, *, samples: int = 256, seed: int = 0) -> EquityEstimate:
    """Sample opponent cards and missing board jointly without replacement.

    Each trial is independent. Burns are unobserved and can be marginalized:
    uniformly drawing the opponent and final board gives the same distribution.
    """
    hole, board = _cards(hole, board)
    _positive_int(samples, "samples")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    rng = random.Random(seed)
    known = set(hole + board)
    remaining = [CARD_INTS[c] for c in FULL_DECK if c not in known]
    hero, public = [CARD_INTS[c] for c in hole], [CARD_INTS[c] for c in board]
    missing = 5 - len(board)
    fixed_score = _EVALUATOR.evaluate(hero, public) if missing == 0 else None
    wins = ties = 0
    for _ in range(samples):
        draw = rng.sample(remaining, 2 + missing)
        final_board = public + draw[2:]
        hero_score = fixed_score if fixed_score is not None else _EVALUATOR.evaluate(hero, final_board)
        other_score = _EVALUATOR.evaluate(draw[:2], final_board)
        wins += hero_score < other_score
        ties += hero_score == other_score
    return EquityEstimate(wins, ties, samples - wins - ties, "monte_carlo")


def exact_river(hole, board) -> EquityEstimate:
    hole, board = _cards(hole, board)
    if len(board) != 5:
        raise ValueError("exact river equity requires five board cards")
    known = set(hole + board)
    remaining = [CARD_INTS[c] for c in FULL_DECK if c not in known]
    public = [CARD_INTS[c] for c in board]
    hero_score = _EVALUATOR.evaluate([CARD_INTS[c] for c in hole], public)
    wins = ties = 0
    for opponent in combinations(remaining, 2):
        other_score = _EVALUATOR.evaluate(list(opponent), public)
        wins += hero_score < other_score
        ties += hero_score == other_score
    return EquityEstimate(wins, ties, 990 - wins - ties, "exact_river")


def preflop_class(hole) -> str:
    hole, _ = _cards(hole, ())
    first, second = sorted(hole, key=lambda card: RANKS.index(card[0]))
    ranks = first[0] + second[0]
    if first[0] == second[0]:
        return ranks
    return ranks + ("s" if first[1] == second[1] else "o")


def preflop_representatives() -> dict[str, tuple[str, str]]:
    representatives = {}
    for i, high in enumerate(RANKS):
        representatives[high + high] = (high + "s", high + "h")
        for low in RANKS[i + 1:]:
            representatives[high + low + "s"] = (high + "s", low + "s")
            representatives[high + low + "o"] = (high + "s", low + "h")
    return representatives


def build_preflop_table(*, samples: int = 10_000, seed: int = 20261027,
                        progress=None) -> dict:
    _positive_int(samples, "samples")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    entries = {}
    for key, cards in preflop_representatives().items():
        result = monte_carlo(cards, (), samples=samples,
                             seed=derived_seed(seed, ALGORITHM, "preflop", key))
        entries[key] = {"wins": result.wins, "ties": result.ties, "losses": result.losses}
        if progress is not None:
            progress(len(entries), 169)
    return {"schema_version": 1, "algorithm": ALGORITHM, "treys": "0.1.8",
            "opponent_range": "uniform_legal", "samples_per_class": samples,
            "seed": seed, "entries": entries}


class PreflopTable:
    """Validated finite-sample estimates for all 169 suit-equivalence classes."""

    def __init__(self, payload: dict):
        if not isinstance(payload, dict):
            raise ValueError("preflop table must be a JSON object")
        if (payload.get("schema_version") != 1 or payload.get("algorithm") != ALGORITHM
                or payload.get("opponent_range") != "uniform_legal"
                or payload.get("treys") != "0.1.8"):
            raise ValueError("incompatible preflop table metadata")
        samples = payload.get("samples_per_class")
        _positive_int(samples, "samples_per_class")
        if type(payload.get("seed")) is not int:
            raise ValueError("preflop table needs an integer seed")
        entries = payload.get("entries")
        if not isinstance(entries, dict) or set(entries) != set(preflop_representatives()):
            raise ValueError("preflop table must contain exactly the 169 classes")
        self._entries = {}
        for key, counts in entries.items():
            if not isinstance(counts, dict) or set(counts) != {"wins", "ties", "losses"}:
                raise ValueError("invalid table outcome counts")
            estimate = EquityEstimate(**counts, method="preflop_lookup")
            if estimate.outcomes != samples:
                raise ValueError("table count total differs from samples_per_class")
            self._entries[key] = estimate

    @classmethod
    def load(cls, path: Path = DEFAULT_TABLE):
        return cls(json.loads(Path(path).read_text()))

    def lookup(self, hole) -> EquityEstimate:
        return self._entries[preflop_class(hole)]


@dataclass(frozen=True)
class EquityConfig:
    samples: int = 256
    seed: int = 20261027
    cache_size: int = 4096

    def __post_init__(self):
        _positive_int(self.samples, "samples")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if type(self.cache_size) is not int or self.cache_size < 0:
            raise ValueError("cache_size must be a nonnegative integer")


class EquityCalculator:
    def __init__(self, config: EquityConfig = EquityConfig(), table: PreflopTable | None = None):
        self._config = config
        self._table = table if table is not None else PreflopTable.load()
        self._cache = OrderedDict()

    @property
    def cache_entries(self) -> int:
        return len(self._cache)

    def evaluate(self, hole, board=()) -> EquityEstimate:
        hole, board = _cards(hole, board)
        if not board:
            return self._table.lookup(hole)
        key = (hole, board)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if len(board) == 5:
            estimate = exact_river(hole, board)
        else:
            estimate = monte_carlo(
                hole, board, samples=self._config.samples,
                seed=derived_seed(self._config.seed, ALGORITHM, "visible_state",
                                  hole, board, self._config.samples),
            )
        if self._config.cache_size:
            self._cache[key] = estimate
            if len(self._cache) > self._config.cache_size:
                self._cache.popitem(last=False)
        return estimate

    def from_observation(self, observation: Observation) -> EquityEstimate:
        return self.evaluate(observation.hole_cards, observation.board)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-table", help="generate all 169 preflop classes")
    build.add_argument("--samples", type=int, default=10_000)
    build.add_argument("--seed", type=int, default=20261027)
    build.add_argument("--output", type=Path, required=True)
    estimate = subparsers.add_parser("estimate", help="estimate uniform-range showdown equity")
    estimate.add_argument("--hole", nargs=2, required=True)
    estimate.add_argument("--board", nargs="*", default=[])
    estimate.add_argument("--samples", type=int, default=256)
    estimate.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    if args.command == "build-table":
        if args.output.exists():
            parser.error("output already exists; choose a new path")
        def progress(done, total):
            if done % 20 == 0 or done == total:
                print(f"Computed {done}/{total} classes", flush=True)
        payload = build_preflop_table(samples=args.samples, seed=args.seed, progress=progress)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(payload, indent=2) + "\n")
    else:
        result = EquityCalculator(EquityConfig(samples=args.samples, seed=args.seed)).evaluate(args.hole, args.board)
        print(json.dumps({**asdict(result), "equity": result.equity,
                          "standard_error": result.standard_error,
                          "opponent_range": "uniform_legal"}, indent=2))


if __name__ == "__main__":
    main()
