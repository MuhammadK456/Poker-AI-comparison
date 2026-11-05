"""A small, explicit information boundary around PokerKit.

Physical seats stay fixed: seat 0 is BB and seat 1 is SB/button. A paired
match swaps *agents*, not cards or seats. Amounts are integer chips.
"""

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
import random

from pokerkit import Automation, Card, Mode, NoLimitTexasHoldem


FULL_DECK = tuple(rank + suit for rank in "23456789TJQKA" for suit in "cdhs")


class Action(IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    MIN_RAISE = 2
    HALF_POT = 3
    POT = 4
    ALL_IN = 5


@dataclass(frozen=True)
class GameConfig:
    starting_stacks: tuple[int, int] = (200, 200)
    small_blind: int = 1
    big_blind: int = 2

    def __post_init__(self):
        if (not isinstance(self.starting_stacks, tuple)
                or len(self.starting_stacks) != 2
                or any(type(x) is not int or x <= 0 for x in self.starting_stacks)):
            raise ValueError("starting_stacks must be a tuple of two positive integers")
        if (type(self.small_blind) is not int or type(self.big_blind) is not int
                or not 0 < self.small_blind <= self.big_blind):
            raise ValueError("blinds must be positive integers with SB <= BB")


@dataclass(frozen=True)
class LegalAction:
    slot: Action
    # Total contribution on this street for raises; None for fold/check/call.
    raise_to: int | None = None


@dataclass(frozen=True)
class PublicAction:
    seat: int
    street: int
    kind: str
    chips_added: int
    raise_to: int | None
    slot: Action


@dataclass(frozen=True)
class Observation:
    seat: int
    hole_cards: tuple[str, ...]
    board: tuple[str, ...]
    street: int
    is_button: bool
    pot: int
    call_amount: int
    stacks: tuple[int, int]
    street_bets: tuple[int, int]
    legal_actions: tuple[LegalAction, ...]
    history: tuple[PublicAction, ...]

    @property
    def action_mask(self) -> tuple[bool, ...]:
        slots = {action.slot for action in self.legal_actions}
        return tuple(slot in slots for slot in Action)


@dataclass(frozen=True)
class HandResult:
    final_stacks: tuple[int, int]
    net_chips: tuple[int, int]
    board: tuple[str, ...]
    history: tuple[PublicAction, ...]


def shuffled_deck(seed: int) -> tuple[str, ...]:
    deck = list(FULL_DECK)
    random.Random(seed).shuffle(deck)
    return tuple(deck)


class PokerHand:
    """Owns private state. Pass observations, never this object, to agents.

    This boundary prevents accidental leakage; Python private attributes are
    not a security sandbox for malicious third-party agents.
    """

    def __init__(self, deck: tuple[str, ...], config: GameConfig = GameConfig()):
        if len(deck) != 52 or set(deck) != set(FULL_DECK):
            raise ValueError("deck must be an exact permutation of the 52 cards")
        self.config = config
        self._history: list[PublicAction] = []
        manual = {Automation.HOLE_DEALING, Automation.BOARD_DEALING,
                  Automation.CARD_BURNING}
        self._state = NoLimitTexasHoldem.create_state(
            tuple(a for a in Automation if a not in manual), True, 0,
            (config.small_blind, config.big_blind), config.big_blind,
            config.starting_stacks, 2, mode=Mode.CASH_GAME,
        )
        # No card dealing has happened yet. Replace the shuffled engine deck
        # before any consumption, so policy randomness cannot affect dealing.
        self._state.deck_cards = deque(Card.parse("".join(deck)))
        self._advance_cards()

    @property
    def terminal(self) -> bool:
        return not self._state.status

    def _advance_cards(self) -> None:
        s = self._state
        while s.status and s.actor_index is None:
            if s.can_burn_card():
                s.burn_card()
            elif s.can_deal_hole():
                s.deal_hole()
            elif s.can_deal_board():
                s.deal_board()
            else:
                raise RuntimeError("engine stalled outside a player decision")
        self._check_chips()

    def _check_chips(self) -> None:
        s = self._state
        if min(s.stacks) < 0 or sum(s.stacks) + s.total_pot_amount != sum(self.config.starting_stacks):
            raise RuntimeError("chip accounting invariant failed")
        if self.terminal and (s.total_pot_amount or sum(s.payoffs) != 0):
            raise RuntimeError("terminal settlement invariant failed")

    def _legal_actions(self) -> tuple[LegalAction, ...]:
        s = self._state
        actions = []
        call = s.checking_or_calling_amount
        if call is None:
            raise RuntimeError("no call/check at a decision")
        if call > 0 and s.can_fold():
            actions.append(LegalAction(Action.FOLD))
        actions.append(LegalAction(Action.CHECK_CALL))
        if s.can_complete_bet_or_raise_to():
            low = s.min_completion_betting_or_raising_to_amount
            high = s.max_completion_betting_or_raising_to_amount
            assert low is not None and high is not None
            pot_after_call = s.total_pot_amount + call
            # Integer half-up rounding avoids Python's ties-to-even round().
            candidates = (
                (Action.MIN_RAISE, low),
                (Action.HALF_POT, max(s.bets) + (pot_after_call + 1) // 2),
                (Action.POT, max(s.bets) + pot_after_call),
                (Action.ALL_IN, high),
            )
            seen = set()
            for slot, amount in candidates:
                amount = max(low, min(high, amount))
                # Fixed priority: minimum, half-pot, pot, all-in.
                if amount not in seen and s.can_complete_bet_or_raise_to(amount):
                    actions.append(LegalAction(slot, amount))
                    seen.add(amount)
        return tuple(actions)

    def observe(self) -> Observation:
        if self.terminal:
            raise ValueError("terminal hand has no acting-player observation")
        s = self._state
        seat = s.actor_index
        assert seat is not None and s.street_index is not None
        return Observation(
            seat=seat, hole_cards=tuple(map(repr, s.hole_cards[seat])),
            board=tuple(map(repr, s.get_board_cards(0))), street=s.street_index,
            is_button=seat == 1, pot=s.total_pot_amount,
            call_amount=s.checking_or_calling_amount,
            stacks=tuple(s.stacks), street_bets=tuple(s.bets),
            legal_actions=self._legal_actions(), history=tuple(self._history),
        )

    def step(self, slot: Action | int) -> None:
        obs = self.observe()
        if type(slot) not in (int, Action):
            raise ValueError("action must be an integer slot or Action")
        selected = next((a for a in obs.legal_actions if a.slot == slot), None)
        if selected is None:
            raise ValueError(f"illegal action slot: {slot}")
        s = self._state
        if selected.slot == Action.FOLD:
            kind, added = "fold", 0
            s.fold()
        elif selected.slot == Action.CHECK_CALL:
            kind = "call" if obs.call_amount else "check"
            added = obs.call_amount
            s.check_or_call()
        else:
            kind = "raise" if max(obs.street_bets) else "bet"
            added = selected.raise_to - obs.street_bets[obs.seat]
            s.complete_bet_or_raise_to(selected.raise_to)
        self._history.append(PublicAction(
            obs.seat, obs.street, kind, added, selected.raise_to, selected.slot,
        ))
        self._advance_cards()

    def result(self) -> HandResult:
        if not self.terminal:
            raise ValueError("hand is not finished")
        s = self._state
        return HandResult(tuple(s.stacks), tuple(s.payoffs),
                          tuple(map(repr, s.get_board_cards(0))), tuple(self._history))
