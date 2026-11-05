from dataclasses import FrozenInstanceError, fields
import random
import unittest

from poker_ai.agents import PassiveAgent, RandomAgent
from poker_ai.environment import Action, FULL_DECK, GameConfig, Observation, PokerHand, shuffled_deck
from poker_ai.match import play_hand


def fixed_deck(prefix):
    cards = prefix.split()
    return tuple(cards + [c for c in FULL_DECK if c not in cards])


class EnvironmentTests(unittest.TestCase):
    def test_heads_up_order_every_street(self):
        hand = PokerHand(shuffled_deck(1))
        for street in range(4):
            seats = (1, 0) if street == 0 else (0, 1)
            for seat in seats:
                obs = hand.observe()
                self.assertEqual((obs.street, obs.seat), (street, seat))
                self.assertEqual(obs.is_button, seat == 1)
                self.assertEqual(len(obs.board), (0, 3, 4, 5)[street])
                hand.step(Action.CHECK_CALL)
        self.assertTrue(hand.terminal)

    def test_fold_payout_and_terminal_access(self):
        hand = PokerHand(FULL_DECK)
        with self.assertRaises(ValueError):
            hand.result()
        hand.step(Action.FOLD)
        self.assertEqual(hand.result().final_stacks, (201, 199))
        self.assertEqual(hand.result().net_chips, (1, -1))
        with self.assertRaises(ValueError):
            hand.observe()
        with self.assertRaises(ValueError):
            hand.step(Action.CHECK_CALL)

    def test_raise_to_and_call_amounts(self):
        hand = PokerHand(FULL_DECK)
        hand.step(Action.MIN_RAISE)
        obs = hand.observe()
        self.assertEqual(obs.street_bets, (2, 4))
        self.assertEqual(obs.call_amount, 2)
        self.assertEqual(obs.pot, 6)
        self.assertEqual(obs.history[-1].chips_added, 3)
        self.assertEqual(obs.history[-1].raise_to, 4)
        hand.step(Action.CHECK_CALL)
        self.assertEqual(hand.observe().pot, 8)
        self.assertEqual(hand.observe().street_bets, (0, 0))

    def test_half_pot_sizing_and_deduplication(self):
        hand = PokerHand(FULL_DECK)
        obs = hand.observe()
        self.assertEqual([(a.slot, a.raise_to) for a in obs.legal_actions], [
            (Action.FOLD, None), (Action.CHECK_CALL, None),
            (Action.MIN_RAISE, 4), (Action.POT, 6), (Action.ALL_IN, 200),
        ])
        hand.step(Action.MIN_RAISE)
        hand.step(Action.CHECK_CALL)
        # Flop pot is 8: half-pot bet of 4 differs from the minimum of 2.
        self.assertEqual(hand.observe().pot, 8)
        hand.step(Action.HALF_POT)
        self.assertEqual(hand.observe().street_bets, (4, 0))
        # Facing 4 into 8: P=12, C=4, M=4; half-pot raise-to is 12.
        hand.step(Action.HALF_POT)
        self.assertEqual(hand.observe().street_bets, (4, 12))

    def test_free_check_cannot_fold_and_invalid_actions_do_not_mutate(self):
        hand = PokerHand(FULL_DECK)
        hand.step(Action.CHECK_CALL)
        before = hand.observe()
        self.assertFalse(before.action_mask[Action.FOLD])
        for bad in (Action.FOLD, -1, 6, "1", True, 1.0):
            with self.assertRaises(ValueError):
                hand.step(bad)
            self.assertEqual(before, hand.observe())

    def test_short_allin_does_not_reopen_betting(self):
        hand = PokerHand(FULL_DECK, GameConfig((5, 200)))
        hand.step(Action.MIN_RAISE)  # SB raises to 4.
        obs = hand.observe()
        self.assertEqual([a.raise_to for a in obs.legal_actions if a.raise_to], [5])
        hand.step(Action.MIN_RAISE)  # BB can only raise all-in to 5 (< full raise).
        obs = hand.observe()
        self.assertEqual(obs.call_amount, 1)
        self.assertEqual([a.slot for a in obs.legal_actions], [Action.FOLD, Action.CHECK_CALL])
        hand.step(Action.CHECK_CALL)
        self.assertTrue(hand.terminal)
        self.assertEqual(sum(hand.result().final_stacks), 205)

    def test_unequal_allin_refund_and_payout(self):
        deck = fixed_deck("As Ks Ah Kh 2s 3c 4d 7h 8s 9c Ts Jd")
        hand = PokerHand(deck, GameConfig((80, 200)))
        hand.step(Action.ALL_IN)
        obs = hand.observe()
        self.assertEqual(obs.call_amount, 78)
        self.assertFalse(any(a.raise_to for a in obs.legal_actions))
        hand.step(Action.CHECK_CALL)
        self.assertEqual(hand.result().final_stacks, (160, 120))
        self.assertEqual(hand.result().net_chips, (80, -80))

    def test_forced_board_tie(self):
        deck = fixed_deck("2c 4c 3d 5d 6c As Ks Qs 7c Js 8c Ts")
        result = play_hand(PokerHand(deck), (PassiveAgent(), PassiveAgent()))
        self.assertEqual(result.board, ("As", "Ks", "Qs", "Js", "Ts"))
        self.assertEqual(result.final_stacks, (200, 200))
        self.assertEqual(result.net_chips, (0, 0))

    def test_low_stacks_automatic_runout(self):
        for stacks in ((1, 1), (2, 1), (1, 200), (200, 1)):
            result = play_hand(PokerHand(FULL_DECK, GameConfig(stacks)),
                               (PassiveAgent(), PassiveAgent()))
            self.assertEqual(sum(result.final_stacks), sum(stacks))
            self.assertEqual(sum(result.net_chips), 0)

    def test_deck_validation_and_no_duplicates(self):
        for bad in (FULL_DECK[:-1], FULL_DECK + ("As",), ("As",) * 52):
            with self.assertRaises(ValueError):
                PokerHand(bad)
        hand = PokerHand(FULL_DECK)
        # White-box engine invariant, separate from the agent API.
        holes = [repr(c) for seat in hand._state.hole_cards for c in seat]
        result = play_hand(hand, (PassiveAgent(), PassiveAgent()))
        dealt = holes + list(result.board) + [repr(c) for c in hand._state.burn_cards]
        self.assertEqual(len(dealt), len(set(dealt)))

    def test_bad_config(self):
        for stacks in ((0, 200), (-1, 2), (2,), [200, 200], (True, 200)):
            with self.assertRaises(ValueError):
                GameConfig(stacks)
        with self.assertRaises(ValueError):
            GameConfig(small_blind=3)

    def test_preflop_hidden_card_noninterference(self):
        deck = list(FULL_DECK)
        other = deck.copy()
        other[0], other[20] = other[20], other[0]  # Opponent card.
        other[2], other[21] = other[21], other[2]
        other[5], other[30] = other[30], other[5]  # Future board.
        self.assertEqual(PokerHand(tuple(deck)).observe(), PokerHand(tuple(other)).observe())

    def test_postflop_hidden_card_noninterference(self):
        deck = list(FULL_DECK)
        other = deck.copy()
        other[1], other[20] = other[20], other[1]  # BB observes; SB cards hidden.
        other[3], other[21] = other[21], other[3]
        other[9], other[30] = other[30], other[9]  # Unrevealed turn.
        hands = [PokerHand(tuple(d)) for d in (deck, other)]
        for hand in hands:
            hand.step(Action.CHECK_CALL)
            hand.step(Action.CHECK_CALL)
        self.assertEqual(hands[0].observe(), hands[1].observe())

    def test_immutable_observation_and_explicit_field_boundary(self):
        obs = PokerHand(FULL_DECK).observe()
        self.assertEqual({f.name for f in fields(Observation)}, {
            "seat", "hole_cards", "board", "street", "is_button", "pot",
            "call_amount", "stacks", "street_bets", "legal_actions", "history",
        })
        with self.assertRaises(FrozenInstanceError):
            obs.pot = 999
        with self.assertRaises(FrozenInstanceError):
            obs.legal_actions[0].raise_to = 999
        with self.assertRaises(TypeError):
            obs.stacks[0] = 999

    def test_fixed_deck_and_separate_randomness(self):
        deck = shuffled_deck(23)
        first = play_hand(PokerHand(deck), (RandomAgent(7), RandomAgent(8)))
        random.seed(999)
        for _ in range(1000):
            random.random()
        second = play_hand(PokerHand(deck), (RandomAgent(7), RandomAgent(8)))
        self.assertEqual(first, second)
        self.assertEqual(deck, shuffled_deck(23))

    def test_2000_hands_conservation_and_legal_masks(self):
        for seed in range(1000):
            for agents in ((PassiveAgent(), PassiveAgent()), (RandomAgent(seed), RandomAgent(seed + 1000))):
                hand = PokerHand(shuffled_deck(seed))
                while not hand.terminal:
                    obs = hand.observe()
                    raises = [a.raise_to for a in obs.legal_actions if a.raise_to is not None]
                    self.assertEqual(len(raises), len(set(raises)))
                    self.assertEqual(sum(obs.action_mask), len(obs.legal_actions))
                    self.assertEqual(sum(obs.stacks) + obs.pot, 400)
                    slot = agents[obs.seat].act(obs)
                    self.assertTrue(obs.action_mask[slot])
                    hand.step(slot)
                self.assertEqual(sum(hand.result().final_stacks), 400)
                self.assertEqual(sum(hand.result().net_chips), 0)


if __name__ == "__main__":
    unittest.main()
