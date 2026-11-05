# Poker AI replay

Illustrative model: training seed 101; demo seed 17. Two seats start with 200 chips each, blinds 1/2. Seat 0 is BB, seat 1 is SB/button.

This offline transcript displays each acting player's own cards. Those cards are never passed to the other agent. The same deal is replayed with agents swapping seats. Hands were not selected for favorable results.

ML selects the highest estimated Q among legal actions. Q is an estimated terminal net return divided by 200, not a win probability or proof that the move is optimal. Equity is showdown pot share against a uniform legal hand.

## Deal 0, ML in seat 0

| Street | Actor / seat | Own cards | Board | Pot / call | Action | Explanation |
|---|---|---|---|---:|---|---|
| Preflop | Traditional / 1 | 4s Kc | — | 3 / 1 | call | call_pot_odds; equity 0.517, call threshold 0.280 |
| Preflop | ML / 0 | 5d Js | — | 4 / 0 | raise to 6 | highest legal Q = +0.116; equity 0.477 |
| Preflop | Traditional / 1 | 4s Kc | — | 8 / 4 | call | call_pot_odds; equity 0.517, call threshold 0.363 |
| Flop | ML / 0 | 5d Js | Ac Th Ks | 12 / 0 | bet to 12 | highest legal Q = +0.094; equity 0.432 |
| Flop | Traditional / 1 | 4s Kc | Ac Th Ks | 24 / 12 | raise to 30 | value_half_pot; equity 0.730, call threshold 0.363 |
| Flop | ML / 0 | 5d Js | Ac Th Ks | 54 / 18 | fold | highest legal Q = -0.105; equity 0.432 |

Final board: Ac Th Ks. ML net: **-18 chips**. Final stacks by seat: (182, 218).

## Deal 0, ML in seat 1

| Street | Actor / seat | Own cards | Board | Pot / call | Action | Explanation |
|---|---|---|---|---:|---|---|
| Preflop | ML / 1 | 4s Kc | — | 3 / 1 | call | highest legal Q = +0.127; equity 0.517 |
| Preflop | Traditional / 0 | 5d Js | — | 4 / 0 | check | check_free; equity 0.477, call threshold 0.030 |
| Flop | Traditional / 0 | 5d Js | Ac Th Ks | 4 / 0 | check | check_free; equity 0.432, call threshold 0.050 |
| Flop | ML / 1 | 4s Kc | Ac Th Ks | 4 / 0 | bet to 198 | highest legal Q = +1.138; equity 0.730 |
| Flop | Traditional / 0 | 5d Js | Ac Th Ks | 202 / 198 | fold | fold_below_call_threshold; equity 0.432, call threshold 0.545 |

Final board: Ac Th Ks. ML net: **+2 chips**. Final stacks by seat: (198, 202).

## Deal 1, ML in seat 0

| Street | Actor / seat | Own cards | Board | Pot / call | Action | Explanation |
|---|---|---|---|---:|---|---|
| Preflop | Traditional / 1 | 9h Th | — | 3 / 1 | call | call_pot_odds; equity 0.542, call threshold 0.280 |
| Preflop | ML / 0 | Qc Js | — | 4 / 0 | raise to 200 | highest legal Q = +0.622; equity 0.590 |
| Preflop | Traditional / 1 | 9h Th | — | 202 / 198 | call | call_pot_odds; equity 0.542, call threshold 0.525 |

Final board: 2c Jc 4d Ad Jh. ML net: **+200 chips**. Final stacks by seat: (400, 0).

## Deal 1, ML in seat 1

| Street | Actor / seat | Own cards | Board | Pot / call | Action | Explanation |
|---|---|---|---|---:|---|---|
| Preflop | ML / 1 | 9h Th | — | 3 / 1 | call | highest legal Q = +0.168; equity 0.542 |
| Preflop | Traditional / 0 | Qc Js | — | 4 / 0 | check | check_free; equity 0.590, call threshold 0.030 |
| Flop | Traditional / 0 | Qc Js | 2c Jc 4d | 4 / 0 | bet to 2 | value_half_pot; equity 0.816, call threshold 0.050 |
| Flop | ML / 1 | 9h Th | 2c Jc 4d | 6 / 2 | call | highest legal Q = +0.028; equity 0.348 |
| Turn | Traditional / 0 | Qc Js | 2c Jc 4d Ad | 8 / 0 | bet to 4 | value_half_pot; equity 0.764, call threshold 0.050 |
| Turn | ML / 1 | 9h Th | 2c Jc 4d Ad | 12 / 4 | call | highest legal Q = -0.015; equity 0.297 |
| River | Traditional / 0 | Qc Js | 2c Jc 4d Ad Jh | 16 / 0 | bet to 16 | value_pot; equity 0.960, call threshold 0.020 |
| River | ML / 1 | 9h Th | 2c Jc 4d Ad Jh | 32 / 16 | fold | highest legal Q = -0.038; equity 0.271 |

Final board: 2c Jc 4d Ad Jh. ML net: **-8 chips**. Final stacks by seat: (208, 192).
