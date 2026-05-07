# Reinforcement Clanker — Game Plan

## Agent Architecture

- **`Clanker`** (abstract base) — shared training loop (`play`, `train`, `evaluate`), epsilon-greedy `act`, epsilon decay
- **`TableClanker`** — Q-table agent for simple games. Save/load via JSON.
- **`DeepClanker`** — DQN agent with replay buffer, target network, HoloNet neural network. Save/load via `torch.save`/`torch.load`.
- **`HoloNet`** — PyTorch `nn.Module`, configurable hidden layers, used by `DeepClanker`

All games are two-player. Agents train via self-play. States are always from the current player's perspective (1 = me, -1 = opponent, 0 = empty). Negamax negation applied in `learn` for zero-sum correctness.

---

## Games

| # | Game | Agent | Status |
|---|---|---|---|
| 1 | Tic Tac Toe | `TableClanker` | ✅ Complete — converges to 100% draws vs self, zero losses vs random |
| 2 | Ultimate Tic Tac Toe | `DeepClanker` | 🔜 Next — environment not yet built |
| 3 | Mancala (Kalah, 6-pit) | `DeepClanker` | Not started |
| 4 | Connect Four | `DeepClanker` | Not started |
| 5 | Gomoku (5-in-a-row, 15×15) | `DeepClanker` | Not started |
| 6 | Othello | `DeepClanker` | Not started |
| 7 | Checkers | `DeepClanker` / deeper RL | Not started |
| 8 | Chess | Deep RL + MCTS | Not started |

Rock Paper Scissors was considered and rejected — no board state, optimal policy is a mixed Nash equilibrium that Q-learning can't represent.

---

## Ultimate Tic Tac Toe — Design Notes

- **State:** flat tuple of all 81 cells (9 local boards × 9 cells) + 1 value for the active board index = 82 inputs to the network. Active board is in the state so the agent implicitly learns to control where it sends the opponent.
- **Actions:** 0–8 (cell within the active local board). `get_legal_actions` returns empty cells within the constrained board, or all empty cells if the target board is already won/full.
- **Rewards:** sparse — only the meta-game outcome (1 win, -1 loss, 0 draw). No intermediate reward for winning local boards.
- **Perspective flipping:** same as TTT — negate all non-zero board values when switching players.

---

## Known Fragilities

- **`TableClanker.load`** parses state tuples from Python's default `str(tuple)` format (e.g. `"(0, 1, -1)"`). Relies on `', '` as separator and single-digit integers. Works for all planned games but would break if states ever contain multi-digit values or different formatting.

---

## Web Frontend Plan

- **Framework:** Flask (full UI control). Streamlit is an alternative if minimal frontend work is preferred — has free hosting via Streamlit Cloud.
- **Hosting:** Render or Railway (free tier, easy Flask deploys).
- **Model upload feature:** accept JSON Q-tables (not pickle) to avoid arbitrary code execution vulnerability. `TableClanker` save/load already uses JSON for this reason.
- **Build after:** training pipeline is proven end-to-end on at least one DQN game.

---

## Next Steps

1. Build `ultimate_ttt_env.py` implementing `GeneralEnvironment`
2. Write training script for Ultimate TTT using `DeepClanker`
3. Verify agent learns meaningful play, evaluate vs random and vs self
