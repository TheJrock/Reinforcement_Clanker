# Reinforcement Clanker — Game Plan

## Agent Architecture

- **`Clanker`** (abstract base) — shared training loop (`play`, `train`, `evaluate`), epsilon-greedy `act`, epsilon decay
- **`TableClanker`** — Q-table agent for simple games. Save/load via JSON.
- **`DeepClanker`** — DQN agent with replay buffer, target network, HoloNet neural network. Save/load via `torch.save`/`torch.load`.
- **`HoloNet`** — PyTorch `nn.Module`, configurable hidden layers, used by `DeepClanker`
- **Future:** AlphaZero-style agent (MCTS + neural network) for Chess and beyond. PPO-based agent for continuous state/action spaces.

All games are two-player. Agents train via self-play. States are always from the current player's perspective. All values are non-negative — position encodes ownership (my pits/pieces always in consistent indices, opponent's always in consistent indices), so no negative inputs to poison ReLU activations. Negamax negation applied in `learn` for zero-sum correctness.

---

## Phase 1 — Discrete Board Games (DQN)

| # | Game | Agent | Status |
|---|---|---|---|
| 1 | Tic Tac Toe | `TableClanker` | ✅ Complete — converges to 100% draws vs self, zero losses vs random |
| 2 | Ultimate Tic Tac Toe | `DeepClanker` | ✅ Complete — 85.8% win rate vs random after 100k episodes (7.3% losses, 6.9% draws). Outperformed Mancala despite greater complexity. Previous failures were entirely due to implementation bugs, not game difficulty. |
| 3 | Mancala (Kalah, 6-pit) | `DeepClanker` | ✅ Complete — 73.6% win rate vs random after 100k episodes. 1000/1000 draws vs self (confirms correct state normalization and Nash convergence). Generalization problem confirmed: less-trained agents beat more-trained due to determinism exploitation — expected, addressed in Phase 1.5. |
| 4 | Connect Four | `DeepClanker` | ✅ Complete — 97.1% win rate vs random after 100k episodes (2.9% losses, 0% draws). Strong result; DQN ceiling estimated at ~98-99%. AlphaZero revisit planned in Phase 2. |
| 5 | Gomoku (5-in-a-row, 15×15) | `DeepClanker` | Not started |
| 6 | Othello | `DeepClanker` | Not started |
| 7 | Checkers | `DeepClanker` | Not started |

Rock Paper Scissors was considered and rejected — no board state, optimal policy is a mixed Nash equilibrium that Q-learning can't represent.

---

## Phase 1.5 — Generational Training (Revisit Phase 1 Games)

DQN agents trained against random opponents converge to brittle, deterministic policies that are easily exploited by any non-random opponent — including a randomly initialized network. Generational training addresses this by progressively improving the quality of the opponent over time, producing robust agents that can handle strategic play.

**Approach:** load the Phase 1 trained model as the starting point → freeze a copy as the opponent → reset epsilon to ~0.3–0.5 → train until the main agent clearly surpasses the frozen copy → save as next generation → repeat. Starting both agents from the same weights gives a roughly 50/50 matchup at the start of each generation, ensuring positive reward signal from the beginning. Each generation only needs to beat the previous one.

**Alternative:** use the frozen target network as the opponent during training — it improves gradually as the main network trains, providing a stable but increasingly capable opponent without explicit checkpoint management.

**Risk — strategy collapse:** generational training can converge along a single strategic trajectory. Each generation only needs to beat the previous one, so the lineage might collectively miss strategies that don't naturally emerge from that path. A later generation might also "forget" counters that an earlier generation knew, creating rock-paper-scissors cycles where no single strategy dominates.

**League play (preferred mitigation):** rather than training each generation against only the most recent checkpoint, maintain a roster of all past checkpoints and sample opponents from the full history. Each generation must play well against everyone, not just the latest — this prevents the agent from specializing narrowly and keeps the strategy space diverse. Used by AlphaStar. Adds checkpoint management overhead but produces more robust agents.

**Note on TTT:** the TableClanker opponent is perfect — optimal play always draws, so no wins are achievable. The success metric is "never loses" rather than "achieves wins." Consider using a draw reward of 0.5 instead of 0 to give the agent a positive signal to climb toward, since draw (0) and unexplored states (Q-values near 0) are otherwise indistinguishable to the network.

All Phase 1 games will be revisited in this phase after the full Phase 1 sweep is complete.

| # | Game | Status |
|---|---|---|
| 1 | Tic Tac Toe | Not started |
| 2 | Ultimate Tic Tac Toe | Not started — also benefits from state encoding fix (see post-mortem) |
| 3 | Mancala | Not started |
| 4 | Connect Four | Not started |
| 5 | Gomoku | Not started |
| 6 | Othello | Not started |
| 7 | Checkers | Not started |

---

## Phase 2 — AlphaZero (MCTS + Neural Network)

| # | Game | Agent | Notes |
|---|---|---|---|
| 8 | Mancala | AlphaZero-style | Starting point for Phase 2 — familiar game, small state space, good for validating the MCTS + neural network pipeline before scaling up. Kalah is solved (P1 wins), so perfect play is verifiable. |
| 9 | Connect Four | AlphaZero-style | Solved (P1 wins). DQN plateaus around 98-99% vs random — AlphaZero is needed to close the gap to provably perfect play. Direct before/after comparison validates Phase 2. |
| 10 | Ultimate Tic Tac Toe | AlphaZero-style | Not fully solved. Complexity exceeds what vanilla DQN can handle well long-term. |
| 11 | Gomoku (15×15) | AlphaZero-style | Large state space (225 cells, high branching factor) likely too complex for DQN to achieve strong play. MCTS is the natural fit. Standard Gomoku is solved (P1 wins) but the proof is deep — empirical strength is the benchmark. |
| 12 | Othello | AlphaZero-style | Not fully solved on 8×8. Complex strategic depth rewards deep lookahead that MCTS handles naturally. DQN will learn reasonable play but won't approach optimal. |
| 13 | Checkers | AlphaZero-style | Solved by Chinook in 2007 — perfect play always draws. First major board game solved by computer; achieving near-perfect play is a meaningful milestone. DQN will fall well short of optimal. |
| 14 | Chess | AlphaZero-style | MCTS + neural network. Standard DQN cannot handle this state space. Long-term goal. |

AlphaZero approach: neural network guides Monte Carlo Tree Search (look-ahead), trained via self-play. Computationally expensive but necessary for Chess-level complexity. Mancala serves as the entry point to validate the pipeline before tackling larger state spaces.

---

## Phase 3 — Partial Observability (Card Games)

Partial observability means the agent cannot see the full game state (e.g. opponent's hand). Requires recurrent networks (LSTM/GRU) to maintain a belief state over time — the agent reasons about what it *doesn't* know. Multi-player games (3+ players) also break the two-player self-play and negamax assumptions — requires multi-agent RL approaches (e.g. independent Q-learning or centralized training with decentralized execution).

| # | Game | Players | Notes |
|---|---|---|---|
| 10 | Blackjack | 2 (vs dealer) | Classic RL textbook example. Simple partial observability, well-understood optimal policy (good benchmark). Entry point to this phase. |
| 11 | Gin Rummy | 2 | Introduces hand management and partial observability in a familiar 2-player format before going multi-player. |
| 12 | Oh Hell (Wizard) | 3–6 | Trick-taking + exact bidding + multi-player. Most direct stepping stone to Skull King — same core mechanics, much simpler card hierarchy. Critical bridge game. |
| 13 | Poker (Texas Hold'em) | 2–9 | Adds bluffing, betting, and opponent modeling on top of partial observability. Active research area. |
| 14 | Skull King | 2–8 | Long-term goal. Family favorite. Combines trick-taking, exact bidding, multi-player, partial observability, and complex card hierarchy (Skull King > Pirates > Mermaids > Black suit > colored suits, with special capture interactions). Feasible after Oh Hell is working. |

---

## Phase 4 — Continuous Control (PPO)

Transition from discrete board games to continuous state and action spaces. Uses PPO (Proximal Policy Optimization) — more stable than DQN for continuous domains, works for both discrete and continuous actions.

### Stepping Stones (Gymnasium environments)

| # | Environment | Purpose |
|---|---|---|
| 1 | CartPole | "Hello world" of RL. Continuous state, discrete actions. |
| 2 | Pendulum | First continuous actions. Classic control problem. |
| 3 | LunarLander | Continuous or discrete. Good intermediate challenge. |
| 4 | BipedalWalker | Harder continuous control. Good pre-robotics challenge. |

### Long-term Goal — Robot Arena

Inspired by a VEX Robotics-style competition: two robots collect rings and place them on poles in an arena before time runs out.

- **Short term:** simplified 2D grid simulation of the arena as a custom Gymnasium environment
- **Long term:** full continuous simulation with robot position/orientation, ring locations, opponent robot
- **Agent:** PPO with continuous state (robot pose, ring positions, time remaining) and continuous actions (motor speeds / movement commands)

This is the most ambitious goal of the project and will require significant simulation work before training can begin.

---

## Ultimate Tic Tac Toe — Post-mortem Notes

Vanilla DQN with self-play failed to converge on UTTT despite extensive tuning. Key findings:

- **Root cause:** UTTT is too complex for vanilla DQN self-play. DQN was designed for single-player games (Atari). Two-player self-play creates non-stationarity far beyond what the target network can handle — both the environment and the opponent change simultaneously.
- **Training against random is insufficient:** Random opponents generate low-quality, inconsistent experiences. The agent never learns to handle coherent strategic threats.
- **Things tried that didn't fix it:** slower epsilon decay, larger replay buffer (100k→500k), lower learning rate, higher target update frequency, only learning on self's turns, next_state perspective fixes.
- **Resolution:** all bugs were fixed (turn tracking via `current_player`, per-transition negate in replay buffer, non-terminal learning, 0/0.5/1 state encoding). Result: 85.8% win rate vs random at 100k episodes. DQN handles UTTT well with a correct implementation — the prior failures were entirely due to bugs, not game complexity.
- **Remaining ideas for Phase 1.5 revisit:**
  - **Frozen target network as opponent:** use `target_network` to select opponent moves during training — provides a stable, consistent opponent that improves gradually. This is essentially what AlphaZero does.
  - **Generational training (league play):** train against random → save checkpoint → train against checkpoint → repeat. Each generation fights a meaningful opponent. Used by OpenAI Five and AlphaStar.
- **State:** flat tuple of 91 values (81 cells + 9 meta-board + 1 active board index). Actions: 0–80 flat integers (`board_index * 9 + cell_index`).
- **State encoding issue (unresolved):** The current encoding uses 1 = my piece, -1 = opponent's piece, 0 = empty. Negative inputs are zeroed by ReLU activations, meaning opponent piece information is systematically suppressed in every hidden layer. This likely contributed to poor learning. When revisiting UTTT, consider: (a) two binary channels — one plane for my pieces, one for opponent's pieces, all values 0/1; (b) two values per cell `[1,0]` / `[0,1]` / `[0,0]`; or (c) shifting to [0, 1] range (map -1→0, 0→0.5, 1→1). Option (a) is what AlphaZero uses and is the cleanest.

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

## Friend/Family Agent Challenge

The goal is to let others train their own agents and compete in a round-robin tournament. The challenge is fun but requires upfront design decisions to keep it fair.

### Fairness Concerns

- **Compute disparity** — the biggest risk. Someone with a GPU or cloud instance can far outrun someone on a laptop CPU. Set a stated compute budget (e.g. "max 2 hours on consumer hardware") so the competition rewards agent design, not hardware access.
- **Training against a known opponent** — if participants know whose agent they'll face, they can overfit to exploit that specific agent's weaknesses rather than learning generally strong play. Use a round-robin format and don't reveal matchups until submission closes.
- **Non-RL approaches** — nothing in a callable interface prevents someone from submitting a minimax or MCTS agent with a handcrafted evaluation function. For games like UTTT, a well-tuned minimax is a fundamentally different class of solver. Decide upfront whether that's allowed.
- **Evaluation variance** — stochastic policies need many games per matchup to produce stable rankings. Run at minimum 100 games per pair, alternating who plays first, to average out first-mover advantage.
- **Verification** — hard to enforce, but be aware that someone could hardcode a lookup table or rules engine behind the `.pth` interface. Inspect submitted code if it matters.

### Implementation Steps

1. **Define a standard `Agent` interface** — a base class (or protocol) that everyone must subclass. At minimum: `__init__(self, game)` and `act(self, state) -> action`. This is the contract the tournament runner depends on; it should not expose training internals.
2. **Build an eval-only `Clanker` wrapper** — a stripped-down class that loads a `.pth` file and exposes only `act`. No optimizer, no replay buffer, no `train` method. This is what gets shared publicly so participants can test their agents against yours without seeing your training code.
3. **Standardize input/output format** — document exactly what `state` looks like for each game (shape, dtype, encoding) and what a valid `action` is. Without this, submissions will be incompatible.
4. **Build a tournament runner** — a script that accepts a directory of submitted agents, runs every pair round-robin for N games (alternating sides), and produces a win/loss/draw matrix and Elo-style ranking. Keep it deterministic (seed RNG) so results are reproducible.
5. **Add a submission validator** — before the tournament, run each submitted agent against a known-weak random agent to confirm it loads, produces valid actions, and doesn't crash or timeout. Reject invalid submissions early.
6. **Set and document rules clearly before opening submissions:**
   - Compute budget (hours, hardware class)
   - Whether non-RL approaches are allowed
   - Which games are in scope
   - Submission format (`.pth` + agent class file)
   - Deadline and how matchups are revealed
7. **Host results** — after the tournament, publish the win/loss matrix and Elo rankings on the web frontend. A simple static page or Streamlit app is enough.

---

## Current Status & Next Steps

1. ✅ Tic Tac Toe — complete
2. ✅ Ultimate Tic Tac Toe — complete
3. ✅ Mancala — complete
4. ✅ Connect Four — complete
5. 🔜 Gomoku — next. Continue Phase 1 through Checkers (all trained against random)
6. Phase 1.5 — revisit all Phase 1 games with generational training
7. Build web frontend after Phase 1.5 is proven on at least one game
8. Phase 2 (AlphaZero: UTTT + Chess) after Phase 1.5
9. Phase 3 (card games) and Phase 4 (continuous control) are long-term goals
