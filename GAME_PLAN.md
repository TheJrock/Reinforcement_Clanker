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
| 2 | Ultimate Tic Tac Toe | `DeepClanker` | ✅ Complete — best result: 85.8% win rate vs random after 100k episodes with n=1 (7.3% losses, 6.9% draws). Retrain with fixed n=5: 77.7% wins, 11.9% losses, 10.4% draws — draws doubled, overall weaker. n=5 lookahead is counterproductive for UTTT against random: further-ahead bootstrapping introduces variance without reliably pointing toward the terminal. n=1 remains the best Phase 1 result and the recommended starting point for Phase 1.5. Previous failures were entirely due to implementation bugs, not game difficulty. |
| 3 | Mancala (Kalah, 6-pit) | `DeepClanker` | ✅ Complete — three training runs attempted. Original (buggy retroactive reward, n=1): 73.6% at 100k. Retrain with fix + n=3: 68.0% at 100k. Retrain with fix + n=1: 68.9% at 100k. All fixed runs converge to ~68-69%, suggesting the original 73.6% was a lucky outlier and the true DQN ceiling is ~68-70%. Retroactive fix had negligible impact (opponent rarely ends game on their move in Mancala). n_steps=3 suboptimal — Mancala's extra turns mean 3 agent-moves often fall within one turn sequence with no cross-turn lookahead. Determinism exploitation confirmed: agent peaks at 75.9% after only 100 episodes then degrades to 68.9% at 100k as epsilon decays and greedy policy gets punished. Addressed in Phase 1.5. 1000/1000 draws vs self (confirms correct state normalization and Nash convergence). |
| 4 | Connect Four | `DeepClanker` | ✅ Complete — 97.1% win rate vs random after 100k episodes (2.9% losses, 0% draws). Strong result; DQN ceiling estimated at ~98-99%. AlphaZero revisit planned in Phase 2. |
| 5 | Gomoku (5-in-a-row, 15×15) | `DeepClanker` | ✅ Complete — 99.9% win rate vs random after 10k episodes (0 draws). Ceiling against random hit by 100 episodes (98%); 10k adds marginal gain. Environment refactored to numpy for ~4x eval speedup. 100k run skipped — unnecessary. |
| 6 | Othello | `DeepClanker` | ✅ Complete — 67.0% win rate vs random after 100k episodes (fixed run). Earlier runs showing degradation (57.1% at 1k → 42.0% at 10k) were corrupted by the `_flush_n_step` compound_sign bug, which wrongly negated terminal rewards at odd buffer positions — teaching the agent that winning was bad for ~1/3 of terminal transitions. Fixed run with n=3: chaotic early (42.8% at 1ep, 33.8% at 10ep) but steadily improves to 67.0% at 100k. DQN is genuinely learning Othello strategy; the earlier "DQN insufficient" conclusion was premature. Determinism exploitation against random is still a ceiling — Phase 1.5 (generational training) is the right next step. |
| 7 | Checkers | `DeepClanker` | ✅ Complete — 87.4% win rate vs random after 100k episodes (6.8% losses, 5.8% draws). Monotonic progression: 48.0% → 50.8% → 53.8% → 70.4% → 79.5% → 87.4% (1/10/100/1k/10k/100k episodes). Strong result with clear late-game acceleration. Three critical bugs found and fixed: (1) `chained_capture_index` not reset after chain completion caused false terminals after every multi-jump; (2) `evaluate()` win/loss display was inverted because `step()` flipped before checking terminal, leaving `current_player` as the loser — training unaffected (bootstrap zeroed at terminal), only display wrong, fixed by flipping back before returning; (3) king encoding `(cell+1)/2` mapped opponent kings (-2) to -0.5, heavily suppressed by LeakyReLU's negative slope — fixed to `(cell+2)/4` which maps all values to [0,1]. King encoding fix produced the largest single improvement: +11.7pp at 100k (77.8% → 87.4%), diverging sharply from 1k episodes onward where kings become strategically relevant. n=5 used. |
| 8 | Backgammon | `DeepClanker` | ✅ Complete — 72.0% win rate vs random after 100k episodes (28.0% losses, 0% draws). Lower ceiling than deterministic games (87.4% checkers) is expected — dice randomness is irreducible, so even perfect play can't guarantee wins against lucky rolls. Sub-move design: each die use = one `step()` call; state includes remaining dice (4 values zero-padded); turn flips when dice exhausted or no more legal moves. State: 56 floats — two 24-value channels (my/opp piece counts, normalized /15), 4 bar/off counts (/15), 4 dice (/6). Action: `from_idx * 26 + to_idx` (650 possible). One critical crash bug fixed during training: `_advance_if_no_moves` with limit=8 failed to resolve closed-board positions (player on bar with all 6 entry points blocked), causing `step()` to return `done=False` with empty legal actions → crash. Fixed with limit=100 and a draw fallback in `step()` if still stuck after 100 attempts. Real backgammon has no draws — fallback is only reached in degenerate/impossible game states. n=5 used. |

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
| 6 | Othello | Not started — 67% vs random in Phase 1 confirms DQN is learning real strategy. Generational training targets determinism exploitation, the remaining ceiling. Expect meaningful improvement over the Phase 1 baseline. |
| 7 | Checkers | Not started — 77.8% vs random in Phase 1. Same plateau pattern as other games; generational training is the right next step. |
| 8 | Backgammon | Not started — 72.0% vs random in Phase 1. Stochasticity caps the ceiling vs random; generational training targets strategic depth and exploitation resistance rather than raw win rate improvement. |

---

## Phase 2 — AlphaZero (MCTS + Neural Network)

**Pipeline validation games** (Mancala, Connect Four): both are fully solved, DQN already near-optimal, and AlphaZero won't produce transformative results — but they're useful for proving the MCTS pipeline works before tackling harder games. All other games below benefit meaningfully from MCTS.

| # | Game | Agent | Notes |
|---|---|---|---|
| 8 | Mancala | AlphaZero-style | **Pipeline validation.** Familiar game, small state space, good for validating the MCTS + neural network pipeline before scaling up. Kalah is solved (P1 wins), so perfect play is verifiable. |
| 9 | Connect Four | AlphaZero-style | **Pipeline validation.** Solved (P1 wins). DQN at 97.1% — AlphaZero closes the remaining gap to provably perfect play and provides a direct before/after comparison. |
| 10 | Ultimate Tic Tac Toe | AlphaZero-style | Not fully solved. Significant strategic depth — DQN got 85.8% vs random but generational training has limits. MCTS expected to be genuinely stronger. |
| 11 | Gomoku (15×15) | AlphaZero-style | Not fully solved (P1 wins in theory but proof is deep). DQN hit 99.9% vs random but random is trivially easy on a 15×15 board — true strategic strength is unknown. MCTS with neural network guidance is the natural fit for the large branching factor. |
| 12 | Othello | AlphaZero-style | Not fully solved on 8×8. DQN failed — complex strategic depth rewards deep lookahead that MCTS handles naturally. MCTS is expected to be transformative here. |
| 13 | Checkers | AlphaZero-style | Solved by Chinook in 2007 — perfect play always draws. First major board game solved by computer; achieving near-perfect play is a meaningful milestone. DQN will fall well short of optimal. |
| 14 | Backgammon | AlphaZero-style | Stochastic — requires chance nodes in MCTS (one node per possible dice outcome, weighted by probability). TD-Gammon proved neural network value learning works here; AlphaZero-style should be stronger still. Revisit depends on Phase 1 DQN results. |
| 15 | Chess | AlphaZero-style | MCTS + neural network. Standard DQN cannot handle this state space. Long-term goal. |

AlphaZero approach: neural network guides Monte Carlo Tree Search (look-ahead), trained via self-play. Computationally expensive but necessary for Chess-level complexity. Mancala serves as the entry point to validate the pipeline before tackling larger state spaces.

---

## Phase 3 — Partial Observability (Card Games)

Partial observability means the agent cannot see the full game state (e.g. opponent's hand). Requires recurrent networks (LSTM/GRU) to maintain a belief state over time — the agent reasons about what it *doesn't* know. Some games in this phase (notably RPS) also require **stochastic policies** — the Nash equilibrium is a probability distribution over actions, not a single best action. DQN produces deterministic policies and cannot represent this; PPO (Proximal Policy Optimization) outputs action distributions and handles it naturally. Multi-player games (3+ players) also break the two-player self-play and negamax assumptions — requires multi-agent RL approaches (e.g. independent Q-learning or centralized training with decentralized execution).

| # | Game | Players | Notes |
|---|---|---|---|
| 10 | Blackjack | 2 (vs dealer) | Classic RL textbook example. Simple partial observability, well-understood optimal policy (good benchmark). Entry point to this phase — introduces partial observability with dense structural information (your cards, dealer up-card, deck probabilities). |
| 11 | Rock Paper Scissors (best-of-3, history-aware) | 2 | Isolates pure opponent modeling with minimal structural knowledge. State is the history of both players' throws; action space is {R, P, S}. Against random opponents there is no exploitable signal — this game is only interesting against patterned agents or humans. Requires PPO (stochastic policy) rather than DQN to represent the mixed strategy Nash equilibrium, and a recurrent architecture (LSTM/GRU) to model opponent tendencies across rounds. Counterintuitively harder than Blackjack despite simpler rules — observations are information-sparse and the entire learning problem reduces to inferring an opponent's throw distribution from a short sequence. Previously rejected as a Phase 1 game; fits here because Phase 3 tools (recurrent networks, policy gradients) address the core issues. **Introduces opponent profiling**: a good agent learns across multiple games against the same person — building a per-player profile of throw frequencies and conditional patterns on top of population-level priors (e.g. humans open with rock more than chance would predict). The opening throw is an information-gathering decision: against an unfamiliar opponent, the throw that maximizes *information about their distribution* may differ from the throw that maximizes immediate expected reward. See Opponent Profiling & Meta-Learning section. |
| 12 | Gin Rummy | 2 | Introduces hand management and partial observability in a familiar 2-player format before going multi-player. |
| 13 | Oh Hell (Wizard) | 3–6 | Trick-taking + exact bidding + multi-player. Most direct stepping stone to Skull King — same core mechanics, much simpler card hierarchy. Critical bridge game. |
| 14 | Poker (Texas Hold'em) | 2–9 | Adds bluffing, betting, and opponent modeling on top of partial observability. Active research area. |
| 15 | Skull King | 2–8 | Long-term goal. Family favorite. Combines trick-taking, exact bidding, multi-player, partial observability, and complex card hierarchy (Skull King > Pirates > Mermaids > Black suit > colored suits, with special capture interactions). Feasible after Oh Hell is working. |

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

### Multi-Agent Adversarial — Orbit Wars

Orbit Wars (Kaggle competition environment) is a Planet Wars-style RTS on a 100×100 continuous board. Players launch fleets from owned planets to capture neutral and enemy planets; the player with the most total ships after 500 turns wins. The environment is provided via `kaggle-environments` (`pip install "kaggle-environments>=1.28.0"`), so no simulation work is needed — focus is entirely on the agent.

This is the bridge between single-agent control and the Robot Arena goal: continuous state, semi-continuous actions (angle is a float, ship count is integer), adversarial multi-agent (1v1 and 4-player FFA), and complex physics (orbiting planets, sun collision, logarithmic fleet speed scaling).

**Why it matters:** breaks all Phase 1–3 assumptions simultaneously — not two-player, not turn-based, not discrete state, not negamax-compatible. Forces development of multi-agent PPO self-play and a proper continuous-space pipeline before tackling the Robot Arena.

**Key challenges:**
- **Orbit prediction:** inner planets rotate at a fixed angular velocity; fleets must be aimed at where the planet *will be*, not where it is. Use `initial_planets` + `angular_velocity` + elapsed turns to compute future positions.
- **Semi-continuous actions:** angle is a float (0–2π), ship count is a positive integer, source planet is a discrete choice. Action space is a mix — use a hybrid head or discretize angle into N bins.
- **Multi-agent self-play:** 4-player FFA breaks two-player negamax. Use independent PPO (each agent trains against the others) or centralized training with decentralized execution (CTDE). Start with 1v1.
- **Turn budget:** 1 second per turn. Inference must be fast — no MCTS at decision time unless heavily pruned.
- **Reward shaping:** terminal reward (ship count at step 500) is very sparse over 500 turns. Intermediate rewards needed: +ships_gained_from_capture, -ships_lost_in_combat, +production_rate_delta.

**Recommended build order:**
1. **Heuristic baseline** — greedy expansion (capture nearest low-garrison planet), threat response (reinforce under-attack planets), orbit-aware aiming. Establishes a non-trivial opponent to train against and a performance floor.
2. **PPO 1v1** — train a PPO agent against the heuristic, then via self-play. Validates the continuous-space pipeline.
3. **PPO 4-player FFA** — extend to multi-agent once 1v1 is stable.
4. **Kaggle competition submission** — the environment is a recurring Kaggle challenge; a mature agent can be submitted whenever the competition runs again.

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

## Othello — Post-mortem Notes

Early runs showed degradation (49.2% at 1k → 28.1% at 10k, then 57.1% at 1k → 42.0% at 10k with n=3) — both were corrupted by the `_flush_n_step` compound_sign bug. The bug wrongly applied a running perspective-flip sign to intermediate rewards, negating terminal rewards for ~1/3 of terminal transitions (those landing at odd buffer positions). The agent was literally learning "winning is bad" in those cases. Fixed run with n=3: 67.0% at 100k, confirming DQN does learn meaningful Othello strategy given correct signals.

- **Bug (fixed) — compound_sign applied to rewards:** `_flush_n_step` accumulated a `compound_sign` that correctly adjusts the bootstrap Q-value for perspective flips, but the same sign was also multiplied into intermediate rewards. Since all stored rewards are already from the agent's perspective, they should never be sign-adjusted. Fix: remove `compound_sign` from the reward accumulation; track the last entry's `negate` flag directly as the bootstrap sign. Affected all n_steps > 1 runs across all games.
- **Bug (fixed) — asymmetric retroactive reward:** `clanker.py` gave the agent `-1` when the opponent won their last move, but no `+1` when the opponent lost on their last move. Fixed to `_on_opponent_terminal` handling both cases symmetrically.
- **Remaining ceiling — determinism exploitation:** as epsilon decays, the agent locks into greedy flip-maximizing play — a well-known beginner trap that hands corners and edges to the opponent. Even random opponents punish this at high episode counts. Addressed in Phase 1.5 via generational training.
- **Phase 1.5 outlook:** generational training directly targets the determinism exploitation ceiling. Starting from the 67% Phase 1 checkpoint, expect meaningful improvement. Phase 2 (AlphaZero) remains the long-term target for strategic depth.

---

## Known Fragilities

- **`TableClanker.load`** parses state tuples from Python's default `str(tuple)` format (e.g. `"(0, 1, -1)"`). Relies on `', '` as separator and single-digit integers. Works for all planned games but would break if states ever contain multi-digit values or different formatting.

---

## Opponent Profiling & Meta-Learning

Applies across all phases. A general-purpose agent knows how to play the game well. A *profile-aware* agent also knows how to play well against *this specific opponent*.

### Two Layers

**Population-level priors** — general tendencies learned across many opponents. For humans: rock-heavy openers in RPS, common opening lines in Chess, predictable loss-aversion patterns, etc. This is the default policy against any new player the agent has never seen.

**Individual profiles** — per-player belief state that diverges from the population prior as games accumulate. After 50 games with your brother, the agent knows his specific tendencies in Chess. After 20 games of RPS with your dad, it knows he almost never throws scissors after a win.

### How Profiles Manifest Per Game Type

**RPS** — the entire game is opponent modeling. Profiles are explicit statistical records: throw frequencies, conditional patterns (what they play after a loss, after a win, after a tie), and sequential tendencies. Very little structural game knowledge exists, so the profile carries most of the weight.

**Structured games (Chess, Checkers, Othello, etc.)** — the opponent's tendencies manifest naturally as board positions. Playing the same person repeatedly populates experience with their specific patterns, providing some implicit adaptation for free through normal training. A more deliberate approach: a small per-opponent adaptation layer on top of the frozen base agent weights, fine-tuned on that person's games.

### The Fine-Tuning Trap

Naively fine-tuning the full agent on one person's games improves performance against them but degrades general strength — the model forgets everything else it learned. This is **catastrophic forgetting**, and it makes naive fine-tuning unsuitable for a multi-user system where the same agent plays many different people.

### Meta-Learning (MAML-Style) Solution

Train the global weights specifically to be *fast-adaptable* — rather than learning the single best policy, learn weights that can be quickly updated to exploit any specific opponent after only a few games. The global weights encode "what patterns to look for and how to exploit them" rather than a fixed strategy. A handful of games against a new opponent then produces meaningful adaptation without degrading base capability. This is the core idea behind MAML (Model-Agnostic Meta-Learning).

### Information-Directed Exploration

In information-sparse games (RPS especially), early moves against an unfamiliar opponent should balance expected immediate reward with **information value** — choosing moves whose responses reveal the most about the opponent's distribution. Against a familiar opponent with a rich profile, exploit what you already know. This tradeoff formalizes as: `utility = expected_reward + λ * information_gain`, where λ decays as the profile matures.

### Personalized Agent Feature (Opt-In)

Users can opt in to having a personal copy of each game's global agent trained specifically on their play. The design:

- **On opt-in:** the global agent's weights are copied to a per-user personal agent. This is the starting point — the personal agent plays identically to the global agent until enough games accumulate to diverge.
- **Always learning:** while opted in, the personal agent updates after every game the user plays, regardless of whether they chose to face the personal or global agent that session.
- **User choice:** at any time, the user can play against the global agent (stable, general-purpose) or their personal agent (adapted to them specifically).
- **Privacy:** the replay buffer stores floating point tensors `(state, action, reward, next_state, done)` — not human-readable game records. Raw move histories are not persisted to the database.

**Making adaptation feel faster** — the personal agent uses more aggressive hyperparameters than the global agent:
- Higher learning rate (e.g. α=0.001 vs global α=0.0001)
- Smaller replay buffer (e.g. 5–10k vs global 100k) so recent games dominate
- Multiple gradient update passes at game end, rather than waiting for the next random batch sample
- Optionally: amplified terminal rewards (e.g. r=2 or r=3) to increase the gradient signal per game. Keep bounded to avoid instability.

**The full trajectory requirement:** amplifying the terminal reward alone only directly updates the last move's Q-value. The rest of the game propagates backward through the buffer over subsequent games. For the adaptation to feel fast, the full in-game trajectory must be in the personal buffer — which it is naturally, since transitions are stored move-by-move during play.

**Catastrophic forgetting mitigation:** periodically mix a small batch from the global training set into the personal agent's updates, so it retains general strength while still specializing. Without this, after hundreds of games against one person the personal agent may over-fit and forget how to handle anything outside their patterns.

**Global agent rebase question (unresolved):** when the global agent is retrained and improved, should existing personal agents be rebased onto the new global weights? Rebasing gives everyone the improved base but partially undoes accumulated personalization. Keeping the old base preserves personalization but widens the gap between personal and global agents over time. Decide before building.

### Web & Infrastructure Implications

- **User accounts are required** — profiles are per-player and must persist across sessions
- **Database storage** — personal agent weights (`.pth` files) stored per user per game, plus opt-in preferences. A simple key-value store (e.g. SQLite for local, Postgres on Render) is sufficient initially.
- **Live read/write during play** — the personal agent is loaded at game start and its updated weights written back after each game. More demanding than a static model serve.
- **Cold start** — first game after opt-in, the personal agent is identical to the global agent. Divergence becomes noticeable after ~50–100 games depending on the game's complexity and how distinctive the user's play style is. Surface a game count in the UI so users know where they are.

---

## Web Frontend Plan

- **Framework:** Flask (full UI control). Streamlit is an alternative if minimal frontend work is preferred — has free hosting via Streamlit Cloud.
- **Hosting:** Render or Railway (free tier, easy Flask deploys).
- **Model upload feature:** accept JSON Q-tables (not pickle) to avoid arbitrary code execution vulnerability. `TableClanker` save/load already uses JSON for this reason.
- **User accounts required** — needed for the Friend/Family Challenge (agent submissions) and for opponent profiling (per-player profiles loaded during play). Accounts are not optional if profiling is in scope.
- **Database required for profiling** — opponent profiles (throw histories, adaptation weights, game records) must persist across sessions. A simple key-value store (e.g. SQLite for local, Postgres on Render) is sufficient initially.
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
5. ✅ Gomoku — complete
6. ✅ Othello — 67.0% vs random at 100k (fixed run), Phase 1.5 revisit planned
7. ✅ Checkers — complete (87.4% vs random at 100k)
8. ✅ Backgammon — complete (72.0% vs random at 100k)
9. Phase 1.5 — revisit Phase 1 games with generational training (excluding Othello)
9. Build web frontend after Phase 1.5 is proven on at least one game
10. Phase 2 (AlphaZero) after Phase 1.5
11. Phase 3 (card games) and Phase 4 (continuous control) are long-term goals
