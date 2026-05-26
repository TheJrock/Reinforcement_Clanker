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
| 1 | Tic Tac Toe | Skipped — TableClanker already converges to optimal play against random; no strategic depth to unlock via generational training. Next meaningful step is Phase 2 (AlphaZero pipeline validation). |
| 2 | Ultimate Tic Tac Toe | Not started — also benefits from state encoding fix (see post-mortem) |
| 3 | Mancala | ✅ Complete — 1 generation saved. Gen 1 beat the Phase 1 100k agent at 100% win rate after only 4000 training episodes, with draws dropping to 0% (gen 1 learned a decisive aggressive strategy). Gen 2 stalled across two attempts with different pool weights (50% previous / 40% pool, then 90% previous / 10% pool) — best win rate 51.9% in either run. Higher previous weight caused faster collapse (0% after 2000 episodes vs 4000), confirming the issue is not a weighting problem. Erratic oscillation between 0% and ~50% is a rock-paper-scissors dynamic: DQN finds narrow counter-strategies that are either completely dominant or completely dominated with no stable middle ground. Win detection verified correct via direct self-play eval test. Gen 1 is the DQN ceiling for Mancala under generational training. Round-robin (0.0, 0.1, gen 1): `a1_0.0` dominates at 100% overall, gen 1 at 50%, `a1_0.1` at 0% — a strict but counterintuitive hierarchy. The 10k random agent beat the generational agent completely, confirming that all three policies are brittle and narrow. Also confirms that 100k random training can produce a *worse* seed than 10k by over-specializing against random play. |
| 4 | Connect Four | ✅ Complete — 40 generations trained. **Recommended agent: `connect_four_a1_36` (81.7% overall win rate in 42-agent full round-robin).** Four round-robins run across training. Clear pattern: the overall win rate floor rises with each band of generations (gens 1–6 dominated by 7–12, dominated by 13–26, dominated by 27–40), but no stable strategy ever emerges — every peak agent has at least one hard counter from earlier gens (gen 36 loses to gen 8 at 100%, gen 37 loses to gen 7 at 100%). Gen 40 regressed to 14th overall (64.2%), confirming the DQN ceiling was hit around gen 35–36 and training beyond that oscillates without improvement. Three-tier structure in final RR: top tier (gens ~27–39, 60%+), middle tier (gens ~13–26, 40–60%), bottom tier (gens 0.0–12, below 40%). Intransitivity is extreme throughout — gen 35 beats gen 36 at 100% yet ranks below it overall, illustrating that no single metric captures the full picture. Notable finding from Mancala round-robin: `a1_0.0` (10k random) beat both `a1_0.1` (100k random) and `a1_1` (gen 1) at 100%, suggesting over-training on random opponents can produce a more rigid, exploitable policy. |
| 5 | Gomoku | In progress — 36 generations trained (gen 36 in progress). Four round-robins and extensive game analysis completed. **First RR (gens 0.1–15):** near-perfect staircase through gen 6, intransitivity emerging in gens 7–14, gen 15 dominant at 97%. **Second RR (gens 0.1–21):** gen 20 leads at 88%, full RPS structure among top gens, gen 10 anomaly persists (still beats gen 21 100%). **Third RR (gens 20–32):** gen 30 leads at 79.8% overall. Perfect three-way RPS cycle at the top: 30 beats 32 (100%), 32 beats 31 (100%), 31 beats 30 (100%). Gen 24 collapses to 17% (worst ever) — overfitted to a specific pool composition, useless against everything else. Non-monotonic ordering confirmed: training past the ceiling produces oscillation without improvement. **Strategic era shift at gen 33:** gen 33 developed a coherent col-row=4 diagonal attack; gen 34 inherited it and added multi-threat construction — building simultaneous threats in column 9 (vertical), row 4, row 5, and row 1 that force opponents to block in multiple places at once. Gen 34 beat all predecessors 100%. **Fourth RR (gens 31–35):** perfect linear dominance chain — 35 > 34 > 33 > 32 > 31, no cycles, each gen beats all predecessors 100%. Strategic inheritance confirmed: gen 35 started from gen 34's weights, struggled at 0% win rate for longer than any prior generation (4k+ episodes), then broke through with both the inherited diagonal attack AND genuine defensive capability. **Full RR (gens 20–35, 136 matchups × 50 games):** gen 35 sweeps 100% against all 15 opponents including gen 20. Gen 34 loses to gen 20 at 0% — gen 34's multi-threat attack is tuned against trained opponents and fails against simple positional play it can't block. Gen 35 beats gen 20 (100%), confirming it acquired defensive capability gen 34 lacked. Clear two-era structure: gens 20–32 form a noisy RPS plateau; gens 33–35 form a clean linear dominance era that supersedes all prior cycles. **Gen 36 regression:** started at 52.5% win rate (inherited gen 35's weights), dropped to 0% after 4k training episodes. Root cause: pool dilution — with 36+ agents in the pool sampled uniformly, the vast majority of training games are against easy old opponents, dominating the replay buffer with low-quality gradient updates. Games also nearly doubled in length (1.5s → 2.9s per game), indicating aimless wandering rather than purposeful play. **Training fix implemented:** pool sampling split into four buckets — previous=40%, recent (last 5 gens)=35%, old=22%, random=3%. Sampling table rebuilt at startup and on each generation save only (not per episode). Will restart training from gen 36 checkpoint once it saves, applying new weights for gen 37 onward. **Tools created:** `watch_game.py` (interactive step-through game viewer), `probe_blocking.py` (isolated tactical scenario tester — confirmed gen 32 plays (6,8) on all threat scenarios regardless of board state). **Assessment:** DQN ceiling approaching but not yet fully hit — gen 35's genuine defensive capability is a meaningful leap. Pool dilution is now the primary training bottleneck; the sampling fix is the most important architectural change made to the training loop. |
| 6 | Othello | Not started — 67% vs random in Phase 1 confirms DQN is learning real strategy. Generational training targets determinism exploitation, the remaining ceiling. Expect meaningful improvement over the Phase 1 baseline. |
| 7 | Checkers | Not started — 77.8% vs random in Phase 1. Same plateau pattern as other games; generational training is the right next step. |
| 8 | Backgammon | Not started — 72.0% vs random in Phase 1. Stochasticity caps the ceiling vs random; generational training targets strategic depth and exploitation resistance rather than raw win rate improvement. |

---

## Phase 2 — MCTS + Neural Network (AlphaZero-style)

**The general algorithm is MCTS (Monte Carlo Tree Search)**, developed around 2006 and used in game-playing programs well before deep learning. AlphaZero is a specific instantiation: MCTS guided by a neural network that outputs both a policy (which moves look promising) and a value (who is winning from this position). Classic MCTS uses random rollouts to estimate value; AlphaZero replaces those rollouts with the neural network's value estimate, which is far more efficient. The selection formula (PUCT) also uses the network's policy as a prior to guide exploration. **Expert Iteration (ExIt)** is a parallel framing of the same idea: use MCTS as an "expert" to generate better moves than the current policy, train the policy to imitate the expert, repeat. Both names describe the same core loop.

**AlphaZero's neural network is an actor-critic architecture** — a policy head and a value head on a shared backbone. This is the same structural foundation as PPO. Understanding PPO first makes AlphaZero's training loop significantly more legible; however, MCTS is proceeding before PPO in this project, and PPO concepts will be revisited when Phase 3 (card games) and Phase 4 (continuous control) require them.

**Note on rules being given:** AlphaZero still requires the environment to supply legal moves and detect game-over conditions — `get_legal_actions()` and the terminal signal in `step()`. This is a meaningful form of rule knowledge being handed to the agent. The agent learns *which legal move is best*, not *what the moves mean* or *what is legal*. MuZero (Phase 5) removes this requirement entirely. See Phase 5 for discussion of the trajectory toward fully learned environments.

**Pipeline validation games** (Mancala, Connect Four): both are fully solved, DQN already near-optimal, and AlphaZero won't produce transformative results — but they're useful for proving the MCTS pipeline works before tackling harder games. All other games below benefit meaningfully from MCTS.

| # | Game | Agent | Notes |
|---|---|---|---|
| 8 | Mancala | AlphaZero-style | **Pipeline validation.** Familiar game, small state space, good for validating the MCTS + neural network pipeline before scaling up. Kalah is solved (P1 wins), so perfect play is verifiable. |
| 9 | Connect Four | AlphaZero-style | **Pipeline validation.** Solved (P1 wins). DQN at 97.1% — AlphaZero closes the remaining gap to provably perfect play and provides a direct before/after comparison. |
| 10 | Ultimate Tic Tac Toe | AlphaZero-style | Not fully solved. Significant strategic depth — DQN got 85.8% vs random but generational training has limits. MCTS expected to be genuinely stronger. |
| 11 | Gomoku (15×15) | AlphaZero-style | Not fully solved (P1 wins in theory but proof is deep). DQN self-play demonstrably failed to learn blocking — 32 generations produced agents that collapse to a fixed position preference (always row 6, col 8) regardless of board state, confirmed by direct Q-value probing. MCTS handles tactics through explicit lookahead rather than requiring the value function to encode every threat pattern. |
| 12 | Othello | AlphaZero-style | Not fully solved on 8×8. DQN learned basic strategy but hit a ceiling — complex positional play rewards deep lookahead that MCTS handles naturally. MCTS is expected to be transformative here. |
| 13 | Checkers | AlphaZero-style | Solved by Chinook in 2007 — perfect play always draws. First major board game solved by computer; achieving near-perfect play is a meaningful milestone. DQN will fall well short of optimal. |
| 14 | Backgammon | AlphaZero-style | Stochastic — requires chance nodes in MCTS (one node per possible dice outcome, weighted by probability). TD-Gammon proved neural network value learning works here; AlphaZero-style should be stronger still. |
| 15 | Chess | AlphaZero-style | MCTS + neural network. Standard DQN cannot handle this state space. Long-term goal. Use `python-chess` as the environment backend — do not build a Chess environment from scratch. Wrap it in a Gymnasium-compatible interface. |

MCTS + neural network approach: the neural network guides tree search (look-ahead), trained via self-play. MCTS handles tactics explicitly through search; the neural network handles strategy through learned value and policy estimates. Mancala serves as the entry point to validate the pipeline before tackling larger state spaces.

### Environment Strategy (Phase 2 onward)

For Phase 1 and 1.5, building custom environments was the right investment — full control over state encoding, reward design, and action space, with clear educational value. Going forward, the investment belongs on the algorithm side, not the environment side.

**Adopt the Gymnasium interface for all new environments.** Gymnasium (`gymnasium`) is the universal standard — implementing `reset()`, `step()`, `observation_space`, and `action_space` makes any environment compatible with the broader RL ecosystem (Stable Baselines3, CleanRL, RLlib). This enables benchmarking custom agents against reference implementations and makes skills transferable to real RL work.

**For Chess:** use `python-chess` as the backend. It handles all rules, move generation, and edge cases (en passant, castling, 50-move rule, repetition) correctly. Wrap it in a thin Gymnasium interface. Building from scratch would cost weeks and produce an inferior result.

**For multi-agent games:** PettingZoo is the Gymnasium equivalent for multi-agent environments. It already includes Backgammon, Chess, Connect Four, Go, and others — worth knowing about before deciding to build custom multi-agent environments.

**Existing custom environments:** no need to rewrite them, but adding a thin Gymnasium-compatible wrapper would allow benchmarking DQN/AlphaZero agents against SB3 baselines on the same environments — a useful one-time investment when Phase 2 begins.

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

## Phase 5 — Model-Based RL (World Models)

Phases 1–4 are all **model-free**: the agent learns purely from environment interactions, with no internal representation of how the environment works. Model-based RL adds a **learned dynamics model** — the agent builds an internal simulator and can plan by imagining rollouts without querying the real environment. This is the next major conceptual jump after PPO.

### Why it matters

Model-free agents are sample-inefficient: they need millions of real environment interactions to learn. A world model amortizes this cost — once the dynamics are learned, the agent can generate unlimited synthetic experience internally. For expensive environments (real robotics, slow simulations, the Robot Arena), this matters enormously.

### MuZero — MCTS with a Fully Learned Environment

AlphaZero still requires the environment to provide legal moves and terminal conditions — `get_legal_actions()` and the done signal are implicit rule knowledge being handed to the agent. **MuZero** (DeepMind, 2020) removes this entirely: it learns three functions jointly from raw observations:
- **Representation:** encode the real observation into a latent state
- **Dynamics:** predict the next latent state and immediate reward given a latent state + action
- **Prediction:** output policy and value from a latent state (same as AlphaZero's network)

MCTS then runs entirely inside the learned latent space — no real environment needed at decision time, and no rules provided. The agent discovers what is legal, what moves do, and when the game ends purely from experience. MuZero matched AlphaZero on Chess, Go, and Shogi while also mastering Atari without being given the rules.

**This is the long-term goal of this project.** The motivation is not efficiency — it is building agents that require zero domain knowledge, which is the only viable approach for problems where we don't already know good solutions. Heuristic opponents, reward shaping, and rule injection all cap the agent at the level of the human knowledge baked in. MuZero-style learning removes that ceiling. The trajectory of this project is deliberately moving in this direction: Phase 1 (DQN, rules given, rewards given) → Phase 2 (MCTS, rules given, rewards given) → Phase 5 (MCTS, rules learned, rewards learned).

**Entry point:** apply MuZero to Connect Four or Mancala first (rules already known, AlphaZero baseline exists for comparison). Then scale to games where the rules are more expensive to query (Backgammon, eventually the Robot Arena).

### Dreamer — World Models for Continuous Control

For continuous control (Phase 4 territory), **Dreamer** (Google, 2020–2023) is the model-based analogue of PPO. It learns a recurrent world model in a compact latent space, then trains an actor-critic entirely inside the model's imagination. Dreamer v3 achieves strong results across diverse domains — Atari, continuous control, and 3D environments — with a single set of hyperparameters.

**Entry point:** replace the PPO agent in BipedalWalker or the Robot Arena with a Dreamer-style agent. Particularly compelling for the Robot Arena where real simulation is expensive.

### Offline RL / Imitation Learning (Related)

Not model-based per se, but adjacent: **offline RL** learns from a fixed dataset of experience (human gameplay, recorded rollouts) without interacting with the environment at all. **Imitation learning** (behavioral cloning, IRL) bootstraps from expert demonstrations. Both are relevant if you ever want to seed agents from recorded human games rather than pure self-play — e.g. training a Chess agent from a database of grandmaster games before self-play begins.

### Transformers in RL

**Decision Transformer** reframes RL as sequence modeling: given a history of (state, action, reward) triples and a target return, predict the next action. No Bellman updates, no replay buffer — just a causal transformer trained with supervised learning on trajectories. **Gato** (DeepMind, 2022) extended this to a single transformer trained across 600+ tasks simultaneously. These haven't displaced PPO for most practical work, but represent a fundamentally different paradigm worth understanding.

### Suggested Phase 5 Build Order

1. **MuZero on Mancala/Connect Four** — validate the learned dynamics model against the known AlphaZero baseline
2. **MuZero on Gomoku or Chess** — first application where the model-free approaches fall short
3. **Dreamer on Robot Arena** — replace PPO with model-based planning for the most sample-hungry environment in the project
4. **Decision Transformer on a Phase 3 card game** — explore the sequence-modeling paradigm on a partial-observability task

---

## DQN Self-Play — Fundamental Limitations

DQN was designed for single-agent Atari: dense rewards, no opponent, discrete actions. Two-player self-play breaks its core assumptions in ways that are not fixable by tuning hyperparameters.

**Non-stationarity:** Bellman updates bootstrap from Q-values that are themselves changing because the opponent is also training. The target network stabilizes this partially but cannot resolve it — both the environment and the opponent are moving targets simultaneously.

**Positional collapse:** In practice, DQN self-play on complex board games collapses Q-values to smooth position preferences rather than state-dependent tactics. The network learns "this cell is generally good" because that is far easier to extract from a noisy, non-stationary signal than "this cell is correct given this specific board configuration." Confirmed empirically in Gomoku: after 32 generations of generational training, all agents play row 6 col 8 on every board state regardless of threats, verified by direct Q-value probing with isolated threat positions.

**Why blocking never emerges:** The `_on_opponent_terminal` mechanism does propagate the "didn't block, then lost" signal back to the previous move — the credit assignment mechanism exists. The failure is distributional: because all pool agents share the same positional collapse, neither player consistently builds credible 4-in-a-row threats, so the signal almost never appears in the replay buffer. Increasing n_steps or starting a new architecture line from a different seed does not fix this — it is a property of the training distribution, not the credit assignment distance.

**Pool dilution:** generational training with a flat uniform pool compounds the distributional problem. Once the pool grows to 30+ agents, the vast majority of training games are against old, easily beaten opponents. The replay buffer fills with low-quality gradient updates that shift the policy toward exploiting weak agents, actively degrading capability against the current target. Empirically confirmed: gen 36 inherited gen 35's weights (starting at 52.5% win rate), then regressed to 0% after 4k training episodes against the full uniform pool. Fix: weighted sampling that concentrates training on recent and the immediate predecessor — previous=40%, recent N gens=35%, old=22%, random=3%.

**The right fix is a different algorithm.** Policy gradient methods (PPO) are more stable under self-play non-stationarity because they learn a probability distribution over actions rather than a deterministic argmax. MCTS handles tactics explicitly through lookahead — the network only needs to estimate who is winning, not encode every tactical pattern. AlphaZero combines both. This is the motivation for Phase 2.

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

## Two-Channel State Encoding (Future Experiment)

All current environments normalize board values to `[0, 1]` using a shift formula (e.g. `(v + 1) / 2`), so no negative inputs reach the network. This is sufficient for Phase 1 and 1.5.

A cleaner alternative is the **two-channel approach** used by AlphaZero and already implemented in Backgammon: rather than a single value per cell encoding both ownership and piece type, use separate non-negative channels — one for the current player's pieces, one for the opponent's. Each channel carries unambiguous, independent gradients.

**Potential benefits:** cleaner gradient signal per feature; network can independently learn "I have a piece here" vs "opponent has a piece here" without sharing a single neuron; more natural for games with multiple piece types (checkers kings, etc.).

**When to try it:** after Phase 1.5 is complete, as a controlled experiment. Train identical agents on the two-channel versions and run direct head-to-head evaluation against the Phase 1.5 agents to isolate the encoding impact.

**Implementation requirement:** to run a fair comparison, the old agents must stay intact. Options:
- Create new `*_env_v2.py` variants with two-channel encoding alongside the originals
- Or build a thin adapter that converts old state format to new, allowing old agents to play against new-encoding agents in a shared evaluation harness

The comparison is only meaningful if the two sets of agents were trained for equivalent episodes against equivalent opponents — so plan the two-channel retrains to mirror whatever Phase 1.5 produced.

---

## Model Naming Convention

Saved models follow the format `{game}_a{N}_{gen}.pth`, stored in each game's `stashed_models/` directory.

- `{game}` — lowercase game name (e.g. `checkers`, `backgammon`)
- `a{N}` — architecture version, starting at `a1`, incrementing whenever a change requires a full retrain from scratch
- `{gen}` — generation within that architecture: `0.0` = random-trained 10k, `0.1` = random-trained 100k, `1`/`2`/`3`... = Phase 1.5 generational checkpoints

Example: `checkers_a1_0.0.pth`, `checkers_a1_0.1.pth`, `checkers_a1_1.pth`, `checkers_a2_0.1.pth`

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
9. 🔄 Phase 1.5 Gomoku — 36 generations trained, gen 36 in progress. Wait for gen 36 to save, then restart with new weighted pool sampling (previous=40%, recent=35%, old=22%, random=3%). Run full RR (gens 20–37+) after several more generations to assess whether the sampling fix sustains the gen 33–35 linear improvement trend.
10. Phase 1.5 remaining games (Othello, Checkers, Backgammon, UTTT) — after Gomoku reaches its DQN ceiling
11. Phase 2 (MCTS/AlphaZero) — start with Mancala as pipeline validation, then Gomoku where DQN demonstrably failed at tactics
12. Phase 3 (card games) and Phase 4 (continuous control) are long-term goals
