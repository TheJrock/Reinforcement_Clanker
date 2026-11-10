# Reinforcement Clanker

AI agents ("clankers") that teach themselves to play board games from scratch using deep reinforcement learning. They are never shown a strategy, an expert game, or a hand-written evaluation function. They learn only by playing, mostly against earlier versions of themselves.

Games: **Tic-Tac-Toe, Connect Four, Mancala, Othello, Gomoku, Checkers, Backgammon, and Ultimate Tic-Tac-Toe.**

## How it works

**The learner.** Each agent is a Deep Q-Network (DQN) written in PyTorch (`clanker.py`). A small neural network looks at the board and estimates how good each possible move is. The agent plays a move, sees how the game turns out, and nudges its estimates toward what actually happened. Experience replay, a target network, and n-step returns keep this stable. Tic-Tac-Toe is small enough to use a plain lookup-table Q-learner instead.

**Generational training.** A single agent playing itself tends to chase its own tail, so training is split into generations (`generational_training.py`):

1. The current agent trains against a mix of opponents: mostly its previous generation and other recent generations, sometimes much older ones, and occasionally a random player.
2. Every 2,000 games it plays 500 evaluation games against its predecessor.
3. It **graduates** if it wins at least 55% of those games and holds at least 45% against the last few generations. That second check stops it from becoming a narrow specialist that only beats one opponent.
4. The graduate is saved, joins the opponent pool, and becomes the bar the next generation has to clear.

**Ranking.** `round_robin.py` plays every saved agent against every other one and ranks them by overall win rate.

Every graduated agent is saved in `generational_agents/`, named `<game>_a<run>_<generation>.pth`. For example, `othello_a2_24.pth` is generation 24 of the second Othello training run. Generations numbered `0.x` are the seed agents a run started from.

## What I learned

- **Ultimate Tic-Tac-Toe** showed the clearest progress. Over 43 generations the agents formed clear tiers, and the top generations won about two-thirds of their round-robin games. It plateaued around generation 40.
- **Gomoku** exposed the limits of the network design. A flat fully-connected network can't recognize that a threat in one corner is the same pattern as a threat in another, so it plateaued. Spotting the same pattern anywhere on the board is what convolutional networks are built for.
- **Othello** showed extreme non-transitivity. A beats B and B beats C, but C beats A. Some of the strongest agents came from early generations, and later ones sometimes got worse overall.
- **Mancala, Checkers, and Backgammon** peaked almost immediately, with their best agents in generation 1. Free extra turns (Mancala), drawn-out stalemates (Checkers), and dice luck (Backgammon) each work against plain DQN.

The planned next step was AlphaZero-style training (a neural network guided by Monte Carlo Tree Search), which is better suited to these games.

## Setup

Requires Python 3.13+. With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/TheJrock/Reinforcement_Clanker.git
cd Reinforcement_Clanker
uv sync
```

Or with pip: `pip install numpy torch tqdm`. If you use pip, run the commands below with `python` instead of `uv run python`.

Run every command from the repository root. Apple Silicon or NVIDIA GPUs are used automatically when available, but a CPU is plenty for playing and watching.

## Play against an agent

```bash
uv run python -m ConnectFour.connect_four_repl
```

Choose `2` to play against the AI, or `1` for two humans sharing the keyboard. Who goes first is random.

| Game | Command | How to enter a move |
|---|---|---|
| Tic-Tac-Toe | `uv run python -m TicTacToe.tic_tac_repl` | Square `0`–`8`. The agent trains for a few seconds first. |
| Connect Four | `uv run python -m ConnectFour.connect_four_repl` | Column `0`–`6` |
| Mancala | `uv run python -m Mancala.mancala_repl` | Pit `0`–`5` |
| Othello | `uv run python -m Othello.othello_repl` | Row, then column |
| Gomoku | `uv run python -m Gomoku.gomoku_repl` | Row, then column (15×15 board) |
| Ultimate Tic-Tac-Toe | `uv run python -m UltimateTicTacToe.ultimate_tic_tac_repl` | Board `0`–`8` (when free to choose), then square `0`–`8` |
| Checkers | `uv run python -m Checkers.checkers_repl` | Pick from a numbered list of legal moves |
| Backgammon | `uv run python -m Backgammon.backgammon_repl` | Pick from a numbered list of legal moves |

## Watch two agents play each other

```bash
uv run python watch_game.py
```

Pick a game, then pick any two saved agents from the numbered list. Press **Enter** to advance one move at a time, or type **q** to quit.

To see how much training mattered, pit an early generation against a late one, for example `ultimate_tictactoe_a1_1` vs. `ultimate_tictactoe_a1_40`, or `connect_four_a1_1` vs. `connect_four_a1_40`.

## Train or rank agents yourself

```bash
uv run python generational_training.py   # continue training from a game's newest agent
uv run python round_robin.py --episodes 200   # rank every agent of a game
```

Training is slow. Each generation can take tens of thousands of games.

## Project layout

```
clanker.py                 # Q-learning agents: tabular and Deep Q-Network
general_environment.py     # shared game-environment interface
generational_training.py   # generational self-play training loop
round_robin.py             # all-vs-all tournament ranking
watch_game.py              # watch two agents play
generational_agents/       # every graduated agent (.pth weights)
<Game>/                    # each game: *_env.py (rules), *_repl.py (play), stashed_models/
web/                       # unfinished browser UI
```

## Status

This project is no longer actively developed. The browser UI in `web/` was started but never finished, and the AlphaZero phase was never begun.
