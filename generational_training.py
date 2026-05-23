import os
import re
import random
from tqdm import tqdm

from clanker import DeepClanker
from Backgammon.backgammon_env import BackgammonEnvironment
from Checkers.checkers_env import CheckersEnvironment
from ConnectFour.connect_four_env import ConnectFourEnvironment
from Gomoku.gomoku_env import GomokuEnvironment
from Mancala.mancala_env import MancalaEnvironment
from Othello.othello_env import OthelloEnvironment
from UltimateTicTacToe.ultimate_tic_tac_toe_env import UltimateTicTacToeEnvironment

WIN_RATE_THRESHOLD = 0.55
EVAL_EPISODES = 1000
EVAL_FREQUENCY = 2000
RANDOM_AGENT_PROBABILITY = 0.1
PREVIOUS_AGENT_PROBABILITY = 0.5
OTHER_AGENT_PROBABILITY = 0.4
EPISODE_BUDGET = 100000

GENERATIONAL_AGENTS_DIR = "generational_agents"
GAMES = {
    "backgammon": {"env": BackgammonEnvironment, "sizes": (56, 650)},
    "checkers": {"env": CheckersEnvironment, "sizes": (33, 1024)},
    "connect_four": {"env": ConnectFourEnvironment, "sizes": (42, 7)},
    "gomoku": {"env": GomokuEnvironment, "sizes": (225, 225)},
    "mancala": {"env": MancalaEnvironment, "sizes": (14, 6)},
    "othello": {"env": OthelloEnvironment, "sizes": (64, 64)},
    "ultimate_tic_tac_toe": {"env": UltimateTicTacToeEnvironment, "sizes": (91, 81)},
}

def train_agent(environment, env_sizes, ClankerClass, initial_agent_path, save_path, agent_path_pool=None, eval_freq=EVAL_FREQUENCY):
    current_agent = ClankerClass(env_sizes[0], env_sizes[1])
    current_agent.load(initial_agent_path)
    current_agent.epsilon = 0.1
    current_agent.epsilon_decay = (0.01 / current_agent.epsilon) ** (1 / (0.8 * EPISODE_BUDGET))

    prev_agent = ClankerClass(env_sizes[0], env_sizes[1])
    prev_agent.load(initial_agent_path)
    prev_agent.epsilon = 0.0
    prev_agent_path = initial_agent_path

    agent_path_pool = agent_path_pool or []
    agent_pool = []
    for agent_path in agent_path_pool:
        agent = ClankerClass(env_sizes[0], env_sizes[1])
        agent.load(agent_path)
        agent.epsilon = 0.0
        agent_pool.append(agent)

    episode_count = 0
    total_episodes = 0
    best_win_rate = 0.0
    generation = 1
    while episode_count < EPISODE_BUDGET:
        for _ in tqdm(range(min(eval_freq, EPISODE_BUDGET - episode_count)), desc=f"Gen {generation} | Episode {episode_count}/{EPISODE_BUDGET}"):
            opponent_choice = random.choices(
                population=['random', 'previous', 'pool'],
                weights=[RANDOM_AGENT_PROBABILITY, PREVIOUS_AGENT_PROBABILITY, OTHER_AGENT_PROBABILITY],
                k=1
            )[0]

            if opponent_choice == 'random':
                opponent = None
            elif opponent_choice == 'previous':
                opponent = prev_agent
            else:
                opponent = random.choice(agent_pool) if agent_pool else None

            current_agent.train(environment, episodes=1, opponent=opponent, load_bar=False)
            episode_count += 1

        counts = current_agent.evaluate(environment, opponent=prev_agent, episodes=EVAL_EPISODES, verbose=False)
        win_rate = counts['wins'] / EVAL_EPISODES
        best_win_rate = max(best_win_rate, win_rate)
        print(f"Win rate: {win_rate:.2%}, Loss rate: {counts['losses'] / EVAL_EPISODES:.2%}, Draw rate: {counts['draws'] / EVAL_EPISODES:.2%}")

        if win_rate >= WIN_RATE_THRESHOLD:
            print(f"New agent saved with win rate {win_rate:.2%}")
            current_agent.save(f"{save_path}_{generation}.pth")
            current_agent.epsilon = 0.1

            pool_agent = ClankerClass(env_sizes[0], env_sizes[1])
            pool_agent.load(prev_agent_path)
            pool_agent.epsilon = 0.0
            agent_pool.append(pool_agent)

            prev_agent_path = f"{save_path}_{generation}.pth"
            prev_agent.load(prev_agent_path)
            generation += 1
            total_episodes += episode_count
            episode_count = 0
            best_win_rate = 0.0

    total_episodes += episode_count
    if generation == 1:
        print(f"Stalled — no checkpoint saved after {total_episodes} episodes. Best win rate vs initial agent: {best_win_rate:.2%}.")
    else:
        print(f"Stalled on generation {generation} after {total_episodes} total episodes. {generation - 1} checkpoint(s) saved. Best win rate this generation: {best_win_rate:.2%}.")

def find_agents(game):
    agents = []
    for fname in os.listdir(GENERATIONAL_AGENTS_DIR):
        match = re.match(rf"^{game}_a(\d+)_([\d.]+)\.pth$", fname)
        if match:
            arch = int(match.group(1))
            gen = float(match.group(2))
            path = os.path.join(GENERATIONAL_AGENTS_DIR, fname)
            agents.append((arch, gen, path))
    return sorted(agents, key=lambda x: (x[0], x[1]))

def get_latest_and_pool(game):
    agents = find_agents(game)
    if not agents:
        return None, None, []
    max_arch = max(a[0] for a in agents)
    arch_agents = [a for a in agents if a[0] == max_arch]
    initial_path = arch_agents[-1][2]
    pool_paths = [a[2] for a in arch_agents[:-1]]
    save_path = os.path.join(GENERATIONAL_AGENTS_DIR, f"{game}_a{max_arch}")
    return initial_path, save_path, pool_paths

def main():
    print("Available games:")
    for name in GAMES:
        print(f"  {name}")
    game = None
    while game not in GAMES:
        game = input("Select game: ").strip().lower()
        if game not in GAMES:
            print(f"Invalid selection. Choose from: {', '.join(GAMES.keys())}")

    initial_path, save_path, pool_paths = get_latest_and_pool(game)
    if initial_path is None:
        print(f"No agents found for '{game}' in {GENERATIONAL_AGENTS_DIR}/. Add at least one agent before training.")
        return

    print(f"\nGame:          {game}")
    print(f"Initial agent: {initial_path}")
    print(f"Pool:          {pool_paths if pool_paths else 'empty (no prior checkpoints)'}")
    print(f"Save path:     {save_path}_{{generation}}.pth\n")

    config = GAMES[game]
    environment = config["env"]()
    train_agent(environment, config["sizes"], DeepClanker, initial_path, save_path, pool_paths)

if __name__ == "__main__":
    main()