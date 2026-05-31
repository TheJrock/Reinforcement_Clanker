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
EVAL_EPISODES = 500
EVAL_FREQUENCY = 2000
RANDOM_AGENT_PROBABILITY = 0.03
PREVIOUS_AGENT_PROBABILITY = 0.40
RECENT_AGENT_PROBABILITY = 0.35
OLD_AGENT_PROBABILITY = 0.22
POOL_WINDOW = 5
EPISODE_BUDGET = 100000

STALL_WINDOW = 5
STALL_WIN_THRESHOLD = 0.25
STALL_EPSILON_THRESHOLD = 0.03
STALL_RESET_EPSILON = 0.6

SECONDARY_EVAL_EPISODES = 100
SECONDARY_WIN_THRESHOLD = 0.45

GENERATIONAL_AGENTS_DIR = "generational_agents"
GAMES = {
    "backgammon":        {"env": BackgammonEnvironment,       "sizes": (56, 650),  "n_steps": 5},
    "checkers":          {"env": CheckersEnvironment,         "sizes": (33, 1024), "n_steps": 5},
    "connect_four":      {"env": ConnectFourEnvironment,      "sizes": (42, 7),    "n_steps": 1},
    "gomoku":            {"env": GomokuEnvironment,           "sizes": (225, 225), "n_steps": 1},
    "mancala":           {"env": MancalaEnvironment,          "sizes": (14, 6),    "n_steps": 1},
    "othello":           {"env": OthelloEnvironment,          "sizes": (64, 64),   "n_steps": 3},
    "ultimate_tictactoe": {"env": UltimateTicTacToeEnvironment, "sizes": (91, 81), "n_steps": 5, "buffer_size": 500000},
}

def train_agent(environment, env_sizes, ClankerClass, initial_agent_path, save_path, agent_path_pool=None, eval_freq=EVAL_FREQUENCY, n_steps=1, buffer_size=100000, starting_generation=1, random_weight=RANDOM_AGENT_PROBABILITY, previous_weight=PREVIOUS_AGENT_PROBABILITY, recent_weight=RECENT_AGENT_PROBABILITY, old_weight=OLD_AGENT_PROBABILITY, pool_window=POOL_WINDOW):
    def make_agent():
        return ClankerClass(env_sizes[0], env_sizes[1], n_steps=n_steps, buffer_size=buffer_size)

    current_agent = make_agent()
    current_agent.load(initial_agent_path)
    current_agent.epsilon = 0.1
    current_agent.epsilon_decay = (0.01 / current_agent.epsilon) ** (1 / (0.8 * EPISODE_BUDGET))

    prev_agent = make_agent()
    prev_agent.load(initial_agent_path)
    prev_agent.epsilon = 0.0
    prev_agent_path = initial_agent_path

    agent_path_pool = agent_path_pool or []
    agent_pool = []
    for agent_path in agent_path_pool:
        agent = make_agent()
        agent.load(agent_path)
        agent.epsilon = 0.0
        agent_pool.append(agent)

    def build_sampling(pool):
        recent = pool[-pool_window:] if pool else []
        old = pool[:-pool_window] if len(pool) > pool_window else []
        population = ['random', 'previous']
        weights = [random_weight, previous_weight]
        if recent:
            population.append('recent')
            weights.append(recent_weight)
        if old:
            population.append('old')
            weights.append(old_weight)
        return population, weights, recent, old

    population, weights, recent_pool, old_pool = build_sampling(agent_pool)

    episode_count = 0
    total_episodes = 0
    best_win_rate = 0.0
    generation = starting_generation
    recent_win_rates = []
    while episode_count < EPISODE_BUDGET:
        checkpoint_path = f"{save_path}_{generation}.pth"
        if episode_count == 0:
            current_agent.save(checkpoint_path)

        for _ in tqdm(range(min(eval_freq, EPISODE_BUDGET - episode_count)), desc=f"Gen {generation} | Episode {episode_count}/{EPISODE_BUDGET}"):
            opponent_choice = random.choices(population=population, weights=weights, k=1)[0]

            if opponent_choice == 'random':
                opponent = None
            elif opponent_choice == 'previous':
                opponent = prev_agent
            elif opponent_choice == 'recent':
                opponent = random.choice(recent_pool)
            else:
                opponent = random.choice(old_pool)

            current_agent.train(environment, episodes=1, opponent=opponent, load_bar=False)
            episode_count += 1

        counts = current_agent.evaluate(environment, opponent=prev_agent, episodes=EVAL_EPISODES, verbose=False)
        win_rate = counts['wins'] / EVAL_EPISODES
        best_win_rate = max(best_win_rate, win_rate)
        current_agent.save(checkpoint_path)
        print(f"Win rate: {win_rate:.2%}, Loss rate: {counts['losses'] / EVAL_EPISODES:.2%}, Draw rate: {counts['draws'] / EVAL_EPISODES:.2%}")

        recent_win_rates.append(win_rate)
        if len(recent_win_rates) > STALL_WINDOW:
            recent_win_rates.pop(0)
        avg_win_rate = sum(recent_win_rates) / len(recent_win_rates)
        if len(recent_win_rates) >= STALL_WINDOW and avg_win_rate < STALL_WIN_THRESHOLD and current_agent.epsilon < STALL_EPSILON_THRESHOLD:
            remaining = EPISODE_BUDGET - episode_count
            current_agent.epsilon = STALL_RESET_EPSILON
            current_agent.epsilon_decay = (current_agent.epsilon_min / STALL_RESET_EPSILON) ** (1 / max(1, int(0.8 * remaining)))
            recent_win_rates.clear()
            print(f"Stall detected (avg win rate {avg_win_rate:.2%} over last {STALL_WINDOW} evals) — epsilon reset to {STALL_RESET_EPSILON:.1f}, decaying over {int(0.8 * remaining)} episodes.")

        if win_rate >= WIN_RATE_THRESHOLD:
            graduate = True
            if agent_pool:
                secondary_sample = agent_pool[-pool_window:]
                secondary_wins = sum(
                    current_agent.evaluate(environment, opponent=a, episodes=SECONDARY_EVAL_EPISODES, verbose=False)['wins']
                    for a in secondary_sample
                )
                secondary_win_rate = secondary_wins / (SECONDARY_EVAL_EPISODES * len(secondary_sample))
                print(f"Secondary eval vs last {len(secondary_sample)} pool agent(s): {secondary_win_rate:.2%}")
                if secondary_win_rate < SECONDARY_WIN_THRESHOLD:
                    print(f"Graduation blocked — secondary win rate {secondary_win_rate:.2%} < {SECONDARY_WIN_THRESHOLD:.0%}.")
                    graduate = False

            if graduate:
                print(f"New agent graduated with win rate {win_rate:.2%}")
                current_agent.epsilon = 0.1
                current_agent.epsilon_decay = (current_agent.epsilon_min / current_agent.epsilon) ** (1 / (0.8 * EPISODE_BUDGET))

                pool_agent = make_agent()
                pool_agent.load(prev_agent_path)
                pool_agent.epsilon = 0.0
                agent_pool.append(pool_agent)
                population, weights, recent_pool, old_pool = build_sampling(agent_pool)

                prev_agent_path = f"{save_path}_{generation}.pth"
                prev_agent.load(prev_agent_path)
                generation += 1
                total_episodes += episode_count
                episode_count = 0
                best_win_rate = 0.0
                recent_win_rates.clear()

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

def get_latest_and_pool(game, arch):
    agents = find_agents(game)
    if not agents:
        return None, None, [], 1
    arch_agents = [a for a in agents if a[0] == arch]
    if not arch_agents:
        return None, None, [], 1
    _, latest_gen, initial_path = arch_agents[-1]
    pool_paths = [a[2] for a in arch_agents[:-1]]
    save_path = os.path.join(GENERATIONAL_AGENTS_DIR, f"{game}_a{arch}")
    starting_generation = max(1, int(latest_gen) + 1)
    return initial_path, save_path, pool_paths, starting_generation

def main():
    print("Available games:")
    for name in GAMES:
        print(f"  {name}")
    game = None
    while game not in GAMES:
        game = input("Select game: ").strip().lower()
        if game not in GAMES:
            print(f"Invalid selection. Choose from: {', '.join(GAMES.keys())}")

    agents = find_agents(game)
    if not agents:
        print(f"No agents found for '{game}' in {GENERATIONAL_AGENTS_DIR}/. Add at least one agent before training.")
        return

    available_archs = sorted(set(a[0] for a in agents))
    print(f"Available architectures: {available_archs}")
    arch = None
    while arch not in available_archs:
        try:
            arch = int(input(f"Select architecture number: ").strip())
            if arch not in available_archs:
                print(f"Invalid selection. Choose from: {available_archs}")
        except ValueError:
            print("Enter a number.")

    initial_path, save_path, pool_paths, starting_generation = get_latest_and_pool(game, arch)
    if initial_path is None:
        print(f"No agents found for '{game}' architecture {arch}.")
        return

    print(f"\nDefault pool weights — random: {RANDOM_AGENT_PROBABILITY}, previous: {PREVIOUS_AGENT_PROBABILITY}, recent: {RECENT_AGENT_PROBABILITY}, old: {OLD_AGENT_PROBABILITY}")
    print(f"Default pool window (recent N gens): {POOL_WINDOW}")
    use_defaults = input("Use default weights? (y/n): ").strip().lower()
    if use_defaults == "y":
        random_weight = RANDOM_AGENT_PROBABILITY
        previous_weight = PREVIOUS_AGENT_PROBABILITY
        recent_weight = RECENT_AGENT_PROBABILITY
        old_weight = OLD_AGENT_PROBABILITY
        pool_window = POOL_WINDOW
    else:
        while True:
            try:
                random_weight = float(input("Random weight: ").strip())
                previous_weight = float(input("Previous agent weight: ").strip())
                recent_weight = float(input("Recent pool weight: ").strip())
                old_weight = float(input("Old pool weight: ").strip())
                pool_window = int(input(f"Pool window — number of recent gens (default {POOL_WINDOW}): ").strip() or POOL_WINDOW)
                break
            except ValueError:
                print("Enter numbers.")

    print(f"\nGame:             {game}")
    print(f"Architecture:     a{arch}")
    print(f"Initial agent:    {initial_path}")
    print(f"Starting gen:     {starting_generation}")
    print(f"Pool:             {pool_paths if pool_paths else 'empty (no prior checkpoints)'}")
    print(f"Weights:          random={random_weight}, previous={previous_weight}, recent={recent_weight}, old={old_weight}")
    print(f"Pool window:      {pool_window}")
    print(f"Save path:        {save_path}_{{generation}}.pth\n")

    config = GAMES[game]
    environment = config["env"]()
    train_agent(environment, config["sizes"], DeepClanker, initial_path, save_path, pool_paths,
                n_steps=config.get("n_steps", 1), buffer_size=config.get("buffer_size", 100000),
                starting_generation=starting_generation, random_weight=random_weight,
                previous_weight=previous_weight, recent_weight=recent_weight,
                old_weight=old_weight, pool_window=pool_window)

if __name__ == "__main__":
    main()