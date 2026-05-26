import os
import argparse
from tqdm import tqdm
from clanker import DeepClanker
from generational_training import GAMES, find_agents

def load_agent(path, sizes, n_steps, buffer_size):
    agent = DeepClanker(sizes[0], sizes[1], n_steps=n_steps, buffer_size=buffer_size)
    agent.load(path)
    agent.epsilon = 0.0
    return agent

def short_name(path):
    return os.path.basename(path).replace(".pth", "")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=None, metavar="N",
                        help="Only evaluate the N most recent agents")
    parser.add_argument("--episodes", type=int, default=200, metavar="N",
                        help="Games per matchup (default: 200)")
    args = parser.parse_args()
    eval_episodes = args.episodes

    print("Available games:")
    for name in GAMES:
        print(f"  {name}")
    game = None
    while game not in GAMES:
        game = input("Select game: ").strip().lower()
        if game not in GAMES:
            print(f"Invalid selection. Choose from: {', '.join(GAMES.keys())}")

    agents_meta = find_agents(game)
    if not agents_meta:
        print(f"No agents found for '{game}'.")
        return

    if args.last is not None:
        agents_meta = agents_meta[-args.last:]

    config = GAMES[game]
    n_steps = config.get("n_steps", 1)
    buffer_size = config.get("buffer_size", 100000)

    print(f"\nFound {len(agents_meta)} agent(s):")
    for _, _, path in agents_meta:
        print(f"  {short_name(path)}")

    print("Loading agents...")
    agents = [(short_name(path), load_agent(path, config["sizes"], n_steps, buffer_size))
              for _, _, path in agents_meta]
    # Second copies for self-matchups, loaded upfront to avoid mid-run prints
    agents_self = [load_agent(path, config["sizes"], n_steps, buffer_size)
                   for _, _, path in agents_meta]
    names = [a[0] for a in agents]
    n = len(agents)

    env_class = config["env"]
    environment = env_class()

    # results[i][j] = (wins, losses, draws) of agent i vs agent j
    results = [[None] * n for _ in range(n)]

    total_pairs = n * (n - 1) // 2 + n
    print(f"\nRunning {total_pairs} matchup(s), {eval_episodes} games each...")

    with tqdm(total=total_pairs * eval_episodes) as pbar:
        for i in range(n):
            for j in range(i, n):
                name_i, agent_i = agents[i]
                name_j, agent_j = agents[j]
                if i == j:
                    agent_j = agents_self[j]
                pbar.set_description(f"{name_i} vs {name_j}")
                combined = {"wins": 0, "losses": 0, "draws": 0}
                for _ in range(eval_episodes):
                    result = agent_i.evaluate(environment, opponent=agent_j, episodes=1, verbose=False, load_bar=False)
                    combined["wins"] += result["wins"]
                    combined["losses"] += result["losses"]
                    combined["draws"] += result["draws"]
                    pbar.update(1)
                results[i][j] = combined
                if i != j:
                    results[j][i] = {"wins": combined["losses"], "losses": combined["wins"], "draws": combined["draws"]}

    # Print win rate matrix
    col_w = max(len(n) for n in names) + 2
    header_w = max(len(n) for n in names)

    print("\n--- Win Rate Matrix (row agent vs column agent) ---\n")
    header = f"{'':>{header_w}} |" + "".join(f" {n:>{col_w}} |" for n in names)
    print(header)
    print("-" * len(header))

    for i, (name_i, _) in enumerate(agents):
        row = f"{name_i:>{header_w}} |"
        for j in range(n):
            r = results[i][j]
            win_rate = r["wins"] / eval_episodes
            row += f" {win_rate:>{col_w}.0%} |"
        print(row)

    # Print total win rates (sum of wins across all opponents)
    print("\n--- Overall Win Rate vs All Opponents ---\n")
    totals = []
    for i, (name_i, _) in enumerate(agents):
        total_wins = sum(results[i][j]["wins"] for j in range(n) if i != j)
        total_games = eval_episodes * (n - 1)
        totals.append((name_i, total_wins / total_games))

    totals.sort(key=lambda x: -x[1])
    for name, rate in totals:
        bar = "█" * int(rate * 40)
        print(f"  {name:>{header_w}}: {rate:>6.1%}  {bar}")

if __name__ == "__main__":
    main()
