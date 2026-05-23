import os
import argparse
from clanker import DeepClanker
from generational_training import GAMES, find_agents

EVAL_EPISODES = 200

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
    args = parser.parse_args()

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

    agents = [(short_name(path), load_agent(path, config["sizes"], n_steps, buffer_size))
              for _, _, path in agents_meta]
    names = [a[0] for a in agents]
    n = len(agents)

    env_class = config["env"]
    environment = env_class()

    # results[i][j] = (wins, losses, draws) of agent i vs agent j
    results = [[None] * n for _ in range(n)]

    total_pairs = n * (n - 1) // 2
    print(f"\nRunning {total_pairs} matchup(s), {EVAL_EPISODES} games each...\n")

    for i in range(n):
        for j in range(i + 1, n):
            name_i, agent_i = agents[i]
            name_j, agent_j = agents[j]
            print(f"  {name_i} vs {name_j}")
            counts = agent_i.evaluate(environment, opponent=agent_j, episodes=EVAL_EPISODES, verbose=False)
            results[i][j] = counts
            results[j][i] = {"wins": counts["losses"], "losses": counts["wins"], "draws": counts["draws"]}

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
            if i == j:
                row += f" {'---':>{col_w}} |"
            else:
                r = results[i][j]
                win_rate = r["wins"] / EVAL_EPISODES
                row += f" {win_rate:>{col_w}.0%} |"
        print(row)

    # Print total win rates (sum of wins across all opponents)
    print("\n--- Overall Win Rate vs All Opponents ---\n")
    totals = []
    for i, (name_i, _) in enumerate(agents):
        total_wins = sum(results[i][j]["wins"] for j in range(n) if i != j)
        total_games = EVAL_EPISODES * (n - 1)
        totals.append((name_i, total_wins / total_games))

    totals.sort(key=lambda x: -x[1])
    for name, rate in totals:
        bar = "█" * int(rate * 40)
        print(f"  {name:>{header_w}}: {rate:>6.1%}  {bar}")

if __name__ == "__main__":
    main()
