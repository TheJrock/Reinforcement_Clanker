import os
import random
from clanker import DeepClanker
from generational_training import GAMES, find_agents

def short_name(path):
    return os.path.basename(path).replace(".pth", "")

def pick_from_list(prompt, options):
    for i, opt in enumerate(options):
        print(f"  [{i}] {opt}")
    while True:
        try:
            idx = int(input(prompt))
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print(f"Enter a number between 0 and {len(options) - 1}.")

def main():
    print("Available games:")
    game_names = list(GAMES.keys())
    game_idx = pick_from_list("Select game: ", game_names)
    game = game_names[game_idx]

    agents_meta = find_agents(game)
    if not agents_meta:
        print(f"No agents found for '{game}'.")
        return

    config = GAMES[game]
    n_steps = config.get("n_steps", 1)
    buffer_size = config.get("buffer_size", 100000)
    names = [short_name(path) for _, _, path in agents_meta]

    print("\nSelect agent 1:")
    idx1 = pick_from_list("Agent 1: ", names)

    print("\nSelect agent 2:")
    idx2 = pick_from_list("Agent 2: ", names)

    _, _, path1 = agents_meta[idx1]
    _, _, path2 = agents_meta[idx2]

    def load(path):
        agent = DeepClanker(config["sizes"][0], config["sizes"][1], n_steps=n_steps, buffer_size=buffer_size)
        agent.load(path)
        agent.epsilon = 0.0
        return agent

    print(f"\nLoading agents...")
    agent1 = load(path1)
    agent2 = load(path2)

    env = config["env"]()
    state = env.reset()
    done = False
    move = 0

    # randomly assign sides, just like clanker.play() does.
    # the board render always shows player 1's pieces as X, so we match
    # the display labels to the actual player assignment.
    a1_side = random.choice([1, -1])
    a2_side = -a1_side
    # player 1 always renders as X, player -1 always renders as O
    side_symbols = {1: "X", -1: "O"}
    side_labels = {
        a1_side: (names[idx1], side_symbols[a1_side]),
        a2_side: (names[idx2], side_symbols[a2_side]),
    }

    print(f"\n=== {names[idx1]} vs {names[idx2]} ===")
    print(f"  {names[idx1]} plays as {side_symbols[a1_side]}")
    print(f"  {names[idx2]} plays as {side_symbols[a2_side]}")
    print("\nPress Enter to step through moves (or 'q' to quit).\n")

    while not done:
        acting_player = env.current_player
        agent_name, symbol = side_labels[acting_player]
        legal_actions = env.get_legal_actions()

        if acting_player == a1_side:
            action = agent1.act(state, legal_actions)
        else:
            action = agent2.act(state, legal_actions)

        state, reward, done = env.step(action)
        move += 1

        row, col = divmod(action, env.board_size) if hasattr(env, "board_size") else (None, None)
        move_str = f"({row}, {col})" if row is not None else str(action)

        key = input(f"Move {move:>3} — {agent_name} [{symbol}] plays {move_str}  [Enter / q]: ")
        if key.strip().lower() == "q":
            print("Quit.")
            return

        if game == "mancala":
            saved = env.current_player
            env.current_player = 1
            env.render()
            env.current_player = saved
        else:
            env.render()

    if reward == 0:
        print("\nResult: Draw")
    else:
        winner_side = env.current_player if reward == 1 else -env.current_player
        winner_name, winner_symbol = side_labels[winner_side]
        print(f"\nResult: {winner_name} [{winner_symbol}] wins")

if __name__ == "__main__":
    main()
