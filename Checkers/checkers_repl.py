import random
from clanker import DeepClanker
from Checkers.checkers_env import CheckersEnvironment

def square_to_pos(square):
    row = square // 4
    col = (square % 4) * 2 + row % 2
    return row, col

def decode_move(action):
    start, end = divmod(action, 32)
    sr, sc = square_to_pos(start)
    er, ec = square_to_pos(end)
    return f"({sr},{sc}) → ({er},{ec})"

def get_human_move(legal_actions):
    print("Legal moves:")
    for i, action in enumerate(legal_actions, 1):
        print(f"  {i}. {decode_move(action)}")
    while True:
        try:
            choice = int(input(f"Enter move number (1-{len(legal_actions)}): "))
            if 1 <= choice <= len(legal_actions):
                return legal_actions[choice - 1]
        except ValueError:
            pass
        print("Invalid choice. Try again.")

def main():
    multiplayer = None
    while multiplayer not in ["1", "2"]:
        multiplayer = input("Play against (1) Human or (2) AI? Enter 1 or 2: ")
    if multiplayer == "1":
        human_vs_human()
    else:
        human_vs_ai()

def human_vs_human():
    env = CheckersEnvironment()
    playing = True
    while playing:
        env.reset()
        done = False
        while not done:
            env.render()
            legal_actions = env.get_legal_actions()
            display_player = "Player 1" if env.current_player == 1 else "Player 2"
            if env.chained_capture_index is not None:
                print(f"{display_player}, you must continue capturing from {square_to_pos(env.chained_capture_index)}.")
            else:
                print(f"{display_player}'s turn.")
            acting_env_player = env.current_player
            action = get_human_move(legal_actions)
            _, reward, done = env.step(action)

        env.render()
        if abs(reward) < 1:
            print("It's a draw!")
        else:
            winner = "Player 1" if acting_env_player == 1 else "Player 2"
            print(f"{winner} wins!")

        if input("Play again? (y/n): ").lower() != "y":
            playing = False

def human_vs_ai():
    env = CheckersEnvironment()
    agent = DeepClanker(33, 1024)
    agent.load("Checkers/stashed_models/checkers_3.5.pth")
    agent.epsilon = 0.0
    playing = True

    while playing:
        state = env.reset()
        done = False
        human_env_player = random.choice([1, -1])
        print("You go first." if human_env_player == 1 else "AI goes first.")

        while not done:
            env.render()
            legal_actions = env.get_legal_actions()
            acting_env_player = env.current_player

            if env.current_player == human_env_player:
                if env.chained_capture_index is not None:
                    print(f"You must continue capturing from {square_to_pos(env.chained_capture_index)}.")
                action = get_human_move(legal_actions)
            else:
                action = agent.act(state, legal_actions)
                print(f"AI plays: {decode_move(action)}")

            state, reward, done = env.step(action)

        env.render()
        if abs(reward) < 1:
            print("It's a draw!")
        elif acting_env_player == human_env_player:
            print("Congratulations! You win!")
        else:
            print("AI wins! Better luck next time.")

        if input("Play again? (y/n): ").lower() != "y":
            playing = False

if __name__ == "__main__":
    main()
