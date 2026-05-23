import random
from clanker import DeepClanker
from Backgammon.backgammon_env import BackgammonEnvironment


def decode_move(action, as_opponent=False):
    from_idx, to_idx = divmod(action, 26)
    if as_opponent:
        src = "Bar" if from_idx == 24 else f"Pt {24 - from_idx}"
        dst = "Off" if to_idx == 25 else f"Pt {24 - to_idx}"
    else:
        src = "Bar" if from_idx == 24 else f"Pt {from_idx + 1}"
        dst = "Off" if to_idx == 25 else f"Pt {to_idx + 1}"
    return f"{src} → {dst}"


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
    mode = None
    while mode not in ["1", "2"]:
        mode = input("Play against (1) Human or (2) AI? Enter 1 or 2: ")
    if mode == "1":
        human_vs_human()
    else:
        human_vs_ai()


def human_vs_human():
    env = BackgammonEnvironment()
    playing = True
    while playing:
        env.reset()
        done = False
        while not done:
            env.render()
            legal_actions = env.get_legal_actions()
            display_player = "Player 1" if env.current_player == 1 else "Player 2"
            if env.bar[0] > 0:
                print(f"{display_player} has {env.bar[0]} piece(s) on the bar — must re-enter first.")
            else:
                print(f"{display_player}'s turn.")
            action = get_human_move(legal_actions)
            _, _, done = env.step(action)

        env.render()
        winner = "Player 1" if env.current_player == 1 else "Player 2"
        print(f"{winner} wins!")

        if input("Play again? (y/n): ").lower() != "y":
            playing = False


def human_vs_ai():
    env = BackgammonEnvironment()
    agent = DeepClanker(56, 650)
    agent.load("Backgammon/stashed_models/backgammon_1.5.pth")
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
            if env.current_player == human_env_player:
                if env.bar[0] > 0:
                    print(f"You have {env.bar[0]} piece(s) on the bar — must re-enter first.")
                action = get_human_move(legal_actions)
            else:
                action = agent.act(state, legal_actions)
                print(f"AI plays: {decode_move(action, as_opponent=True)}")

            state, _, done = env.step(action)
            for skipped_player, dice in env.last_skips:
                who = "You" if skipped_player == human_env_player else "AI"
                print(f"{who} rolled {dice} but had no legal moves — turn skipped.")

        env.render()
        if env.current_player == human_env_player:
            print("Congratulations! You win!")
        else:
            print("AI wins! Better luck next time.")

        if input("Play again? (y/n): ").lower() != "y":
            playing = False


if __name__ == "__main__":
    main()
