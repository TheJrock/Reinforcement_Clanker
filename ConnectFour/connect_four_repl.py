import random

from clanker import DeepClanker
from ConnectFour.connect_four_env import ConnectFourEnvironment

def main():
    multiplayer = None
    while multiplayer not in ["1", "2"]:
        multiplayer = input("Play against (1) Human or (2) AI? Enter 1 or 2: ")
        if multiplayer == "1":
            human_vs_human()
        elif multiplayer == "2":
            human_vs_ai()
        else:
            print("Invalid input. Please enter 1 or 2.")

def human_vs_human():
    env = ConnectFourEnvironment()
    playing = True
    while playing:
        state = env.reset()
        done = False

        while not done:
            env.render()
            display_player = 1 if env.current_player == 1 else 2
            legal_actions = env.get_legal_actions()
            invalid_move = True
            while invalid_move:
                pit = int(input(f"Player {display_player}, enter your move (0-6): "))
                if pit not in legal_actions:
                    print("Invalid move. Try again.")
                else:
                    invalid_move = False
            state, reward, done = env.step(pit)

        env.render()
        if reward == 0:
            print("It's a draw!")
        else:
            winner_side = env.current_player if reward == 1 else -env.current_player
            print(f"Player {1 if winner_side == 1 else 2} wins!")

        play_again = input("Do you want to play again? (y/n): ").lower()
        if play_again != "y":
            playing = False

def human_vs_ai():
    env = ConnectFourEnvironment()
    agent = DeepClanker(42, 7)
    agent.load("generational_agents/connect_four_a1_36.pth")
    agent.epsilon = 0.0
    # counts = agent.evaluate(env, episodes=1000, verbose=False)
    # print(f"AI evaluation against random: {counts}")
    playing = True

    while playing:
        state = env.reset()
        done = False
        human_side = random.choice([1, -1])
        print("You go first." if human_side == 1 else "AI goes first.")

        while not done:
            env.render()
            legal_actions = env.get_legal_actions()
            if env.current_player == human_side:
                invalid_move = True
                while invalid_move:
                    pit = int(input("Your move (0-6): "))
                    if pit not in legal_actions:
                        print("Invalid move. Try again.")
                    else:
                        invalid_move = False
                state, reward, done = env.step(pit)
            else:
                action = agent.act(state, legal_actions)
                print(f"AI chooses slot {action}.")
                state, reward, done = env.step(action)

        env.render()
        if reward == 0:
            print("It's a draw!")
        else:
            winner_side = env.current_player if reward == 1 else -env.current_player
            if winner_side == human_side:
                print("Congratulations! You win!")
            else:
                print("AI wins! Better luck next time.")

        play_again = input("Do you want to play again? (y/n): ").lower()
        if play_again != "y":
            playing = False

if __name__ == "__main__":
    main()