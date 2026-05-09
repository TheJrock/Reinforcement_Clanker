import random

from clanker import DeepClanker
from Mancala.mancala_env import MancalaEnvironment

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
    env = MancalaEnvironment()
    playing = True
    while playing:
        state = env.reset()
        current_player = 1
        done = False

        while not done:
            env.render()
            legal_actions = env.get_legal_actions()
            invalid_move = True
            while invalid_move:
                pit = int(input(f"Player {current_player}, enter your move (0-5): "))
                action = pit if current_player == 1 else pit + 7
                if action not in legal_actions:
                    print("Invalid move. Try again.")
                else:
                    invalid_move = False
            state, reward, done = env.step(action)
            current_player = 2 if current_player == 1 else 1

        env.render()
        if reward == 1:
            print(f"Player {2 if current_player == 1 else 1} wins!")
        else:
            print("It's a draw!")

        play_again = input("Do you want to play again? (y/n): ").lower()
        if play_again != "y":
            playing = False

def human_vs_ai():
    env = MancalaEnvironment()
    state = env.reset()
    agent = DeepClanker(91, 81)
    agent.load("stashed_models/mancala_deep.pth")
    agent.epsilon = 0.0
    playing = True

    while playing:
        state = env.reset()
        done = False
        current_player = random.randint(1, 2)
        if current_player == 2:
            print("AI goes first.")
        else:
            print("You go first.")
        while not done:
            env.render()
            legal_actions = env.get_legal_actions()
            if current_player == 1:
                invalid_move = True
                while invalid_move:
                    pit = int(input(f"Player {current_player}, enter your move (0-5): "))
                    action = pit if current_player == 1 else pit + 7
                    if action not in legal_actions:
                        print("Invalid move. Try again.")
                    else:
                        invalid_move = False
            else:
                action = agent.act(state, legal_actions)
                print(f"AI chooses pit {action % 7}.")
            
            state, reward, done = env.step(action)
            current_player = 2 if current_player == 1 else 1
        
        env.render()
        if reward == 1:
            if current_player == 2:
                print("Congratulations! You win!")
            else:
                print("AI wins! Better luck next time.")
        else:
            print("It's a draw!")
        
        play_again = input("Do you want to play again? (y/n): ").lower()
        if play_again != "y":
            playing = False

if __name__ == "__main__":
    main()