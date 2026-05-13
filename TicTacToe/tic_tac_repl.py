import random

from clanker import DeepClanker, TableClanker
from TicTacToe.tic_tac_toe_env import TicTacToeEnvironment

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
    env = TicTacToeEnvironment()
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
                action = int(input(f"Player {current_player}, enter your move (0-8): "))
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
    env = TicTacToeEnvironment()
    agent = TableClanker()
    agent.train(env, episodes=100000)
    # agent = DeepClanker(9, 9)
    # agent.load("TicTacToe/stashed_models/tictactoe_2.3.pth")
    agent.epsilon = 0.0
    playing = True

    while playing:
        state = env.reset()
        done = False
        current_player = random.randint(1, 2)
        while not done:
            env.render()
            if current_player == 1:
                action = int(input("Your move (0-8): "))
            else:
                legal_actions = env.get_legal_actions()
                action = agent.act(state, legal_actions)
                print(f"AI chooses: {action}")
            
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
