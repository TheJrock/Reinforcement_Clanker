import random

from clanker import DeepClanker
from Gomoku.gomoku_env import GomokuEnvironment

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
    env = GomokuEnvironment()
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
                row = int(input(f"Player {current_player}, enter row (0-{env.board_size - 1}): "))
                col = int(input(f"Player {current_player}, enter column (0-{env.board_size - 1}): "))
                action = row * env.board_size + col
                if action not in legal_actions:
                    print("Invalid move. Try again.")
                else:
                    invalid_move = False
            state, reward, done = env.step(action)
            current_player = 2 if current_player == 1 else 1

        env.render()
        if reward == 1:
            winner = 2 if current_player == 1 else 1
            print(f"Player {winner} wins!")
        else:
            print("It's a draw!")

        play_again = input("Do you want to play again? (y/n): ").lower()
        if play_again != "y":
            playing = False

def human_vs_ai():
    env = GomokuEnvironment()
    agent = DeepClanker(env.board_size ** 2, env.board_size ** 2)
    agent.load("generational_agents/gomoku_a1_45.pth")
    agent.epsilon = 0.0
    playing = True

    while playing:
        state = env.reset()
        done = False
        current_player = random.randint(1, 2)
        print("You go first." if current_player == 1 else "AI goes first.")
        while not done:
            env.render()
            legal_actions = env.get_legal_actions()
            if current_player == 1:
                invalid_move = True
                while invalid_move:
                    row = int(input(f"Player {current_player}, enter row (0-{env.board_size - 1}): "))
                    col = int(input(f"Player {current_player}, enter column (0-{env.board_size - 1}): "))
                    action = row * env.board_size + col
                    if action not in legal_actions:
                        print("Invalid move. Try again.")
                    else:
                        invalid_move = False
            else:
                action = agent.act(state, legal_actions)
                print(f"AI chooses row {action // env.board_size} and column {action % env.board_size}")
            
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