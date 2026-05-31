import random

from clanker import DeepClanker
from UltimateTicTacToe.ultimate_tic_tac_toe_env import UltimateTicTacToeEnvironment

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
    env = UltimateTicTacToeEnvironment()
    playing = True
    while playing:
        state = env.reset()
        current_player = 1
        done = False

        while not done:
            env.render()
            legal_actions = env.get_legal_actions()
            invalid_move = True
            board_num = state[-1]
            while invalid_move:
                if board_num == 9:
                    board_num = int(input("You can play in any board. Enter the board number (0-8): "))
                cell = int(input(f"Playing in board {board_num}, enter your move (0-8): "))
                action = board_num * 9 + cell
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
    env = UltimateTicTacToeEnvironment()
    agent = DeepClanker(91, 81)
    agent.load("generational_agents/ultimate_tictactoe_a1_0.1.pth")
    agent.epsilon = 0.0
    playing = True

    while playing:
        state = env.reset()
        done = False
        current_player = random.randint(1, 2)
        if current_player == 2:
            print("You are O, AI is X. AI goes first.")
        else:
            print("You are X, AI is O. You go first.")
        while not done:
            env.render()
            legal_actions = env.get_legal_actions()
            if current_player == 1:
                invalid_move = True
                board_num = state[-1]
                while invalid_move:
                    if board_num == 9:
                        board_num = int(input("You can play in any board. Enter the board number (0-8): "))
                    cell = int(input(f"Playing in board {board_num}, enter your move (0-8): "))
                    action = board_num * 9 + cell
                    if action not in legal_actions:
                        print("Invalid move. Try again.")
                    else:
                        invalid_move = False
            else:
                action = agent.act(state, legal_actions)
                print(f"AI chooses cell {action % 9} in board {action // 9}")
            
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