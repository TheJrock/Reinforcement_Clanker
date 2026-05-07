from ultimate_tic_tac_toe_env import UltimateTicTacToeEnvironment

def main():
    env = UltimateTicTacToeEnvironment()
    state = env.reset()
    done = False
    
    while not done:
        env.render()
        board_num = state[-1]
        if board_num == -1:
            board_num = int(input("You can play in any board. Enter the board number (0-8): "))
        cell = int(input(f"Playing in board {board_num}, enter your move (0-8): "))
        state, reward, done = env.step(board_num * 9 + cell)
    
    env.render()
    if reward == 1:
        print("You win!")
    elif reward == -1:
        print("You lose!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    main()