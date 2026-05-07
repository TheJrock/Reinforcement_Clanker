from tic_tac_toe_env import TicTacToeEnvironment

def main():
    env = TicTacToeEnvironment()
    state = env.reset()
    done = False
    
    while not done:
        env.render()
        action = int(input("Enter your move (0-8): "))
        state, reward, done = env.step(action)
    
    env.render()
    if reward == 1:
        print("You win!")
    elif reward == -1:
        print("You lose!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    main()