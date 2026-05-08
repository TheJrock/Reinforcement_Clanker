from clanker import DeepClanker
from tic_tac_toe_env import TicTacToeEnvironment

EVAL_EPISODES = 1000
TRAIN_EPISODES = 10000

def main():
    env = TicTacToeEnvironment()
    deep_agent = DeepClanker(9, 9)
    deep_agent.train(env, episodes=TRAIN_EPISODES)
    deep_agent.save("stashed_deep_models/ultimate_tictactoe_deep.pth")
    counts = deep_agent.evaluate(env, opponent=None, episodes=EVAL_EPISODES)
    print(counts)
    print(f"Win rate: {counts['wins'] / EVAL_EPISODES:.2%}, Loss rate: {counts['losses'] / EVAL_EPISODES:.2%}, Draw rate: {counts['draws'] / EVAL_EPISODES:.2%}")

if __name__ == "__main__":
    main()