from clanker import Clanker
from tic_tac_toe_env import TicTacToeEnvironment

EVAL_EPISODES = 1
TRAIN_EPISODES = 100000

def main():
    env = TicTacToeEnvironment()
    agent = Clanker()
    agent.train(env, episodes=TRAIN_EPISODES)
    counts = agent.evaluate(env, opponent=agent, episodes=EVAL_EPISODES, verbose=True)
    print(counts)
    print(f"Win rate: {counts['wins'] / EVAL_EPISODES:.2%}, Loss rate: {counts['losses'] / EVAL_EPISODES:.2%}, Draw rate: {counts['draws'] / EVAL_EPISODES:.2%}")

if __name__ == "__main__":
    main()