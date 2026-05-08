from clanker import TableClanker, DeepClanker
from tic_tac_toe_env import TicTacToeEnvironment

EVAL_EPISODES = 1000
TRAIN_EPISODES = 100000

def main():
    env = TicTacToeEnvironment()
    deep_agent = DeepClanker(9, 9)
    # deep_agent.train(env, episodes=TRAIN_EPISODES)
    deep_agent.load("stashed_deep_models/tictactoe_deep.pth")
    table_agent = TableClanker()
    table_agent.train(env, episodes=TRAIN_EPISODES)
    counts = deep_agent.evaluate(env, opponent=table_agent, episodes=EVAL_EPISODES)
    print(counts)
    print(f"Win rate: {counts['wins'] / EVAL_EPISODES:.2%}, Loss rate: {counts['losses'] / EVAL_EPISODES:.2%}, Draw rate: {counts['draws'] / EVAL_EPISODES:.2%}")

if __name__ == "__main__":
    main()