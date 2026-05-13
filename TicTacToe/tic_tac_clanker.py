from clanker import TableClanker, DeepClanker
from TicTacToe.tic_tac_toe_env import TicTacToeEnvironment

EVAL_EPISODES = 1000
TRAIN_EPISODES = 100000

def main():
    env = TicTacToeEnvironment()
    table_agent = TableClanker()
    epsilon_decay = .01 ** (1 / (0.8 * TRAIN_EPISODES))
    deep_agent = DeepClanker(9, 9, epsilon_decay=epsilon_decay)
    # table_agent.train(env, episodes=TRAIN_EPISODES)
    deep_agent.train(env, episodes=TRAIN_EPISODES)
    deep_agent.save("TicTacToe/stashed_models/tictactoe_deep.pth")
    counts = deep_agent.evaluate(env, opponent=None, episodes=EVAL_EPISODES, verbose=False)
    print(counts)
    print(f"Win rate: {counts['wins'] / EVAL_EPISODES:.2%}, Loss rate: {counts['losses'] / EVAL_EPISODES:.2%}, Draw rate: {counts['draws'] / EVAL_EPISODES:.2%}")

if __name__ == "__main__":
    main()