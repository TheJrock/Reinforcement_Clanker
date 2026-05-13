from clanker import DeepClanker
from ConnectFour.connect_four_env import ConnectFourEnv

EVAL_EPISODES = 1000
TRAIN_EPISODES = 100000

def main():
    env = ConnectFourEnv()
    epsilon_decay = .01 ** (1 / (0.8 * TRAIN_EPISODES))
    deep_agent = DeepClanker(42, 7, epsilon_decay=epsilon_decay)
    deep_agent.train(env, episodes=TRAIN_EPISODES)
    deep_agent.save("ConnectFour/stashed_models/connect_four_deep.pth")
    counts = deep_agent.evaluate(env, opponent=None, episodes=EVAL_EPISODES, verbose=False)
    print(counts)
    print(f"Win rate: {counts['wins'] / EVAL_EPISODES:.2%}, Loss rate: {counts['losses'] / EVAL_EPISODES:.2%}, Draw rate: {counts['draws'] / EVAL_EPISODES:.2%}")

if __name__ == "__main__":
    main()