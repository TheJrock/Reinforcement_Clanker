from clanker import DeepClanker
from Mancala.mancala_env import MancalaEnvironment

EVAL_EPISODES = 1000
TRAIN_EPISODES = [1, 10, 100, 1000, 10000, 100000]

def main():
    env = MancalaEnvironment()

    for batch_size in TRAIN_EPISODES:
        epsilon_decay = .01 ** (1 / (0.8 * batch_size))
        deep_agent = DeepClanker(14, 6, epsilon_decay=epsilon_decay, n_steps=1)
        deep_agent.train(env, episodes=batch_size)
        deep_agent.save(f"Mancala/stashed_models/mancala_{batch_size}.pth")
        counts = deep_agent.evaluate(env, opponent=None, episodes=EVAL_EPISODES, verbose=False)
        print(counts)
        print(f"Win rate: {counts['wins'] / EVAL_EPISODES:.2%}, Loss rate: {counts['losses'] / EVAL_EPISODES:.2%}, Draw rate: {counts['draws'] / EVAL_EPISODES:.2%}")

if __name__ == "__main__":
    main()