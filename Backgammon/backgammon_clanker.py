from clanker import DeepClanker
from Backgammon.backgammon_env import BackgammonEnvironment

EVAL_EPISODES = 1000
TRAIN_EPISODES = [100000]

def main():
    env = BackgammonEnvironment()

    for batch_size in TRAIN_EPISODES:
        epsilon_decay = 0.01 ** (1 / (0.8 * batch_size))
        agent = DeepClanker(56, 650, epsilon_decay=epsilon_decay, n_steps=5)
        agent.train(env, episodes=batch_size)
        agent.save(f"Backgammon/stashed_models/backgammon_{batch_size}.pth")
        counts = agent.evaluate(env, opponent=None, episodes=EVAL_EPISODES, verbose=False)
        print(counts)
        print(f"Win rate: {counts['wins'] / EVAL_EPISODES:.2%}, Loss rate: {counts['losses'] / EVAL_EPISODES:.2%}, Draw rate: {counts['draws'] / EVAL_EPISODES:.2%}")

if __name__ == "__main__":
    main()
