from clanker import DeepClanker
from Othello.othello_env import OthelloEnvironment

EVAL_EPISODES = 1000
TRAIN_EPISODES = 1000

def main():
    env = OthelloEnvironment()

    for batch_size in [1000, 10000, 100000]:
        epsilon_decay = .01 ** (1 / (0.8 * batch_size))
        deep_agent = DeepClanker(env.board_size ** 2, env.board_size ** 2, epsilon_decay=epsilon_decay)
        deep_agent.train(env, episodes=batch_size)
        deep_agent.save(f"Othello/stashed_models/othello_deep_{batch_size}.pth")
        # deep_agent.load("Othello/stashed_models/othello_1.4.pth")
        counts = deep_agent.evaluate(env, opponent=None, episodes=EVAL_EPISODES, verbose=False)
        print(counts)
        print(f"Win rate: {counts['wins'] / EVAL_EPISODES:.2%}, Loss rate: {counts['losses'] / EVAL_EPISODES:.2%}, Draw rate: {counts['draws'] / EVAL_EPISODES:.2%}")

if __name__ == "__main__":
    main()