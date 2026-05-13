from clanker import DeepClanker
from UltimateTicTacToe.ultimate_tic_tac_toe_env import UltimateTicTacToeEnvironment

EVAL_EPISODES = 1000
TRAIN_EPISODES = 100000

def main():
    env = UltimateTicTacToeEnvironment()
    epsilon_decay = .01 ** (1 / (0.8 * TRAIN_EPISODES))
    deep_agent = DeepClanker(91, 81, epsilon_decay=epsilon_decay, buffer_size=500000)
    deep_agent.train(env, episodes=TRAIN_EPISODES)
    deep_agent.save("UltimateTicTacToe/stashed_models/ultimate_tictactoe_deep.pth")
    counts = deep_agent.evaluate(env, opponent=None, episodes=EVAL_EPISODES, verbose=False)
    print(counts)
    print(f"Win rate: {counts['wins'] / EVAL_EPISODES:.2%}, Loss rate: {counts['losses'] / EVAL_EPISODES:.2%}, Draw rate: {counts['draws'] / EVAL_EPISODES:.2%}")

if __name__ == "__main__":
    main()