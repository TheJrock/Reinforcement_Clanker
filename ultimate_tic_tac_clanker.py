from clanker import DeepClanker
from ultimate_tic_tac_toe_env import UltimateTicTacToeEnvironment

EVAL_EPISODES = 100
TRAIN_EPISODES = 10000

def main():
    env = UltimateTicTacToeEnvironment()
    deep_agent = DeepClanker(91, 81)
    deep_agent.train(env, episodes=TRAIN_EPISODES)
    deep_agent.save("stashed_deep_models/ultimate_tictactoe_deep.pth")
    counts = deep_agent.evaluate(env, opponent=None, episodes=EVAL_EPISODES, verbose=False)
    print(counts)
    print(f"Win rate: {counts['wins'] / EVAL_EPISODES:.2%}, Loss rate: {counts['losses'] / EVAL_EPISODES:.2%}, Draw rate: {counts['draws'] / EVAL_EPISODES:.2%}")

if __name__ == "__main__":
    main()