import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from clanker import DeepClanker

AGENT_CONFIGS = {
    "connect_four": {
        "model_path": "generational_agents/connect_four_a1_36.pth",
        "in_out_size": (42, 7),
    },
    "mancala": {
        "model_path": "generational_agents/mancala_a2_1.pth",
        "in_out_size": (14, 6),
        "kwargs": {"loss": "huber", "grad_clip": 10.0},
    },
    "checkers": {
        "model_path": "generational_agents/checkers_a1_1.pth",
        "in_out_size": (33, 1024),
        "kwargs": {"n_steps": 5},
    },
}

_agents = {}

def load_agents():
    for game, config in AGENT_CONFIGS.items():
        model_path = config["model_path"]
        in_size, out_size = config["in_out_size"]
        kwargs = config.get("kwargs", {})
        agent = DeepClanker(in_size, out_size, **kwargs)
        agent.load(os.path.join(ROOT, model_path))
        agent.epsilon = 0.0
        _agents[game] = agent
    return _agents

load_agents()

def get_action(game_name, state, legal_actions):
    return _agents[game_name].get_best_action(state, legal_actions)