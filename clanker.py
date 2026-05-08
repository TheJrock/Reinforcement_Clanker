import random
import torch
import torch.nn as nn
import collections
import json
from abc import ABC, abstractmethod
from tqdm import tqdm

class Clanker(ABC):
    def __init__(self, epsilon=1.0, alpha=0.1, gamma=0.99, epsilon_min=0.01, epsilon_decay=0.99999):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def train(self, environment, episodes):
        for episode in tqdm(range(episodes)):
            self.play(environment, learning=True, opponent=None, verbose=False)
            self.update_epsilon()

    def evaluate(self, environment, episodes, opponent=None, verbose=False):
        current_epsilon = self.epsilon
        if opponent:
            opponent_epsilon = opponent.epsilon
            opponent.epsilon = 0.0
        self.epsilon = 0.0
        counts = {"wins": 0, "losses": 0, "draws": 0}
        for episode in tqdm(range(episodes)):
            reward, turn = self.play(environment, learning=False, opponent=opponent, verbose=verbose)

            if reward == 0:
                counts["draws"] += 1
            elif (turn - 1) % 2 == 0:
                counts["wins"] += 1
            else:
                counts["losses"] += 1

        self.epsilon = current_epsilon
        if opponent:
            opponent.epsilon = opponent_epsilon

        return counts

    def play(self, environment, learning, opponent=None, verbose=False):
        state = environment.reset()
        done = False
        turn = random.randint(0, 1)
        prev_state = None
        prev_action = None

        while not done:
            legal_actions = environment.get_legal_actions()
            if turn % 2 == 0:
                action = self.act(state, legal_actions)
            elif opponent:
                action = opponent.act(state, legal_actions)
            else:
                action = random.choice(legal_actions)
            next_state, reward, done = environment.step(action)
            if learning:
                if done and reward == 1 and prev_state is not None:
                    self.learn(prev_state, prev_action, -1, state, True)
                self.learn(state, action, reward, next_state, done)
            prev_state = state
            prev_action = action
            state = next_state
            if verbose: environment.render()

        return reward, turn

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def act(self, state, legal_actions):
        if not legal_actions:
            raise ValueError("No legal actions available")
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        return self.get_best_action(state, legal_actions)

    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def get_best_action(self, state, legal_actions):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass


class TableClanker(Clanker):
    def __init__(self, epsilon=1.0, alpha=0.1, gamma=0.99, epsilon_min=0.01, epsilon_decay=0.99999):
        super().__init__(epsilon, alpha, gamma, epsilon_min, epsilon_decay)
        self.q_table = {}

    
    def get_best_action(self, state, legal_actions):
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in legal_actions}
        
        best_action = max(legal_actions, key=lambda action: self.q_table[state].get(action, 0.0))
        return best_action

    def learn(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            max_next_q = max(self.q_table.get(next_state, {}).values(), default=0.0)
            target = reward + self.gamma * (-max_next_q)
        current_q = self.q_table.get(state, {}).get(action, 0.0)
        self.q_table.setdefault(state, {})[action] = current_q + self.alpha * (target - current_q)

    def save(self, path):
        serializable = {str(state): actions for state, actions in self.q_table.items()}
        with open(path, 'w') as f:
            json.dump(serializable, f)

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        self.q_table = {tuple(int(x) for x in state[1:-1].split(', ')): {int(a): v for a, v in actions.items()} for state, actions in data.items()}

class DeepClanker(Clanker):
    def __init__(self, input_size, output_size, hidden_layers=[128, 128], batch_size=32, update_freq=1000, min_replay_size=None, epsilon=1.0, alpha=0.0001, gamma=0.99, epsilon_min=0.01, epsilon_decay=0.99999, gpu_device=None):
        super().__init__(epsilon, alpha, gamma, epsilon_min, epsilon_decay)
        self.network = HoloNet(input_size, output_size, hidden_layers)
        self.target_network = HoloNet(input_size, output_size, hidden_layers)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = collections.deque(maxlen=100000)
        self.steps = 0
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.min_replay_size = min_replay_size or batch_size

        if gpu_device is None:
            if torch.backends.mps.is_available():
                gpu_device = "mps"
                print("Using MPS (Apple Silicon GPU)")
            elif torch.cuda.is_available():
                gpu_device = "cuda"
                print("Using CUDA (NVIDIA GPU)")
            else:
                gpu_device = "cpu"
                print("No GPU found, using CPU")
        self.gpu_device = torch.device(gpu_device)
        self.network.to(self.gpu_device)
        self.target_network.to(self.gpu_device)

    def get_best_action(self, state, legal_actions):
        q_values = self.network(torch.tensor(state, dtype=torch.float32).to(self.gpu_device))
        best_action = max(legal_actions, key=lambda action: q_values[action].item())
        return best_action

    def learn(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) < self.min_replay_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32).to(self.gpu_device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.gpu_device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.gpu_device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.gpu_device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.gpu_device)
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(dim=1).values
            targets = rewards + self.gamma * (-max_next_q) * (1 - dones)
        predicted_q = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(predicted_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.gpu_device))
        self.target_network.load_state_dict(self.network.state_dict())

class HoloNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=[128, 128]):
        super(HoloNet, self).__init__()
        sizes = [input_size] + hidden_layers + [output_size]
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)