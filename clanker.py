import random
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class Clanker(ABC):
    def __init__(self, epsilon=1.0, alpha=0.1, gamma=0.99, epsilon_min=0.01, epsilon_decay=0.99999):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def train(self, environment, episodes):
        for episode in range(episodes):
            self.play(environment, learning=True, opponent=None, verbose=False)
            self.update_epsilon()

    def evaluate(self, environment, episodes, opponent=None, verbose=False):
        current_epsilon = self.epsilon
        if opponent:
            opponent_epsilon = opponent.epsilon
            opponent.epsilon = 0.0
        self.epsilon = 0.0
        counts = {"wins": 0, "losses": 0, "draws": 0}
        for episode in range(episodes):
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
        pass

    def load(self, path):
        pass

class DeepClanker(Clanker):
    def __init__(self, epsilon=1.0, alpha=0.1, gamma=0.99, epsilon_min=0.01, epsilon_decay=0.99999):
        super().__init__(epsilon, alpha, gamma, epsilon_min, epsilon_decay)
        # Placeholder for neural network initialization

    def get_best_action(self, state, legal_actions):
        # Placeholder for neural network action selection
        return random.choice(legal_actions)

    def learn(self, state, action, reward, next_state, done):
        # Placeholder for neural network training
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass