from abc import ABC, abstractmethod


class GeneralEnvironment(ABC):

    # all inheritance classes should also have self.current_player which is 1 or -1, and should switch after each step

    @abstractmethod
    def reset(self):
        """Reset the game to the initial state. Return the starting state from the current player's perspective."""
        pass

    @abstractmethod
    def step(self, action):
        """Apply action. Return (next_state, reward, done) where next_state is from the next player's perspective."""
        pass

    @abstractmethod
    def get_legal_actions(self):
        """Return a list of legal actions for the current player."""
        pass

    @abstractmethod
    def is_terminal(self):
        """Return True if the game is over."""
        pass

    @abstractmethod
    def get_result(self):
        """Return the result of a terminal state: 1 for win, -1 for loss, 0 for draw. Should only be called when is_terminal() is True."""
        pass
    
    @abstractmethod
    def render(self):
        """Display the current board state."""
        pass
