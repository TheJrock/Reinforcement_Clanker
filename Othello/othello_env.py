import numpy as np

from general_environment import GeneralEnvironment

_DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class OthelloEnvironment(GeneralEnvironment):

    def __init__(self, board_size=8):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1
        self._init_board()

    def _init_board(self):
        mid = self.board_size // 2
        self.board[mid - 1, mid - 1] = -1  # opponent (White)
        self.board[mid,     mid    ] = -1  # opponent (White)
        self.board[mid - 1, mid    ] =  1  # current player (Black)
        self.board[mid,     mid - 1] =  1  # current player (Black)

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self._init_board()
        return self._to_agent_state()

    def _to_agent_state(self):
        return tuple((self.board.flatten() + 1) / 2)

    def step(self, action):
        row, col = divmod(action, self.board_size)
        self.board[row, col] = 1
        for r, c in self._get_flips(row, col):
            self.board[r, c] = 1

        self.board *= -1
        self.current_player *= -1

        if not self.get_legal_actions():
            # Opponent has no moves — flip back, current player plays again
            self.board *= -1
            self.current_player *= -1
            if not self.get_legal_actions():
                # Neither player has moves — game over, count pieces
                own = int(np.sum(self.board == 1))
                opp = int(np.sum(self.board == -1))
                if own > opp:
                    reward = 1
                elif own < opp:
                    reward = -1
                else:
                    reward = 0
                return self._to_agent_state(), reward, True

        return self._to_agent_state(), 0, False

    def _get_flips(self, row, col):
        flips = []
        for dr, dc in _DIRECTIONS:
            candidates = []
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == -1:
                candidates.append((r, c))
                r += dr
                c += dc
            if candidates and 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == 1:
                flips.extend(candidates)
        return flips

    def get_legal_actions(self):
        return [
            int(row * self.board_size + col)
            for row, col in np.argwhere(self.board == 0)
            if self._get_flips(int(row), int(col))
        ]

    def is_terminal(self):
        if self.get_legal_actions():
            return False
        self.board *= -1
        opponent_has_moves = bool(self.get_legal_actions())
        self.board *= -1
        return not opponent_has_moves

    def get_result(self):
        own = int(np.sum(self.board == 1))
        opp = int(np.sum(self.board == -1))
        if own > opp:
            return 1
        elif own < opp:
            return -1
        return 0

    def render(self):
        symbols = {0: ".", 1: "X", -1: "O"}
        print("  " + " ".join(str(c) for c in range(self.board_size)))
        for r in range(self.board_size):
            row_str = " ".join(symbols[v * self.current_player] for v in self.board[r])
            print(f"{r} {row_str}")
        print()
