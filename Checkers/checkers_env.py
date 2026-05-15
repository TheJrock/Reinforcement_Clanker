from general_environment import GeneralEnvironment

class CheckersEnvironment(GeneralEnvironment):

    NEIGHBORS = {0: (None, 4, None, None),
                 1: (4, 5, None, None),
                 2: (5, 6, None, None),
                 3: (6, 7, None, None),
                 4: (8, 9, 0, 1),
                 5: (9, 10, 1, 2),
                 6: (10, 11, 2, 3),
                 7: (11, None, 3, None),
                 8: (None, 12, None, 4),
                 9: (12, 13, 4, 5),
                 10: (13, 14, 5, 6),
                 11: (14, 15, 6, 7),
                 12: (16, 17, 8, 9),
                 13: (17, 18, 9, 10),
                 14: (18, 19, 10, 11),
                 15: (19, None, 11, None),}

    def __init__(self):
        self.board = [1] * 12 + [0] * 8 + [-1] * 12
        self.current_player = 1
        self._init_board()

    def _init_board(self):
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row][col] = -1  # opponent (White)
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row][col] = 1  # current player (Black)

    def reset(self):
        self.board = [[0] * self.board_size for _ in range(self.board_size)]
        self.current_player = 1
        self._init_board()
        return self._to_agent_state()

    def _to_agent_state(self):
        return tuple((cell + 1) / 2 for row in self.board for cell in row)

    def step(self, action):
        # Placeholder: Implement move logic, captures, kinging, and game over conditions
        raise NotImplementedError("Move logic not implemented yet")