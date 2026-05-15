from general_environment import GeneralEnvironment

class CheckersEnvironment(GeneralEnvironment):

    NEIGHBORS = {
                0: (None, 4, None, None),
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
                15: (19, None, 11, None),
                16: (None, 20, None, 12),
                17: (20, 21, 12, 13),
                18: (21, 22, 13, 14),
                19: (22, 23, 14, 15),
                20: (24, 25, 16, 17),
                21: (25, 26, 17, 18),
                22: (26, 27, 18, 19),
                23: (27, None, 19, None),
                24: (None, 28, None, 20),
                25: (28, 29, 20, 21),
                26: (29, 30, 21, 22),
                27: (30, 31, 22, 23),
                28: (None, None, 24, 25),
                29: (None, None, 25, 26),
                30: (None, None, 26, 27),
                31: (None, None, 27, None)
                }

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