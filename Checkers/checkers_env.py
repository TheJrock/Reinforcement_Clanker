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
        self.chained_capture_index = None

    def reset(self):
        self.board = [1] * 12 + [0] * 8 + [-1] * 12
        self.current_player = 1
        self.chained_capture_index = None
        return self._to_agent_state()

    def _to_agent_state(self):
        board_state = tuple((cell + 1) / 2 for cell in self.board)
        chain = 0 if self.chained_capture_index is None else (self.chained_capture_index + 1) / 32
        return board_state + (chain,)

    def step(self, action):
        start_index, end_index = divmod(action, 32)

    def get_legal_actions(self):
        forward_directions = (0, 1) if self.current_player == 1 else (2, 3)
        starting_indices = [i for i, cell in enumerate(self.board) if cell == self.current_player]
        if self.chained_capture_index is not None:
            starting_indices = [self.chained_capture_index]
        simple_moves = []
        captures = []
        for start in starting_indices:
            for direction in forward_directions:
                neighbor = self.NEIGHBORS[start][direction]
                if neighbor is not None and self.board[neighbor] == 0:
                    simple_moves.append(start * 32 + neighbor)
                elif neighbor is not None and self.board[neighbor] == -self.current_player:
                    jump_neighbor = self.NEIGHBORS[neighbor][direction]
                    if jump_neighbor is not None and self.board[jump_neighbor] == 0:
                        captures.append(start * 32 + jump_neighbor)
        return captures if captures else simple_moves