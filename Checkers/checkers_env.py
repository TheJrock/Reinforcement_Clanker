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
        self.stale_counter = 0

    def reset(self):
        self.board = [1] * 12 + [0] * 8 + [-1] * 12
        self.current_player = 1
        self.chained_capture_index = None
        self.stale_counter = 0
        return self._to_agent_state()

    def _to_agent_state(self):
        board_state = tuple((cell + 1) / 2 for cell in self.board)
        chain = 0 if self.chained_capture_index is None else (self.chained_capture_index + 1) / 32
        return board_state + (chain,)

    def step(self, action):
        start_index, end_index = divmod(action, 32)
        is_capture = abs(end_index - start_index) > 5
        is_promotion = self.board[start_index] == 1 and ((self.current_player == 1 and end_index >= 28) or (self.current_player == -1 and end_index <= 3))
        if is_capture or is_promotion:
             self.stale_counter = 0
        else:
            self.stale_counter += 1
            if self.stale_counter >= 40:
                return self._to_agent_state(), 0, True
        self.board[end_index] = 2 if is_promotion else self.board[start_index]
        self.board[start_index] = 0
        if not is_capture:
            self.chained_capture_index = None
        else:
            for direction in range(4):
                neighbor = self.NEIGHBORS[start_index][direction]
                if neighbor is not None and self.NEIGHBORS[neighbor][direction] == end_index:
                    captured_index = neighbor
                    break
            self.board[captured_index] = 0
            if self._can_capture(end_index):
                self.chained_capture_index = end_index
                return self._to_agent_state(), 0, False
            self.chained_capture_index = None
        self.current_player *= -1
        for i in range(32):
            self.board[i] *= -1
        if self.is_terminal():
            # flip back so current_player == winner, matching the convention evaluate() expects
            self.current_player *= -1
            for i in range(32):
                self.board[i] *= -1
            return self._to_agent_state(), 1, True

        return self._to_agent_state(), 0, False

    def get_legal_actions(self):
        forward_directions = (0, 1) if self.current_player == 1 else (2, 3)
        starting_indices = [i for i, cell in enumerate(self.board) if cell > 0]
        if self.chained_capture_index is not None:
            starting_indices = [self.chained_capture_index]
        simple_moves = []
        captures = []
        for start in starting_indices:
            valid_directions = [0, 1, 2, 3] if self.board[start] == 2 else forward_directions
            for direction in valid_directions:
                neighbor = self.NEIGHBORS[start][direction]
                if neighbor is not None and self.board[neighbor] == 0:
                    simple_moves.append(start * 32 + neighbor)
                elif neighbor is not None and self.board[neighbor] < 0:
                    jump_neighbor = self.NEIGHBORS[neighbor][direction]
                    if jump_neighbor is not None and self.board[jump_neighbor] == 0:
                        captures.append(start * 32 + jump_neighbor)
        return captures if captures else simple_moves
    
    def _can_capture(self, index):
        valid_directions = []
        if self.board[index] == 2 or self.current_player == 1:
            valid_directions += [0, 1]
        if self.board[index] == 2 or self.current_player == -1:
            valid_directions += [2, 3]
        for direction in valid_directions:
            neighbor = self.NEIGHBORS[index][direction]
            if neighbor is not None and self.board[neighbor] < 0:
                jump_neighbor = self.NEIGHBORS[neighbor][direction]
                if jump_neighbor is not None and self.board[jump_neighbor] == 0:
                    return True
        return False
    
    def is_terminal(self):
        return not self.get_legal_actions()
    
    def get_result(self):
        if not self.is_terminal():
            raise ValueError("get_result should only be called on terminal states")
        return 1
    
    def render(self):
        symbols = {0: '.', 1: 'b', -1: 'r', 2: 'B', -2: 'R'}
        row_order = range(7, -1, -1) if self.current_player == 1 else range(8)
        sep = "  +" + "-+" * 8
        print("    " + " ".join(str(c) for c in range(8)))
        print(sep)
        for board_row in row_order:
            row_str = f"{board_row} |"
            for col in range(8):
                if (board_row + col) % 2 == 0:
                    idx = board_row * 4 + (col // 2 if board_row % 2 == 0 else (col - 1) // 2)
                    row_str += symbols[self.board[idx]] + "|"
                else:
                    row_str += " |"
            print(row_str)
            print(sep)
        active = "P1 (b/B)" if self.current_player == 1 else "P2 (b/B)"
        if self.chained_capture_index is not None:
            active += f" — chained capture from square {self.chained_capture_index}"
        print(f"  Active: {active}\n")