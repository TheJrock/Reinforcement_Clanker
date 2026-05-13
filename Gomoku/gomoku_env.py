from general_environment import GeneralEnvironment


class GomokuEnvironment(GeneralEnvironment):

    def __init__(self, board_size=15, win_length=5):
        self.board_size = board_size
        self.win_length = win_length
        self.board = [[0] * board_size for _ in range(board_size)]
        self.current_player = 1

    def reset(self):
        self.board = [[0] * self.board_size for _ in range(self.board_size)]
        self.current_player = 1
        return self._to_agent_state()

    def _to_agent_state(self):
        return tuple((v + 1) / 2 for row in self.board for v in row)

    def step(self, action):
        row, col = (action // self.board_size, action % self.board_size)
        if self.board[row][col] != 0:
            raise ValueError("Invalid action: cell is not empty")

        self.board[row][col] = 1

        if self._check_win(row, col):
            return self._to_agent_state(), 1, True

        if not self.get_legal_actions():
            return self._to_agent_state(), 0, True

        for r in range(self.board_size):
            for c in range(self.board_size):
                self.board[r][c] *= -1
        self.current_player *= -1

        return self._to_agent_state(), 0, False

    def _check_win(self, row, col):
        player = self.board[row][col]
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= self.win_length:
                return True
        return False

    def get_legal_actions(self):
        return [
            r * self.board_size + c
            for r in range(self.board_size)
            for c in range(self.board_size)
            if self.board[r][c] == 0
        ]

    def is_terminal(self):
        if not self.get_legal_actions():
            return True
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r][c] != 0 and self._check_win(r, c):
                    return True
        return False

    def get_result(self):
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r][c] == 1 and self._check_win(r, c):
                    return 1
        return 0

    def render(self):
        symbols = {0: ".", 1: "X", -1: "O"}
        header = "   " + "  ".join(f"{c:2}" for c in range(self.board_size))
        print(header)
        for r in range(self.board_size):
            row_str = "  ".join(f" {symbols[v * self.current_player]}" for v in self.board[r])
            print(f"{r:2} {row_str}")
            print()
        print()
