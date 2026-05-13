from general_environment import GeneralEnvironment


class ConnectFourEnvironment(GeneralEnvironment):

    def __init__(self):
        self.board = [[0] * 7 for _ in range(6)]
        self.current_player = 1

    def reset(self):
        self.board = [[0] * 7 for _ in range(6)]
        self.current_player = 1
        return self._to_agent_state()

    def _to_agent_state(self):
        return tuple((v + 1) / 2 for row in self.board for v in row)

    def step(self, action):
        row = None
        for r in reversed(range(6)):
            if self.board[r][action] == 0:
                row = r
                break
        if row is None:
            raise ValueError("Invalid action: column is full")

        self.board[row][action] = 1

        if self._check_win(row, action):
            return self._to_agent_state(), 1, True

        if not self.get_legal_actions():
            return self._to_agent_state(), 0, True

        for r in range(6):
            for c in range(7):
                self.board[r][c] *= -1
        self.current_player *= -1

        return self._to_agent_state(), 0, False

    def _check_win(self, row, col):
        player = self.board[row][col]
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < 6 and 0 <= c < 7 and self.board[r][c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= 4:
                return True
        return False

    def get_legal_actions(self):
        return [c for c in range(7) if self.board[0][c] == 0]

    def is_terminal(self):
        if not self.get_legal_actions():
            return True
        for r in range(6):
            for c in range(7):
                if self.board[r][c] != 0 and self._check_win(r, c):
                    return True
        return False

    def get_result(self):
        for r in range(6):
            for c in range(7):
                if self.board[r][c] == 1 and self._check_win(r, c):
                    return 1
        return 0

    def render(self):
        symbols = {0: ".", 1: "X", -1: "O"}
        for row in self.board:
            print(" ".join(symbols[v * self.current_player] for v in row))
        print(" ".join(str(c) for c in range(7)))
        print()
