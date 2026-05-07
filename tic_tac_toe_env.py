from general_environment import GeneralEnvironment

class TicTacToeEnvironment(GeneralEnvironment):

    def __init__(self):
        self.board = [0] * 9
        self.current_player = 1

    def reset(self):
        self.board = [0] * 9
        self.current_player = 1
        return tuple(self.board)

    def step(self, action):
        if self.board[action] != 0:
            raise ValueError("Invalid action: Cell is already occupied")

        self.board[action] = 1
        done = self.is_terminal()
        reward = 0

        if done:
            result = self.get_result()
            if result == 1:
                reward = 1
        else:
            for i in range(len(self.board)):
                self.board[i] *= -1
            self.current_player *= -1

        return tuple(self.board), reward, done

    def get_legal_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

    def is_terminal(self):
        lines = self._get_board_lines()
        for line in lines:
            if line[0] != 0 and line[0] == line[1] == line[2]:
                return True

        return all(cell != 0 for cell in self.board)

    def render(self):
        symbols = {0: " ", self.current_player: "X", -self.current_player: "O"}
        rows = []
        for row in range(3):
            cells = [f" {symbols[self.board[row * 3 + col]]} " for col in range(3)]
            rows.append("|".join(cells))
        print(("\n---+---+---\n").join(rows))
        print()

    def get_result(self):
        if not self.is_terminal():
            raise ValueError("Game is not over yet")
        lines = self._get_board_lines()
        for line in lines:
            if line[0] != 0 and line[0] == line[1] == line[2]:
                return 1
        return 0

    def _get_board_lines(self):
        return [
            [self.board[0], self.board[1], self.board[2]],
            [self.board[3], self.board[4], self.board[5]],
            [self.board[6], self.board[7], self.board[8]],
            [self.board[0], self.board[3], self.board[6]],
            [self.board[1], self.board[4], self.board[7]],
            [self.board[2], self.board[5], self.board[8]],
            [self.board[0], self.board[4], self.board[8]],
            [self.board[2], self.board[4], self.board[6]],
        ]
