from general_environment import GeneralEnvironment

class TicTacToeEnvironment(GeneralEnvironment):

    WIN_LINES = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]

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
        done = self.is_local_terminal()
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

    def is_local_terminal(self):
        for line in self.WIN_LINES:
            if self.board[line[0]] != 0 and self.board[line[0]] == self.board[line[1]] == self.board[line[2]]:
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
        if not self.is_local_terminal():
            raise ValueError("Game is not over yet")
        for line in self.WIN_LINES:
            if self.board[line[0]] != 0 and self.board[line[0]] == self.board[line[1]] == self.board[line[2]]:
                return 1
        return 0
