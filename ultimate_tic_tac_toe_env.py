from general_environment import GeneralEnvironment

class UltimateTicTacToeEnvironment(GeneralEnvironment):

    WIN_LINES = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]

    def __init__(self):
        self.board = [[0] * 9 for _ in range(9)]
        self.meta_board = [0] * 9
        self.current_player = 1
        self.next_board = None

    def reset(self):
        self.board = [[0] * 9 for _ in range(9)]
        self.meta_board = [0] * 9
        self.current_player = 1
        self.next_board = None
        cells = tuple(cell for row in self.board for cell in row)
        meta = tuple(self.meta_board)
        active = (self.next_board if self.next_board is not None else -1,)
        return cells + meta + active

    def step(self, action):
        board_index = action // 9
        cell_index = action % 9
        done = False
        if self.next_board is not None and board_index != self.next_board:
            raise ValueError("Invalid action: Must play in the specified board")
        if self.board[board_index][cell_index] != 0:
            raise ValueError("Invalid action: Cell is already occupied")

        self.board[board_index][cell_index] = 1
        if self.is_local_terminal(self.board[board_index]):
            result = self.get_result(self.board[board_index])
            self.meta_board[board_index] = result
            if self.is_global_terminal():
                done = True
        reward = 0

        if done:
            result = self.get_result(self.meta_board, global_board=True)
            if result == 1:
                reward = 1
        else:
            if self.is_local_terminal(self.board[cell_index]):
                self.next_board = None
            else:
                self.next_board = cell_index
            for i in range(9):
                self.meta_board[i] *= -1
                for j in range(9):
                    self.board[i][j] *= -1
            self.current_player *= -1

        cells = tuple(cell for row in self.board for cell in row)
        meta = tuple(self.meta_board)
        active = (self.next_board if self.next_board is not None else -1,)
        return cells + meta + active

    def get_legal_actions(self):
        if self.next_board is not None:
            return [self.next_board * 9 + i for i in range(9) if self.board[self.next_board][i] == 0]
        else:
            actions = []
            for board_index in range(9):
                if not self.is_local_terminal(self.board[board_index]):
                    for cell_index in range(9):
                        if self.board[board_index][cell_index] == 0:
                            actions.append(board_index * 9 + cell_index)
            return actions

    def is_local_terminal(self, board):
        for line in self.WIN_LINES:
            if board[line[0]] != 0 and board[line[0]] == board[line[1]] == board[line[2]]:
                return True                
        return all(cell != 0 for cell in board)
    
    def is_global_terminal(self):
        for line in self.WIN_LINES:
            if self.meta_board[line[0]] != 0 and self.meta_board[line[0]] == self.meta_board[line[1]] == self.meta_board[line[2]]:
                return True
        for board in self.board:
            if not self.is_local_terminal(board):
                return False
        return True

    def get_result(self, board, global_board=False):
        if not global_board and not self.is_local_terminal(board):
            raise ValueError("Local game is not over yet")
        elif global_board and not self.is_global_terminal():
            raise ValueError("Global game is not over yet")
        for line in self.WIN_LINES:
            if board[line[0]] != 0 and board[line[0]] == board[line[1]] == board[line[2]]:
                return 1
        return 0
