from general_environment import GeneralEnvironment

class MancalaEnvironment(GeneralEnvironment):

    def __init__(self):
        self.board = [4] * 6 + [0] + [-4] * 6 + [0]
        self.current_player = 1

    def reset(self):
        self.board = [4] * 6 + [0] + [-4] * 6 + [0]
        self.current_player = 1
        return tuple(abs(v) for v in self.board)

    def step(self, action):
        pit = action if self.current_player == 1 else action + 7
        stones = abs(self.board[pit])
        if stones == 0:
            raise ValueError("Invalid action: pit is empty")

        self.board[pit] = 0
        opponent_goal = 13 if self.current_player == 1 else 6
        own_goal = 6 if self.current_player == 1 else 13

        while stones > 0:
            pit = (pit + 1) % 14
            if pit == opponent_goal:
                continue
            elif pit == own_goal:
                self.board[pit] += self.current_player
            else:
                self.board[pit] += self.current_player if pit < 7 else -self.current_player
            stones -= 1

        capture = pit != own_goal and self.board[pit] == 1
        if capture:
            mirror = 12 - pit
            self.board[own_goal] += self.current_player * (abs(self.board[mirror]) + 1)
            self.board[mirror] = 0
            self.board[pit] = 0

        done = self.is_terminal()
        reward = 0
        if done:
            reward = self.get_result()
        elif pit != own_goal:
            for pit in range(14):
                self.board[pit] *= -1
            self.current_player *= -1

        if self.current_player == -1:
            return tuple(abs(v) for v in self.board[7:] + self.board[:7]), reward, done
        return tuple(abs(v) for v in self.board), reward, done

    def get_legal_actions(self):
        actions = []
        for pit in range(len(self.board)):
            if self.board[pit] > 0 and pit not in [6, 13]:
                actions.append(pit % 7)

        return actions

    def is_terminal(self):
        return all(pit == 0 for pit in self.board[0:6]) or all(pit == 0 for pit in self.board[7:13])

    def get_result(self):
        playerOne_score = sum(abs(pit) for pit in self.board[:7])
        playerTwo_score = sum(abs(pit) for pit in self.board[7:])
        if playerOne_score > playerTwo_score:
            return 1 if self.current_player == 1 else -1
        elif playerOne_score < playerTwo_score:
            return -1 if self.current_player == 1 else 1

        return 0

    def render(self):
        b = [abs(v) for v in self.board]
        if self.current_player == 1:
            top_pits    = [b[i] for i in range(12, 6, -1)]
            bottom_pits = [b[i] for i in range(0, 6)]
            left_store, right_store = b[13], b[6]
            top_label, bottom_label = "P2", "P1"
        else:
            top_pits    = [b[i] for i in range(5, -1, -1)]
            bottom_pits = [b[i] for i in range(7, 13)]
            left_store, right_store = b[6], b[13]
            top_label, bottom_label = "P1", "P2"
        sep = "+------+----+----+----+----+----+----+------+"
        print(sep)
        print(f"|  {top_label}  | {' | '.join(f'{v:2}' for v in top_pits)} |  {bottom_label}  |")
        print(f"| {left_store:4} |{'    |' * 6}  {right_store:3} |")
        print(f"|      | {' | '.join(f'{v:2}' for v in bottom_pits)} |      |")
        print(sep)
        print()