import random
from general_environment import GeneralEnvironment


class BackgammonEnvironment(GeneralEnvironment):
    """
    Board: 24 points indexed 0-23. Current player moves from index 23 → 0.
    Positive values = current player's pieces, negative = opponent's pieces.
    Home board is indices 0-5. Re-entry from bar targets indices 18-23.

    Action encoding: from_idx * 26 + to_idx
      from_idx: 0-23 = board point, 24 = bar
      to_idx:   0-23 = board point, 25 = bear off

    State: 56 floats — two 24-value channels (my/opp piece counts normalized /15),
           bar and borne-off counts for each player (/15), remaining dice (/6, zero-padded to 4).
    """

    def __init__(self):
        self.board = [0] * 24
        self.bar = [0, 0]    # [current_player, opponent]
        self.off = [0, 0]    # [current_player, opponent]
        self.current_player = 1
        self.dice = []

    def _initial_board(self):
        board = [0] * 24
        # Current player's pieces (positive)
        board[23] = 2   # point 24
        board[12] = 5   # point 13
        board[7] = 3    # point 8
        board[5] = 5    # point 6
        # Opponent's pieces (negative), mirrored
        board[0] = -2   # opponent's point 24 → our point 1
        board[11] = -5  # opponent's point 13 → our point 12
        board[16] = -3  # opponent's point 8  → our point 17
        board[18] = -5  # opponent's point 6  → our point 19
        return board

    def reset(self):
        self.board = self._initial_board()
        self.bar = [0, 0]
        self.off = [0, 0]
        self.current_player = 1
        self.dice = self._roll_dice()
        self._advance_if_no_moves()
        return self._to_agent_state()

    def _roll_dice(self):
        d1, d2 = random.randint(1, 6), random.randint(1, 6)
        return [d1] * 4 if d1 == d2 else sorted([d1, d2], reverse=True)

    def _to_agent_state(self):
        my_pieces = tuple(max(0, v) / 15 for v in self.board)
        opp_pieces = tuple(max(0, -v) / 15 for v in self.board)
        extras = (self.bar[0] / 15, self.bar[1] / 15, self.off[0] / 15, self.off[1] / 15)
        dice_enc = tuple(d / 6 for d in self.dice) + (0.0,) * (4 - len(self.dice))
        return my_pieces + opp_pieces + extras + dice_enc

    def step(self, action):
        from_idx, to_idx = divmod(action, 26)
        die = self._infer_die(from_idx, to_idx)
        self.dice.remove(die)

        if from_idx == 24:
            self.bar[0] -= 1
        else:
            self.board[from_idx] -= 1

        if to_idx == 25:
            self.off[0] += 1
            if self.off[0] == 15:
                return self._to_agent_state(), 1, True
        else:
            if self.board[to_idx] == -1:
                self.board[to_idx] = 0
                self.bar[1] += 1
            self.board[to_idx] += 1

        if not self.dice or not self.get_legal_actions():
            self._flip_perspective()
            self.dice = self._roll_dice()
            self._advance_if_no_moves()
            if not self.get_legal_actions() and not self.is_terminal():
                return self._to_agent_state(), 0, True

        return self._to_agent_state(), 0, False

    def _infer_die(self, from_idx, to_idx):
        if from_idx == 24:
            return 24 - to_idx
        if to_idx == 25:
            exact = from_idx + 1
            if exact in self.dice:
                return exact
            return min(d for d in self.dice if d > from_idx)
        return from_idx - to_idx

    def _flip_perspective(self):
        self.board = [-x for x in reversed(self.board)]
        self.bar[0], self.bar[1] = self.bar[1], self.bar[0]
        self.off[0], self.off[1] = self.off[1], self.off[0]
        self.current_player *= -1

    def _advance_if_no_moves(self):
        attempts = 0
        while not self.get_legal_actions() and not self.is_terminal() and attempts < 100:
            self._flip_perspective()
            self.dice = self._roll_dice()
            attempts += 1

    def get_legal_actions(self):
        actions = set()

        if self.bar[0] > 0:
            for die in set(self.dice):
                to_idx = 24 - die  # die=1 → idx 23, die=6 → idx 18
                if self.board[to_idx] >= -1:
                    actions.add(24 * 26 + to_idx)
            return list(actions)

        can_bear_off = self._can_bear_off()

        for die in set(self.dice):
            for from_idx in range(24):
                if self.board[from_idx] <= 0:
                    continue
                to_idx = from_idx - die
                if to_idx >= 0:
                    if self.board[to_idx] >= -1:
                        actions.add(from_idx * 26 + to_idx)
                elif can_bear_off:
                    if to_idx == -1:
                        actions.add(from_idx * 26 + 25)
                    else:
                        # Overshoot: only valid if no pieces at higher home board indices
                        if not any(self.board[j] > 0 for j in range(from_idx + 1, 6)):
                            actions.add(from_idx * 26 + 25)

        return list(actions)

    def _can_bear_off(self):
        if self.bar[0] > 0:
            return False
        return all(self.board[i] <= 0 for i in range(6, 24))

    def is_terminal(self):
        return self.off[0] == 15 or self.off[1] == 15

    def get_result(self):
        if not self.is_terminal():
            raise ValueError("get_result should only be called on terminal states")
        return 1

    def render(self):
        def fmt(val):
            if val == 0:
                return '  .  '
            return f'{val:+4d} '

        active = 'P1' if self.current_player == 1 else 'P2'
        print(f"\n  {active}'s turn | Dice: {self.dice} | Bar: mine={self.bar[0]} opp={self.bar[1]} | Off: mine={self.off[0]} opp={self.off[1]}")
        print(f"  Direction of movement: Pt 24 → Pt 1 → Off\n")

        # Top row: indices 23 down to 12 (Pt 24 down to Pt 13), split by bar at midpoint
        top_left  = list(range(23, 17, -1))   # indices 23-18 → Pt 24-19
        top_right = list(range(17, 11, -1))   # indices 17-12 → Pt 18-13

        # Bottom row: indices 0 up to 11 (Pt 1 up to Pt 12), split by bar at midpoint
        bot_left  = list(range(0, 6))          # indices 0-5  → Pt 1-6
        bot_right = list(range(6, 12))         # indices 6-11 → Pt 7-12

        sep = "  " + "-" * 30 + "+" + "-" * 30

        def row_str(indices):
            return "  " + "".join(fmt(self.board[i]) for i in indices)

        def label_str(indices):
            return "  " + "".join(f"{i+1:>4} " for i in indices)

        print(label_str(top_left) + "  |" + label_str(top_right))
        print(row_str(top_left)   + "  |" + row_str(top_right))
        print(sep)
        print(row_str(bot_left)   + "  |" + row_str(bot_right))
        print(label_str(bot_left) + "  |" + label_str(bot_right))
        print()
