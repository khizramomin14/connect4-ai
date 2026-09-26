"""
Connect 4 game engine.

Board representation:
    6 rows x 7 columns, stored as a NumPy int array.
    0  = empty cell
    1  = Player 1's piece
   -1  = Player 2's piece

Row 0 is the TOP of the board, row 5 is the BOTTOM (where pieces settle
first when dropped).
"""

import numpy as np

ROWS = 6
COLS = 7


class Connect4Env:
    def __init__(self):
        self.board = None
        self.current_player = None  # 1 or -1
        self.done = False
        self.winner = None  # 1, -1, or 0 for draw
        self.reset()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self):
        """Start a new game. Returns the initial state."""
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.get_state()

    def valid_moves(self):
        """Return list of column indices that are not full."""
        return [c for c in range(COLS) if self.board[0, c] == 0]

    def step(self, action):
        """
        Drop the current player's piece into column `action`.

        Returns: (next_state, reward, done)
            reward is from the perspective of the player who just moved:
                +1  win
                -1  illegal move (should not happen if you mask actions)
                 0  draw or game still ongoing
        """
        if self.done:
            raise RuntimeError("Game is already over. Call reset().")

        if action not in self.valid_moves():
            # Illegal move — heavily penalize and end the episode.
            self.done = True
            self.winner = -self.current_player
            return self.get_state(), -1, True

        row = self._get_drop_row(action)
        self.board[row, action] = self.current_player

        if self._check_winner(row, action):
            self.done = True
            self.winner = self.current_player
            reward = 1
        elif len(self.valid_moves()) == 0:
            self.done = True
            self.winner = 0  # draw
            reward = 0
        else:
            reward = 0

        if not self.done:
            self.current_player *= -1  # switch turns

        return self.get_state(), reward, self.done

    def get_state(self):
        """Return a copy of the board (safe to store in replay buffer)."""
        return self.board.copy()

    def render(self):
        symbols = {0: ".", 1: "X", -1: "O"}
        print()
        for r in range(ROWS):
            print(" ".join(symbols[v] for v in self.board[r]))
        print(" ".join(str(c) for c in range(COLS)))
        print()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _get_drop_row(self, col):
        """Find the lowest empty row in this column."""
        for r in range(ROWS - 1, -1, -1):
            if self.board[r, col] == 0:
                return r
        raise ValueError(f"Column {col} is full")

    def _check_winner(self, row, col):
        """Check if the piece just placed at (row, col) creates 4-in-a-row."""
        player = self.board[row, col]
        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal down-right
            (1, -1),  # diagonal down-left
        ]
        for dr, dc in directions:
            count = 1
            count += self._count_direction(row, col, dr, dc, player)
            count += self._count_direction(row, col, -dr, -dc, player)
            if count >= 4:
                return True
        return False

    def _count_direction(self, row, col, dr, dc, player):
        count = 0
        r, c = row + dr, col + dc
        while 0 <= r < ROWS and 0 <= c < COLS and self.board[r, c] == player:
            count += 1
            r += dr
            c += dc
        return count


# ----------------------------------------------------------------------
# Quick manual test when run directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    env = Connect4Env()
    env.render()

    # Simulate a quick vertical win for player 1 in column 3
    test_moves = [3, 0, 3, 0, 3, 0, 3]  # P1 drops 4x in col 3, P2 drops in col 0
    for move in test_moves:
        if env.done:
            break
        state, reward, done = env.step(move)
        env.render()
        print(f"Move: col {move} | reward: {reward} | done: {done}")

    print(f"Winner: {env.winner}")