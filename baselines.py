"""
Baseline opponents for Connect 4.

These are NOT learning agents — they're fixed-strategy players used to:
  1. Sanity-check that your environment works
  2. Provide an evaluation benchmark for your trained DQN agent later
     ("DQN wins X% of games vs random", "DQN wins X% vs minimax depth-4")

Both agents expose the same interface:
    agent.choose_action(env) -> column index (int)
so you can swap them in and out interchangeably during evaluation.
"""

import random
import math
from env import Connect4Env, ROWS, COLS


# ----------------------------------------------------------------------
# Random Agent
# ----------------------------------------------------------------------
class RandomAgent:
    """Picks any valid column uniformly at random."""

    def choose_action(self, env: Connect4Env):
        return random.choice(env.valid_moves())


# ----------------------------------------------------------------------
# Minimax Agent (with alpha-beta pruning)
# ----------------------------------------------------------------------
class MinimaxAgent:
    """
    Depth-limited minimax with alpha-beta pruning.

    `player` is which side this agent plays as (1 or -1) — needed so the
    heuristic evaluation knows which player's position to score highly.
    """

    def __init__(self, player, depth=4):
        self.player = player
        self.depth = depth

    def choose_action(self, env: Connect4Env):
        _, best_col = self._minimax(
            env.board.copy(),
            depth=self.depth,
            alpha=-math.inf,
            beta=math.inf,
            maximizing=True,
            current_player=self.player,
        )
        # Fallback safety net: if minimax somehow returns None
        # (shouldn't happen if there's at least one valid move)
        if best_col is None:
            best_col = random.choice(env.valid_moves())
        return best_col

    # ------------------------------------------------------------------
    def _minimax(self, board, depth, alpha, beta, maximizing, current_player):
        valid_moves = self._valid_moves(board)
        terminal, winner = self._check_terminal(board)

        if terminal:
            if winner == self.player:
                return (1_000_000 + depth), None  # win sooner = better
            elif winner == -self.player:
                return (-1_000_000 - depth), None  # loss sooner = worse
            else:
                return 0, None  # draw

        if depth == 0:
            return self._heuristic(board), None

        if maximizing:
            value = -math.inf
            best_col = random.choice(valid_moves)
            for col in valid_moves:
                child = self._simulate_move(board, col, current_player)
                score, _ = self._minimax(
                    child, depth - 1, alpha, beta, False, -current_player
                )
                if score > value:
                    value = score
                    best_col = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value, best_col
        else:
            value = math.inf
            best_col = random.choice(valid_moves)
            for col in valid_moves:
                child = self._simulate_move(board, col, current_player)
                score, _ = self._minimax(
                    child, depth - 1, alpha, beta, True, -current_player
                )
                if score < value:
                    value = score
                    best_col = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value, best_col

    # ------------------------------------------------------------------
    # Helpers (operate on raw board arrays, not the Env, so we don't
    # mutate the real game state while searching)
    # ------------------------------------------------------------------
    def _valid_moves(self, board):
        return [c for c in range(COLS) if board[0, c] == 0]

    def _simulate_move(self, board, col, player):
        new_board = board.copy()
        for r in range(ROWS - 1, -1, -1):
            if new_board[r, col] == 0:
                new_board[r, col] = player
                break
        return new_board

    def _check_terminal(self, board):
        """Returns (is_terminal, winner) where winner is 1, -1, 0 (draw), or None."""
        for player in (1, -1):
            if self._has_won(board, player):
                return True, player
        if len(self._valid_moves(board)) == 0:
            return True, 0
        return False, None

    def _has_won(self, board, player):
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                if all(board[r, c + i] == player for i in range(4)):
                    return True
        # Vertical
        for c in range(COLS):
            for r in range(ROWS - 3):
                if all(board[r + i, c] == player for i in range(4)):
                    return True
        # Diagonal down-right
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if all(board[r + i, c + i] == player for i in range(4)):
                    return True
        # Diagonal up-right
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                if all(board[r - i, c + i] == player for i in range(4)):
                    return True
        return False

    def _heuristic(self, board):
        """
        Simple positional heuristic used when search hits depth 0.
        Scores based on how many 2/3-in-a-row windows each player has,
        plus a bonus for controlling the center column (statistically
        the strongest column in Connect 4).
        """
        score = 0

        # Center column control
        center_col = board[:, COLS // 2]
        score += 3 * int(sum(center_col == self.player))
        score -= 3 * int(sum(center_col == -self.player))

        # Score all windows of 4 (horizontal, vertical, diagonal)
        score += self._score_all_windows(board, self.player)
        score -= self._score_all_windows(board, -self.player)

        return score

    def _score_all_windows(self, board, player):
        total = 0
        opponent = -player

        def score_window(window):
            count_p = sum(1 for v in window if v == player)
            count_o = sum(1 for v in window if v == opponent)
            count_e = sum(1 for v in window if v == 0)
            if count_p == 4:
                return 100
            elif count_p == 3 and count_e == 1:
                return 5
            elif count_p == 2 and count_e == 2:
                return 2
            elif count_o == 3 and count_e == 1:
                return -4  # block opponent threats
            return 0

        # Horizontal
        for r in range(ROWS):
            row = board[r, :]
            for c in range(COLS - 3):
                total += score_window(row[c:c + 4])
        # Vertical
        for c in range(COLS):
            col = board[:, c]
            for r in range(ROWS - 3):
                total += score_window(col[r:r + 4])
        # Diagonal down-right
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                window = [board[r + i, c + i] for i in range(4)]
                total += score_window(window)
        # Diagonal up-right
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                window = [board[r - i, c + i] for i in range(4)]
                total += score_window(window)

        return total


# ----------------------------------------------------------------------
# Quick manual test: Random vs Minimax
# ----------------------------------------------------------------------
if __name__ == "__main__":
    env = Connect4Env()
    random_agent = RandomAgent()
    minimax_agent = MinimaxAgent(player=-1, depth=4)  # minimax plays as O (-1)

    env.reset()
    env.render()

    while not env.done:
        if env.current_player == 1:
            action = random_agent.choose_action(env)
            print(f"Random (X) plays column {action}")
        else:
            action = minimax_agent.choose_action(env)
            print(f"Minimax (O) plays column {action}")

        state, reward, done = env.step(action)
        env.render()

    if env.winner == 0:
        print("Result: Draw")
    else:
        print(f"Result: Player {env.winner} wins")