"""
Neural network for the DQN agent.

Input:  the board state, encoded as a 2-channel 6x7 tensor
            channel 0 = 1 where the CURRENT player has a piece, else 0
            channel 1 = 1 where the OPPONENT has a piece, else 0
        (encoding it relative to "current player" rather than raw
         1 / -1 values makes the network's job easier — it always
         reasons from "my pieces vs their pieces" regardless of
         which physical player it's controlling)

Output: 7 Q-values, one per column — Q(state, action) for each
        possible move.
"""

import torch
import torch.nn as nn
import numpy as np

from env import ROWS, COLS


def encode_state(board: np.ndarray, current_player: int) -> np.ndarray:
    """
    Convert a raw board (values in {-1, 0, 1}) into the 2-channel
    tensor the network expects, from `current_player`'s perspective.
    """
    my_pieces = (board == current_player).astype(np.float32)
    opp_pieces = (board == -current_player).astype(np.float32)
    return np.stack([my_pieces, opp_pieces], axis=0)  # shape: (2, ROWS, COLS)


class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        flat_size = 64 * ROWS * COLS

        self.fc = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, COLS),  # one Q-value per column
        )

    def forward(self, x):
        # x shape: (batch, 2, ROWS, COLS)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)  # shape: (batch, COLS)


# ----------------------------------------------------------------------
# Quick sanity test when run directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from env import Connect4Env

    env = Connect4Env()
    env.step(3)  # make one move so the board isn't empty

    state_tensor = encode_state(env.board, env.current_player)
    print("Encoded state shape:", state_tensor.shape)  # expect (2, 6, 7)

    model = DQN()
    # Add batch dimension: (1, 2, 6, 7)
    input_tensor = torch.tensor(state_tensor).unsqueeze(0)
    q_values = model(input_tensor)

    print("Q-values shape:", q_values.shape)  # expect (1, 7)
    print("Q-values:", q_values.detach().numpy())

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params:,}")