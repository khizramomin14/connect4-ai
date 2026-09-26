"""
Play against your trained Connect 4 DQN agent from the command line.

Usage:
    python play.py
    python play.py --model checkpoints/dqn_ep5000.pt
    python play.py --first    (you go first instead of the AI)
"""

import argparse

from env import Connect4Env
from agent import DQNAgent


def print_instructions():
    print("=" * 50)
    print("CONNECT 4 vs DQN AI")
    print("=" * 50)
    print("You are 'O'. The AI is 'X' (unless you chose --first).")
    print("Enter a column number (0-6) to drop your piece.")
    print("Type 'quit' at any time to exit.")
    print("=" * 50)


def get_human_move(env):
    while True:
        raw = input(f"Your move (columns {env.valid_moves()}): ").strip()
        if raw.lower() in ("quit", "exit", "q"):
            return None
        if not raw.isdigit():
            print("Please enter a number.")
            continue
        col = int(raw)
        if col not in env.valid_moves():
            print(f"Column {col} is invalid or full. Try again.")
            continue
        return col


def play_game(agent, human_player):
    env = Connect4Env()
    env.reset()
    ai_player = -human_player

    symbols = {human_player: "O (you)", ai_player: "X (AI)"}

    env.render()

    while not env.done:
        current = env.current_player

        if current == human_player:
            col = get_human_move(env)
            if col is None:
                print("Game exited.")
                return
        else:
            print("AI is thinking...")
            col = agent.choose_action(env, training=False)
            print(f"AI plays column {col}")

        env.step(col)
        env.render()

    if env.winner == 0:
        print("Result: It's a draw!")
    elif env.winner == human_player:
        print("You win! 🎉")
    else:
        print("AI wins! Better luck next time.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Connect 4 against the trained DQN agent")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/dqn_final.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--first",
        action="store_true",
        help="If set, you play first (as Player 1) instead of the AI",
    )
    args = parser.parse_args()

    agent = DQNAgent()
    agent.load(args.model)
    print(f"Loaded model: {args.model}")

    print_instructions()

    human_player = 1 if args.first else -1
    play_game(agent, human_player)