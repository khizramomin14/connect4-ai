"""
Evaluation script for the trained Connect 4 DQN agent.

Runs the trained agent against:
  - RandomAgent
  - MinimaxAgent at several depths (2, 3, 4)

...as BOTH Player 1 and Player 2 (since Connect 4 is not symmetric —
going first is a real advantage, so testing both sides gives a fairer
picture of the agent's actual strength).

Produces a clean summary table you can drop directly into your report.
"""

import argparse

from env import Connect4Env
from agent import DQNAgent
from baselines import RandomAgent, MinimaxAgent


def play_match(agent, opponent, agent_player, num_games):
    """
    Play `num_games` with the trained agent as `agent_player` (1 or -1).
    Returns dict with wins/draws/losses counts.
    """
    results = {"wins": 0, "draws": 0, "losses": 0}
    env = Connect4Env()

    for _ in range(num_games):
        env.reset()
        while not env.done:
            if env.current_player == agent_player:
                action = agent.choose_action(env, training=False)
            else:
                action = opponent.choose_action(env)
            env.step(action)

        if env.winner == agent_player:
            results["wins"] += 1
        elif env.winner == 0:
            results["draws"] += 1
        else:
            results["losses"] += 1

    return results


def run_full_evaluation(model_path, num_games=50):
    agent = DQNAgent()
    agent.load(model_path)
    print(f"Loaded model from {model_path}\n")

    opponents = {
        "Random": RandomAgent(),
        "Minimax (depth=2)": MinimaxAgent(player=None, depth=2),
        "Minimax (depth=3)": MinimaxAgent(player=None, depth=3),
        "Minimax (depth=4)": MinimaxAgent(player=None, depth=4),
    }

    print(f"{'Opponent':<20}{'As P1 (W/D/L)':<20}{'As P2 (W/D/L)':<20}{'Overall Win%':<15}")
    print("-" * 75)

    for name, opponent in opponents.items():
        # Agent as Player 1 (goes first)
        if hasattr(opponent, "player"):
            opponent.player = -1
        res_p1 = play_match(agent, opponent, agent_player=1, num_games=num_games)

        # Agent as Player 2 (goes second)
        if hasattr(opponent, "player"):
            opponent.player = 1
        res_p2 = play_match(agent, opponent, agent_player=-1, num_games=num_games)

        total_wins = res_p1["wins"] + res_p2["wins"]
        total_games = num_games * 2
        overall_win_pct = 100 * total_wins / total_games

        p1_str = f"{res_p1['wins']}/{res_p1['draws']}/{res_p1['losses']}"
        p2_str = f"{res_p2['wins']}/{res_p2['draws']}/{res_p2['losses']}"

        print(f"{name:<20}{p1_str:<20}{p2_str:<20}{overall_win_pct:<15.1f}")

    print("\nFormat: Wins/Draws/Losses out of", num_games, "games per side")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Connect 4 DQN agent")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/dqn_final.pt",
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=50,
        help="Number of games per side (P1/P2) against each opponent",
    )
    args = parser.parse_args()

    run_full_evaluation(args.model, args.games)