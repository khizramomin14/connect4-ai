"""
Training loop for the Connect 4 DQN agent.

The agent plays against ITSELF (self-play): the same policy network
controls both players. This is standard for two-player zero-sum games —
the agent effectively bootstraps its own opponent, and as it improves,
so does the "opponent" it's training against, creating a natural
curriculum.

Every EVAL_EVERY episodes, we pause training and measure the agent's
win rate against the fixed baseline agents (random, minimax) — this is
what actually tells us if it's learning anything useful, since the
training loss alone doesn't indicate playing strength.
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from env import Connect4Env
from agent import DQNAgent
from model import encode_state
from baselines import RandomAgent, MinimaxAgent

# ----------------------------------------------------------------------
# Hyperparameters / config
# ----------------------------------------------------------------------
NUM_EPISODES = 15_000
EVAL_EVERY = 250          # run an evaluation match every N episodes
EVAL_GAMES = 30           # games per evaluation opponent
CHECKPOINT_EVERY = 1000   # save model weights every N episodes
CHECKPOINT_DIR = "checkpoints"
LOG_EVERY = 100           # print progress every N episodes


def play_self_play_episode(env, agent):
    """
    Play one full game with the agent controlling BOTH players.
    Store transitions for both players' perspectives into the replay
    buffer, then run one training step.

    Returns: (episode_loss_avg, winner)
    """
    env.reset()
    losses = []

    # We need to remember each player's last (state, action) so we can
    # push the transition once we see the resulting next_state/reward.
    pending = {1: None, -1: None}

    while not env.done:
        player = env.current_player
        pre_state = encode_state(env.board, player)

        action = agent.choose_action(env, training=False, eval_epsilon=0.1)
        _, reward, done = env.step(action)

        # If there's a pending transition for THIS player from their
        # previous turn, finalize it now that we know what happened
        # after their move (note: the opponent's move happens between
        # this player's two turns, so "next_state" for a pending
        # transition is the state right before this new move).
        if pending[player] is not None:
            prev_state, prev_action = pending[player]
            # From the perspective of `player`, the reward for their
            # earlier move is 0 (game continued) unless this current
            # move ended the game as a loss caused by opponent — but
            # since Connect 4 rewards are only assigned to the mover
            # who completes 4-in-a-row, we handle the "loss" case
            # separately below.
            agent.store_transition(prev_state, prev_action, 0.0, pre_state, False)

        pending[player] = (pre_state, action)

        if done:
            # The player who just moved gets `reward` (1 for win, 0 for draw).
            post_state = encode_state(env.board, player)
            agent.store_transition(pre_state, action, reward, post_state, True)

            # The OTHER player's pending transition (from their last move)
            # must be updated too: if this move caused a win, the other
            # player effectively "lost" as a result of the position they
            # left behind — assign them the negative reward.
            other = -player
            if pending[other] is not None and reward == 1:
                other_prev_state, other_prev_action = pending[other]
                agent.store_transition(
                    other_prev_state, other_prev_action, -1.0, pre_state, True
                )

        loss = agent.train_step()
        if loss is not None:
            losses.append(loss)

    avg_loss = float(np.mean(losses)) if losses else None
    return avg_loss, env.winner


def evaluate_vs_opponent(agent, opponent, agent_player, num_games):
    """
    Play `num_games` between the trained agent (greedy, no exploration)
    and a fixed baseline opponent. `agent_player` is which side (1 or -1)
    the trained agent plays as.
    Returns win rate (fraction) for the trained agent.
    """
    wins = 0
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
            wins += 1

    return wins / num_games


def run_training():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    env = Connect4Env()
    agent = DQNAgent()

    random_opponent = RandomAgent()
    minimax_opponent = MinimaxAgent(player=None, depth=3)  # player set per-eval below

    history = {
        "episode": [],
        "loss": [],
        "win_rate_random": [],
        "win_rate_minimax": [],
    }

    start_time = time.time()

    for episode in range(1, NUM_EPISODES + 1):
        avg_loss, winner = play_self_play_episode(env, agent)

        if episode % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            eps = agent.current_epsilon()
            loss_str = f"{avg_loss:.4f}" if avg_loss is not None else "N/A"
            print(
                f"Episode {episode:>6} | epsilon={eps:.3f} | "
                f"loss={loss_str} | buffer={len(agent.replay_buffer):>6} | "
                f"elapsed={elapsed:.0f}s"
            )

        if episode % EVAL_EVERY == 0:
            # Evaluate as Player 1 against random and minimax
            minimax_opponent.player = -1
            wr_random = evaluate_vs_opponent(agent, random_opponent, 1, EVAL_GAMES)
            wr_minimax = evaluate_vs_opponent(agent, minimax_opponent, 1, EVAL_GAMES)

            history["episode"].append(episode)
            history["loss"].append(avg_loss if avg_loss is not None else 0.0)
            history["win_rate_random"].append(wr_random)
            history["win_rate_minimax"].append(wr_minimax)

            print(
                f"  >> Eval @ ep {episode}: "
                f"win% vs random = {wr_random*100:.1f}% | "
                f"win% vs minimax(d3) = {wr_minimax*100:.1f}%"
            )

        if episode % CHECKPOINT_EVERY == 0:
            path = os.path.join(CHECKPOINT_DIR, f"dqn_ep{episode}.pt")
            agent.save(path)
            agent.save(os.path.join(CHECKPOINT_DIR, "dqn_latest.pt"))

    # Final save
    agent.save(os.path.join(CHECKPOINT_DIR, "dqn_final.pt"))
    print("Training complete. Final model saved to checkpoints/dqn_final.pt")

    plot_training_curves(history)


def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["episode"], history["win_rate_random"], label="vs Random")
    axes[0].plot(history["episode"], history["win_rate_minimax"], label="vs Minimax (d3)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Win rate")
    axes[0].set_title("Win rate over training")
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    axes[1].plot(history["episode"], history["loss"])
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Avg training loss")
    axes[1].set_title("Training loss over time")

    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("Saved training curves to training_curves.png")


if __name__ == "__main__":
    run_training()