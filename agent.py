"""
DQN Agent for Connect 4.

Wraps the DQN network (model.py) with everything needed to actually learn:
  - Replay buffer: stores past (state, action, reward, next_state, done)
    transitions and samples random minibatches from them. This breaks
    the correlation between consecutive game states, which stabilizes
    training (using only the most recent transitions would make the
    network overfit to whatever it just saw).
  - Epsilon-greedy action selection: balances exploration (trying random
    moves to discover good strategies) vs exploitation (using what it's
    already learned).
  - Two networks (policy + target): the target network is a periodically
    -synced snapshot of the policy network, used to compute stable
    training targets. Without this, the network would be "chasing a
    moving target" and training becomes unstable.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN, encode_state
from env import COLS


# ----------------------------------------------------------------------
# Replay Buffer
# ----------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ----------------------------------------------------------------------
# DQN Agent
# ----------------------------------------------------------------------
class DQNAgent:
    def __init__(
        self,
        lr=1e-4,
        gamma=0.99,
        buffer_capacity=50_000,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=20_000,
        target_update_freq=1000,  # sync target net every N training steps
        device=None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # target net is never trained directly

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss — more robust than MSE

        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.steps_done = 0

        self.target_update_freq = target_update_freq
        self.train_steps = 0

    # ------------------------------------------------------------------
    def current_epsilon(self):
        """Linearly decay epsilon from epsilon_start to epsilon_end."""
        fraction = min(1.0, self.steps_done / self.epsilon_decay_steps)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    # ------------------------------------------------------------------
    def choose_action(self, env, training=True, eval_epsilon=0.0):
        """
        Epsilon-greedy action selection.
        `env` is a Connect4Env instance — used to get valid moves and
        the current player for state encoding.

        `eval_epsilon`: when training=False, use this small epsilon
        instead of 0.0. Fully greedy (epsilon=0) evaluation is
        deterministic — the same opponent will always produce the exact
        same game, so N "games" collapse into 1 game repeated N times.
        A small eval_epsilon (e.g. 0.1) introduces enough variety across
        games to get a statistically meaningful win rate, while still
        mostly reflecting the learned policy.
        """
        valid_moves = env.valid_moves()
        epsilon = self.current_epsilon() if training else eval_epsilon

        if training:
            self.steps_done += 1

        if random.random() < epsilon:
            return random.choice(valid_moves)

        state = encode_state(env.board, env.current_player)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]

        # Mask invalid columns (set their Q-value to -infinity so they're
        # never selected as the max)
        masked_q = np.full(COLS, -np.inf, dtype=np.float32)
        for col in valid_moves:
            masked_q[col] = q_values[col]

        return int(np.argmax(masked_q))

    # ------------------------------------------------------------------
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    def train_step(self):
        """
        Sample a minibatch from the replay buffer and perform one
        gradient update on the policy network. Returns the loss value
        (or None if there isn't enough data yet).
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Q(s, a) for the actions actually taken
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                # Bellman target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)
            # Clamp targets to a sane range. Since true rewards only ever
            # take values in {-1, 0, 1} and gamma < 1, Q-values should
            # never legitimately need to exceed roughly [-1, 1]. If they
            # do, it's a sign of Q-value overestimation compounding
            # through self-play — clamping here stops that runaway
            # feedback loop from blowing up the loss.
            targets = torch.clamp(targets, min=-1.0, max=1.0)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — prevents occasional huge updates from
        # destabilizing training
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    # ------------------------------------------------------------------
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())


# ----------------------------------------------------------------------
# Quick sanity test when run directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from env import Connect4Env

    env = Connect4Env()
    agent = DQNAgent()

    print(f"Using device: {agent.device}")
    print(f"Starting epsilon: {agent.current_epsilon():.3f}")

    # Play a few random steps and store transitions to test the pipeline
    state = env.reset()
    for _ in range(10):
        pre_state = encode_state(env.board, env.current_player)
        action = agent.choose_action(env, training=True)
        _, reward, done = env.step(action)
        post_state = encode_state(env.board, env.current_player)

        agent.store_transition(pre_state, action, reward, post_state, done)

        if done:
            env.reset()

    print(f"Replay buffer size after 10 steps: {len(agent.replay_buffer)}")
    print(f"Epsilon after 10 steps: {agent.current_epsilon():.3f}")

    # Not enough data yet for a real batch (batch_size=64 by default),
    # so train_step should safely return None
    loss = agent.train_step()
    print(f"Loss (expected None, buffer too small): {loss}")