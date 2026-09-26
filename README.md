# Connect 4 AI — Self-Play Deep Q-Learning

An AI/ML semester project implementing a Connect 4 game engine with an autonomous
AI player trained using Deep Q-Learning (DQN) via self-play.

## Overview

This project builds a complete pipeline: a verified Connect 4 game engine, baseline
opponents (random + minimax), a convolutional Q-network, and a self-play training
loop — resulting in an AI agent that can play Connect 4 against a human or other
agents.

## Project Structure
connect4-ai/
├── env.py # Connect 4 game engine (board, moves, win detection)
├── model.py # DQN neural network + board-state encoding
├── agent.py # DQN agent (replay buffer, epsilon-greedy, training step)
├── baselines.py # Random agent + Minimax agent (alpha-beta pruning)
├── train.py # Self-play training loop + evaluation + checkpointing
├── evaluate.py # Post-training evaluation vs all baselines
├── play.py # CLI to play against the trained agent
└── checkpoints/ # Saved model weights (.pt files)

## Requirements

```bash
pip install torch numpy matplotlib
```

## How It Works

- **State representation**: the board is encoded as a 2-channel tensor
  (your pieces vs. opponent pieces) so the same network can play as either side.
- **Network**: 3 convolutional layers + 2 fully-connected layers, outputting
  a Q-value for each of the 7 columns.
- **Training**: the agent plays against itself (self-play), storing experience
  in a replay buffer and learning via the Bellman equation, with a target
  network for stability.
- **Reward**: +1 for winning, -1 for an illegal move, 0 for an ongoing game
  or draw.

## Usage

**Train the agent from scratch:**
```bash
python train.py
```
This runs self-play training, periodically evaluates against baseline
opponents, saves checkpoints to `checkpoints/`, and plots training curves
to `training_curves.png`.

**Evaluate a trained model:**
```bash
python evaluate.py --model checkpoints/dqn_latest.pt --games 50
```
Reports win/draw/loss rates against Random and Minimax (depth 2–4) as both
Player 1 and Player 2.

**Play against the trained AI:**
```bash
python play.py --model checkpoints/dqn_latest.pt
```
Or to go first yourself:
```bash
python play.py --model checkpoints/dqn_latest.pt --first
```

## Results

| Opponent            | As P1 (W/D/L) | As P2 (W/D/L) | Overall Win % |
|---------------------|---------------|---------------|---------------|
| Random              | 38/0/12       | 37/0/13       | 75.0%         |
| Minimax (depth=2)   | 0/0/50        | 0/0/50        | 0.0%          |
| Minimax (depth=3)   | 0/0/50        | 0/0/50        | 0.0%          |

The trained agent reliably beats random play but loses consistently to
deliberate search-based opponents (Minimax) — a known limitation of pure
self-play, discussed in detail in the project report.

## Known Limitations & Future Work

- Pure self-play never exposes the agent to deliberate multi-move planning,
  so it has a blind spot against search-based opponents.
- Future improvement: mixed-opponent/curriculum training (interleaving
  self-play with games against Minimax), or an AlphaZero-style MCTS +
  policy/value network approach.

