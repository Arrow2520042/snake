"""Headless training script for Snake DQN/CNN agent.

Supports N parallel environments (single-threaded, round-robin stepping)
for higher throughput.  CUDA used automatically when available.

Usage:
    python train.py --episodes 1000
    python train.py --agent cnn --simple-rewards --board-size 10 --num-envs 64
    python train.py --level levels/mymap.json --board-size 20
    python train.py --init-checkpoint model.pth --episodes 500
"""

import argparse
import datetime
import json
import os
import time

import numpy as np
import torch
from game import SnakeGameAI


def _load_walls(level_path, board_blocks):
    """Load wall set from a level JSON file."""
    with open(level_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    walls = set()
    for item in data:
        cx, cy = int(item[0]), int(item[1])
        if 0 <= cx < board_blocks and 0 <= cy < board_blocks:
            walls.add((cx, cy))
    return walls


def train(episodes=1000, max_steps=15000, save_every=200, level_path=None,
          init_checkpoint=None, log_name=None, save_path='model.pth',
          board_size=20, num_envs=1, fresh=False,
          agent_type='dqn', simple_rewards=False):

    state_mode = 'grid' if agent_type == 'cnn' else 'features'

    if agent_type == 'cnn':
        from cnn_agent import CNNAgent
        agent = CNNAgent(board_size=board_size)
    else:
        from dqn_agent import DQNAgent
        agent = DQNAgent()

    walls = None
    if level_path and os.path.isfile(level_path):
        walls = _load_walls(level_path, board_size)
        print(f'Loaded level: {level_path} ({len(walls)} walls)')

    # Create parallel environments
    envs = []
    for _ in range(num_envs):
        env = SnakeGameAI(render=False, board_blocks=board_size,
                          state_mode=state_mode, simple_rewards=simple_rewards)
        if walls:
            env.walls = walls
        envs.append(env)

    device_name = getattr(agent, 'device', 'cpu')
    print(f'Agent: {agent_type.upper()} | device: {device_name} | '
          f'simple_rewards: {simple_rewards} | n_step: {agent.n_step}')

    if init_checkpoint and os.path.isfile(init_checkpoint):
        agent.load(init_checkpoint, weights_only=fresh)
        if fresh:
            print(f'Loaded weights only: {init_checkpoint} (fresh training state: eps={agent.eps:.4f})')
        else:
            print(f'Loaded checkpoint: {init_checkpoint} (eps={agent.eps:.4f}, steps={agent.steps})')

    # Logging setup
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if log_name:
        log_dir = os.path.join('logs', log_name, ts)
    else:
        log_dir = os.path.join('logs', ts)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, 'rewards.npz')
    # Accumulate arrays in memory, flush periodically
    all_scores = []
    all_rewards = []
    all_steps = []

    try:
        info_path = os.path.join(log_dir, 'info.txt')
        with open(info_path, 'w', encoding='utf-8') as fi:
            fi.write(f'agent_type: {agent_type}\n')
            fi.write(f'simple_rewards: {simple_rewards}\n')
            fi.write(f'device: {getattr(agent, "device", "cpu")}\n')
            fi.write(f'level_path: {level_path}\n')
            fi.write(f'init_checkpoint: {init_checkpoint}\n')
            fi.write(f'episodes: {episodes}\n')
            fi.write(f'max_steps: {max_steps}\n')
            fi.write(f'board_size: {board_size}\n')
            fi.write(f'num_envs: {num_envs}\n')
            fi.write(f'n_step: {agent.n_step}\n')
            fi.write(f'eps_start: {agent.eps:.6f}\n')
            fi.write(f'lr_start: {agent.optimizer.param_groups[0]["lr"]:.1e}\n')
            fi.write(f'started: {ts}\n')
    except Exception:
        pass

    print(f'Training {episodes} episodes | board={board_size} | '
          f'envs={num_envs} | updates/round={max(1, num_envs // 16)} | logs -> {log_dir}')

    best_score = 0
    best_avg50 = 0.0
    best_ckpt_path = os.path.join(log_dir, 'best.pth')
    stagnation_counter = 0
    STAGNATION_THRESHOLD = 15000  # episodes without meaningful improvement before rollback
    STAGNATION_TOLERANCE = 0.9   # tolerate avg50 down to 90% of best before counting as stagnation
    ROLLBACK_EPS_THRESHOLD = 0.3  # don't rollback while still exploring (eps >= 0.3)
    updates_per_round = max(1, num_envs // 16)  # e.g. 128 envs → 8 updates/round
    t_start = time.time()
    recent_scores = []
    recent_rewards = []
    recent_losses = []
    recent_qvals = []

    # -- parallel episode loop ------------------------------------------
    # Each env runs its own episode concurrently; we round-robin step them.
    states = [env.reset() for env in envs]
    rewards_acc = [0.0] * num_envs
    steps_acc = [0] * num_envs
    ep_counter = 0  # total finished episodes

    while ep_counter < episodes:
        # Batched inference: one forward pass for all envs
        actions = agent.act_batch(states)

        # Step all envs with their actions
        for i, env in enumerate(envs):
            if ep_counter >= episodes:
                break
            action = int(actions[i])
            next_state, reward, done, info = env.play_step(action, skip_events=True)
            agent.push(i, states[i], action, reward, next_state, done,
                       snake_length=len(env.snake))
            rewards_acc[i] += reward
            steps_acc[i] += 1
            states[i] = next_state

            if done or steps_acc[i] >= max_steps:
                ep_counter += 1
                score = env.score
                ep_reward = rewards_acc[i]
                recent_scores.append(score)
                recent_rewards.append(ep_reward)
                avg50 = sum(recent_scores[-50:]) / len(recent_scores[-50:])
                avgR50 = sum(recent_rewards[-50:]) / len(recent_rewards[-50:])

                if score > best_score:
                    best_score = score

                # Track best avg50 and save best checkpoint
                if len(recent_scores) >= 50:
                    if avg50 > best_avg50:
                        best_avg50 = avg50
                        stagnation_counter = 0
                        agent.save(best_ckpt_path)
                    elif avg50 >= best_avg50 * STAGNATION_TOLERANCE:
                        # Within tolerance band — don't count as stagnation
                        pass
                    else:
                        stagnation_counter += 1

                    # Rollback to best checkpoint on prolonged stagnation
                    if (stagnation_counter >= STAGNATION_THRESHOLD
                            and agent.eps < ROLLBACK_EPS_THRESHOLD
                            and os.path.isfile(best_ckpt_path)):
                        old_eps = agent.eps
                        agent.load(best_ckpt_path)
                        agent.eps = old_eps  # keep current exploration rate
                        stagnation_counter = 0
                        print(f'  >> ROLLBACK to best (avg50={best_avg50:.1f}) at ep {ep_counter}')

                all_scores.append(score)
                all_rewards.append(ep_reward)
                all_steps.append(steps_acc[i])

                if ep_counter % 500 == 0:
                    np.savez_compressed(log_path,
                        scores=np.array(all_scores, dtype=np.int16),
                        rewards=np.array(all_rewards, dtype=np.float16),
                        steps=np.array(all_steps, dtype=np.uint16),
                        losses=np.array(recent_losses, dtype=np.float32),
                        qvals=np.array(recent_qvals, dtype=np.float32))

                if ep_counter % 1000 == 0 or ep_counter == 1:
                    elapsed = time.time() - t_start
                    lr = agent.optimizer.param_groups[0]['lr']
                    avgSteps = sum(all_steps[-50:]) / max(1, len(all_steps[-50:]))
                    avg_loss = sum(recent_losses[-200:]) / max(1, len(recent_losses[-200:])) if recent_losses else 0.0
                    avg_q = sum(recent_qvals[-200:]) / max(1, len(recent_qvals[-200:])) if recent_qvals else 0.0
                    print(f'ep {ep_counter}/{episodes} | '
                          f'best={best_score} avg50={avg50:.1f} '
                          f'avgR50={avgR50:.1f} avgSteps={avgSteps:.0f} | '
                          f'loss={avg_loss:.4f} Q={avg_q:.2f} | '
                          f'eps={agent.eps:.4f} lr={lr:.1e} '
                          f'elapsed={elapsed:.0f}s')
                    agent.step_scheduler(avg50)

                if ep_counter % save_every == 0:
                    agent.save(os.path.join(log_dir, f'model_ep{ep_counter}.pth'))

                # Reset this env for next episode
                states[i] = env.reset()
                rewards_acc[i] = 0.0
                steps_acc[i] = 0

        # Multiple gradient updates per round to compensate for many envs
        for _ in range(updates_per_round):
            loss_val, q_val = agent.update()
            if loss_val is not None:
                recent_losses.append(loss_val)
                recent_qvals.append(q_val)
        agent.decay_epsilon()  # once per round, not per gradient update

    # -- save final models ----------------------------------------------
    np.savez_compressed(log_path,
        scores=np.array(all_scores, dtype=np.int16),
        rewards=np.array(all_rewards, dtype=np.float16),
        steps=np.array(all_steps, dtype=np.uint16),
        losses=np.array(recent_losses, dtype=np.float32),
        qvals=np.array(recent_qvals, dtype=np.float32))
    agent.save(os.path.join(log_dir, save_path))
    agent.save(save_path)
    print(f'Done. Best score: {best_score}. Model saved to {save_path}')

    try:
        run_info = os.path.join(log_dir, 'run_info.txt')
        with open(run_info, 'w', encoding='utf-8') as ri:
            ri.write(f'best_score: {best_score}\n')
            ri.write(f'final_eps: {agent.eps:.6f}\n')
            ri.write(f'final_lr: {agent.optimizer.param_groups[0]["lr"]:.1e}\n')
            ri.write(f'total_time: {time.time() - t_start:.1f}s\n')
    except Exception:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Snake DQN agent (headless)')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--max-steps', type=int, default=15000)
    parser.add_argument('--save-every', type=int, default=10000)
    parser.add_argument('--level', type=str, default=None, help='Path to level JSON')
    parser.add_argument('--init-checkpoint', type=str, default=None, help='Path to .pth to resume')
    parser.add_argument('--log-name', type=str, default=None, help='Subdirectory name for logs')
    parser.add_argument('--save', type=str, default='model.pth', help='Final model filename')
    parser.add_argument('--board-size', type=int, default=20, help='Board size (curriculum learning)')
    parser.add_argument('--num-envs', type=int, default=128, help='Parallel environments (e.g. 64-128)')
    parser.add_argument('--agent', type=str, default='dqn', choices=['dqn', 'cnn'],
                        help='Agent type: dqn (feature vector) or cnn (grid observation)')
    parser.add_argument('--simple-rewards', action='store_true',
                        help='Use simplified reward: +10 food, -10 death, -0.01 step')
    parser.add_argument('--fresh', action='store_true',
                        help='Load weights only, reset training state (eps, optimizer, steps)')
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        max_steps=args.max_steps,
        save_every=args.save_every,
        level_path=args.level,
        init_checkpoint=args.init_checkpoint,
        log_name=args.log_name,
        save_path=args.save,
        board_size=args.board_size,
        num_envs=args.num_envs,
        fresh=args.fresh,
        agent_type=args.agent,
        simple_rewards=args.simple_rewards,
    )
