"""Headless training script for Snake DQN agent.

Supports N parallel environments (single-threaded, round-robin stepping)
for higher throughput on multi-core CPUs.

Usage:
    python train.py --episodes 1000
    python train.py --level levels/mymap.json --board-size 20
    python train.py --init-checkpoint model.pth --episodes 500
    python train.py --num-envs 8 --episodes 2000
"""

import argparse
import csv
import datetime
import json
import os
import time

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
          board_size=20, num_envs=1, fresh=False):
    from dqn_agent import DQNAgent

    walls = None
    if level_path and os.path.isfile(level_path):
        walls = _load_walls(level_path, board_size)
        print(f'Loaded level: {level_path} ({len(walls)} walls)')

    # Create parallel environments
    envs = []
    for _ in range(num_envs):
        env = SnakeGameAI(render=False, board_blocks=board_size)
        if walls:
            env.walls = walls
        envs.append(env)

    agent = DQNAgent()

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

    csv_path = os.path.join(log_dir, 'rewards.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'score', 'total_reward', 'steps', 'eps', 'timestamp'])

    try:
        info_path = os.path.join(log_dir, 'info.txt')
        with open(info_path, 'w', encoding='utf-8') as fi:
            fi.write(f'level_path: {level_path}\n')
            fi.write(f'init_checkpoint: {init_checkpoint}\n')
            fi.write(f'episodes: {episodes}\n')
            fi.write(f'max_steps: {max_steps}\n')
            fi.write(f'board_size: {board_size}\n')
            fi.write(f'num_envs: {num_envs}\n')
            fi.write(f'started: {ts}\n')
    except Exception:
        pass

    print(f'Training {episodes} episodes | board={board_size} | '
          f'envs={num_envs} | logs -> {log_dir}')

    best_score = 0
    t_start = time.time()
    recent_scores = []
    recent_rewards = []

    # -- parallel episode loop ------------------------------------------
    # Each env runs its own episode concurrently; we round-robin step them.
    states = [env.reset() for env in envs]
    rewards_acc = [0.0] * num_envs
    steps_acc = [0] * num_envs
    ep_counter = 0  # total finished episodes

    while ep_counter < episodes:
        # Step all envs
        for i, env in enumerate(envs):
            if ep_counter >= episodes:
                break
            action = agent.act(states[i])
            next_state, reward, done, info = env.play_step(action, skip_events=True)
            agent.push(states[i], action, reward, next_state, done)
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

                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([
                        ep_counter, score, f'{ep_reward:.2f}',
                        steps_acc[i], f'{agent.eps:.4f}',
                        datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
                    ])

                if ep_counter % 50 == 0 or ep_counter == 1:
                    elapsed = time.time() - t_start
                    lr = agent.optimizer.param_groups[0]['lr']
                    print(f'ep {ep_counter}/{episodes} | '
                          f'best={best_score} avg50={avg50:.1f} '
                          f'avgR50={avgR50:.1f} '
                          f'eps={agent.eps:.4f} lr={lr:.1e} '
                          f'elapsed={elapsed:.0f}s')
                    agent.step_scheduler(avg50)

                if ep_counter % save_every == 0:
                    agent.save(os.path.join(log_dir, f'model_ep{ep_counter}.pth'))

                # Reset this env for next episode
                states[i] = env.reset()
                rewards_acc[i] = 0.0
                steps_acc[i] = 0

        # One shared update per round (after stepping all envs)
        agent.update()

    # -- save final models ----------------------------------------------
    agent.save(os.path.join(log_dir, save_path))
    agent.save(save_path)
    print(f'Done. Best score: {best_score}. Model saved to {save_path}')

    try:
        run_info = os.path.join(log_dir, 'run_info.txt')
        with open(run_info, 'w', encoding='utf-8') as ri:
            ri.write(f'best_score: {best_score}\n')
            ri.write(f'final_eps: {agent.eps:.6f}\n')
            ri.write(f'total_time: {time.time() - t_start:.1f}s\n')
    except Exception:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Snake DQN agent (headless)')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--max-steps', type=int, default=15000)
    parser.add_argument('--save-every', type=int, default=200)
    parser.add_argument('--level', type=str, default=None, help='Path to level JSON')
    parser.add_argument('--init-checkpoint', type=str, default=None, help='Path to .pth to resume')
    parser.add_argument('--log-name', type=str, default=None, help='Subdirectory name for logs')
    parser.add_argument('--save', type=str, default='model.pth', help='Final model filename')
    parser.add_argument('--board-size', type=int, default=20, help='Board size (curriculum learning)')
    parser.add_argument('--num-envs', type=int, default=1, help='Parallel environments (e.g. 4-8)')
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
    )
