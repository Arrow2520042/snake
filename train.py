"""Headless training script for Snake DQN agent.

Usage:
    python train.py --episodes 1000
    python train.py --level levels/mymap.json --board-size 20
    python train.py --init-checkpoint model.pth --episodes 500
"""

import argparse
import csv
import datetime
import json
import os
import time

from game import SnakeGameAI


def train(episodes=1000, max_steps=15000, save_every=200, level_path=None,
          init_checkpoint=None, log_name=None, save_path='model.pth',
          board_size=20):
    from dqn_agent import DQNAgent

    env = SnakeGameAI(render=False, board_blocks=board_size)

    # Load level walls
    if level_path and os.path.isfile(level_path):
        with open(level_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        walls = set()
        for item in data:
            cx, cy = int(item[0]), int(item[1])
            if 0 <= cx < env.board_blocks and 0 <= cy < env.board_blocks:
                walls.add((cx, cy))
        env.walls = walls
        print(f'Loaded level: {level_path} ({len(walls)} walls)')

    agent = DQNAgent()

    if init_checkpoint and os.path.isfile(init_checkpoint):
        agent.load(init_checkpoint)
        print(f'Loaded checkpoint: {init_checkpoint}')

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
            fi.write(f'started: {ts}\n')
    except Exception:
        pass

    print(f'Training {episodes} episodes | board={board_size} | '
          f'logs -> {log_dir}')

    best_score = 0
    t_start = time.time()

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        ep_steps = 0

        for t in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, info = env.play_step(action, skip_events=True)
            agent.push(state, action, reward, next_state, done)
            agent.update()
            total_reward += reward
            state = next_state
            ep_steps += 1
            if done:
                break

        score = env.score
        if score > best_score:
            best_score = score
            agent.save(os.path.join(log_dir, 'best.pth'))

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                ep, score, f'{total_reward:.2f}', ep_steps,
                f'{agent.eps:.4f}',
                datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
            ])

        if ep % 50 == 0 or ep == 1:
            elapsed = time.time() - t_start
            print(f'ep {ep}/{episodes} | score={score} best={best_score} '
                  f'reward={total_reward:.1f} eps={agent.eps:.4f} '
                  f'elapsed={elapsed:.0f}s')

        if ep % save_every == 0:
            ckpt = os.path.join(log_dir, f'model_ep{ep}.pth')
            agent.save(ckpt)

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
    )
