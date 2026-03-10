"""Hyperparameter sweep for Snake DQN agent.

Runs multiple training sessions with different hyperparameter combinations
and logs results for comparison.

Usage:
    python sweep.py
    python sweep.py --episodes 500 --level levels/test1.json
"""

import argparse
import csv
import datetime
import itertools
import os
import time

from game import SnakeGameAI


def run_single(params, episodes, max_steps, level_path, board_size):
    from dqn_agent import DQNAgent

    env = SnakeGameAI(render=False, board_blocks=board_size)

    if level_path and os.path.isfile(level_path):
        import json
        with open(level_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        walls = set()
        for item in data:
            cx, cy = int(item[0]), int(item[1])
            if 0 <= cx < env.board_blocks and 0 <= cy < env.board_blocks:
                walls.add((cx, cy))
        env.walls = walls

    agent = DQNAgent(
        lr=params.get('lr', 1e-3),
        gamma=params.get('gamma', 0.99),
        batch_size=params.get('batch_size', 64),
        tau=params.get('tau', 0.005),
    )
    agent.eps_decay = params.get('eps_decay', 0.9995)

    scores = []
    best_score = 0
    t0 = time.time()

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        for t in range(max_steps):
            action = agent.act(state)
            ns, reward, done, info = env.play_step(action, skip_events=True)
            agent.push(0, state, action, reward, ns, done)
            agent.update()
            total_reward += reward
            state = ns
            if done:
                break
        scores.append(env.score)
        if env.score > best_score:
            best_score = env.score
        agent.decay_epsilon()

    elapsed = time.time() - t0
    avg_last50 = sum(scores[-50:]) / min(50, len(scores)) if scores else 0
    return {
        'best_score': best_score,
        'avg_last50': avg_last50,
        'elapsed': elapsed,
        'final_eps': agent.eps,
    }


def sweep(episodes=500, max_steps=15000, level_path=None, board_size=20):
    search_space = {
        'lr': [1e-3, 5e-4, 1e-4],
        'gamma': [0.99, 0.95],
        'batch_size': [64, 128],
        'tau': [0.005, 0.01],
        'eps_decay': [0.9995, 0.999],
    }

    keys = sorted(search_space.keys())
    combos = list(itertools.product(*(search_space[k] for k in keys)))
    print(f'Sweep: {len(combos)} combinations x {episodes} episodes each')

    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('logs', f'sweep_{ts}')
    os.makedirs(log_dir, exist_ok=True)

    results_path = os.path.join(log_dir, 'sweep_results.csv')
    with open(results_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(keys + ['best_score', 'avg_last50', 'elapsed', 'final_eps'])

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        label = ' | '.join(f'{k}={v}' for k, v in params.items())
        print(f'[{i + 1}/{len(combos)}] {label}')

        result = run_single(params, episodes, max_steps, level_path,
                            board_size)

        with open(results_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(
                [params[k] for k in keys]
                + [result['best_score'], f"{result['avg_last50']:.2f}",
                   f"{result['elapsed']:.1f}", f"{result['final_eps']:.4f}"]
            )

        print(f'  -> best={result["best_score"]} avg50={result["avg_last50"]:.2f} '
              f'time={result["elapsed"]:.0f}s')

    print(f'\nSweep complete. Results: {results_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter sweep for Snake DQN')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--max-steps', type=int, default=15000)
    parser.add_argument('--level', type=str, default=None)
    parser.add_argument('--board-size', type=int, default=20)
    args = parser.parse_args()

    sweep(episodes=args.episodes, max_steps=args.max_steps,
          level_path=args.level, board_size=args.board_size)
