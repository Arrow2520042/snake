import argparse
import os
import csv
import datetime
import json
import numpy as np
from game import SnakeGameAI, Point
from dqn_agent import DQNAgent

DEFAULT_MAX_STEPS = 15000


def train(episodes=2000, max_steps=DEFAULT_MAX_STEPS, save_path='model.pth', device='auto', init_checkpoint=None):
    env = SnakeGameAI(render=False)
    # If a level path was provided via env var, try to load it into env.walls
    lvl = os.environ.get('SNAKE_LEVEL_PATH')
    if lvl:
        try:
            with open(lvl, 'r', encoding='utf-8') as f:
                data = json.load(f)
            conv = set()
            for item in data:
                try:
                    cx, cy = int(item[0]), int(item[1])
                    px = getattr(env, 'board_x', 0) + cx * getattr(env, 'bs', 20)
                    py = getattr(env, 'board_y', 0) + cy * getattr(env, 'bs', 20)
                    conv.add(Point(int(px), int(py)))
                except Exception:
                    pass
            env.walls = conv
            env.level_name = os.path.splitext(os.path.basename(lvl))[0]
        except Exception:
            env.level_name = 'default'

    # resolve device
    if device == 'auto':
        try:
            import torch as _torch
            dev = 'cuda' if _torch.cuda.is_available() else 'cpu'
        except Exception:
            dev = 'cpu'
    else:
        dev = device
    agent = DQNAgent(device=dev)
    if init_checkpoint:
        try:
            agent.load(init_checkpoint)
            print('Loaded init checkpoint:', init_checkpoint)
        except Exception as e:
            print('Warning: failed to load init checkpoint:', e)

    # prepare logging directory per-level
    level_name = getattr(env, 'level_name', None) or 'default'
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logs_root = 'logs'
    logs_dir = os.path.join(logs_root, level_name, timestamp)
    os.makedirs(logs_dir, exist_ok=True)

    # write run info for reproducibility
    try:
        info_path = os.path.join(logs_dir, 'info.txt')
        with open(info_path, 'w', encoding='utf-8') as fi:
            fi.write(f'timestamp: {timestamp}\n')
            fi.write(f'level: {level_name}\n')
            fi.write(f'save_path: {save_path}\n')
            fi.write(f'device: {dev}\n')
            fi.write(f'episodes: {episodes}\n')
    except Exception:
        pass

    # CSV for episode metrics
    csv_path = os.path.join(logs_dir, 'rewards.csv')
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'episode',
        'reward',
        'score',
        'steps',
        'avg_reward_last50',
        'avg_score_last50',
        'avg_steps_last50',
        'eps'
    ])

    # write run info
    try:
        info_path = os.path.join(logs_dir, 'run_info.txt')
        with open(info_path, 'w', encoding='utf-8') as infof:
            infof.write(f'timestamp: {timestamp}\n')
            infof.write(f'episodes: {episodes}\n')
            infof.write(f'max_steps: {max_steps}\n')
            infof.write(f'save_path: {save_path}\n')
            infof.write(f'device: {device}\n')
    except Exception:
        pass

    rewards = []
    scores = []
    steps_hist = []
    for ep in range(1, episodes + 1):
        state = env.reset()
        # if a level file was provided via env, derive nicer name
        if hasattr(env, 'level_name') and env.level_name:
            level_name = env.level_name
        total_reward = 0
        ep_steps = 0
        for t in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, info = env.play_step(action)
            total_reward += reward
            ep_steps = t + 1
            agent.push(state, action, reward, next_state, done)
            agent.update()
            if t % 100 == 0:
                agent.sync_target()
            state = next_state
            if done:
                break

        rewards.append(total_reward)
        scores.append(float(env.score))
        steps_hist.append(float(ep_steps))

        avg_reward_50 = float(np.mean(rewards[-50:])) if rewards else 0.0
        avg_score_50 = float(np.mean(scores[-50:])) if scores else 0.0
        avg_steps_50 = float(np.mean(steps_hist[-50:])) if steps_hist else 0.0

        if ep % 50 == 0:
            print(
                f'Episode {ep}/{episodes}  '
                f'avg_reward_last50={avg_reward_50:.2f}  '
                f'avg_score_last50={avg_score_50:.2f}  '
                f'avg_steps_last50={avg_steps_50:.1f}  '
                f'eps={agent.eps:.3f}'
            )

        # write metrics to CSV per-episode
        csv_writer.writerow([
            ep,
            float(total_reward),
            float(env.score),
            float(ep_steps),
            avg_reward_50,
            avg_score_50,
            avg_steps_50,
            float(getattr(agent, 'eps', 0.0))
        ])

        # periodic checkpoint
        if ep % 200 == 0:
            base = 'model'
            ckpt_path = os.path.join(logs_dir, f'{base}_ep{ep}.pth')
            try:
                agent.save(ckpt_path)
                print(
                    f'Saved checkpoint to {ckpt_path}  '
                    f'avg_score_last50={avg_score_50:.2f}  '
                    f'avg_steps_last50={avg_steps_50:.1f}'
                )
            except Exception as e:
                print('Warning: failed to save checkpoint:', e)

    # final save into logs dir and to requested save_path
    final_name = os.path.join(logs_dir, os.path.basename(save_path))
    try:
        agent.save(final_name)
        print('Final model saved to', final_name)
    except Exception as e:
        print('Warning: failed to save final model to logs dir:', e)

    # also save to user-requested path if different
    if os.path.abspath(final_name) != os.path.abspath(save_path):
        try:
            agent.save(save_path)
            print('Final model also saved to', save_path)
        except Exception as e:
            print('Warning: failed to save final model to requested path:', e)

    csv_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--max-steps', type=int, default=DEFAULT_MAX_STEPS, help='Maximum moves per episode')
    parser.add_argument('--level', type=str, default='')
    parser.add_argument('--device', type=str, default='auto', help="Device for DQN: 'auto','cpu' or 'cuda'")
    parser.add_argument('--checkpoint-interval', type=int, default=200)
    parser.add_argument('--save', type=str, default='model.pth')
    parser.add_argument('--init-checkpoint', type=str, default='', help='Path to .pth to initialize DQN agent before training')
    args = parser.parse_args()

    if args.level:
        os.environ['SNAKE_LEVEL_PATH'] = args.level
    train(
        episodes=args.episodes,
        max_steps=args.max_steps,
        save_path=args.save,
        device=args.device,
        init_checkpoint=(args.init_checkpoint or None)
    )
