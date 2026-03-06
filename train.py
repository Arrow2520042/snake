import argparse
import time
import os
import csv
import datetime
import json
import numpy as np
from game import SnakeGameAI, Point
from rl_agent import QLearningAgent
try:
    from dqn_agent import DQNAgent
except Exception:
    DQNAgent = None


def train(episodes=2000, max_steps=1000, save_path='q_table.npy', device='auto'):
    env = SnakeGameAI(render=False)
    # If a level path was provided via env var, try to load it into env.walls
    lvl = os.environ.get('SNAKE_LEVEL_PATH')
    if lvl:
        try:
            with open(lvl, 'r', encoding='utf-8') as f:
                data = json.load(f)
            env.walls = set(Point(int(x), int(y)) for x, y in data)
            env.level_name = os.path.splitext(os.path.basename(lvl))[0]
        except Exception:
            env.level_name = 'default'
    # choose agent based on save_path extension or availability
    use_dqn = False
    agent = None
    # allow train to select based on save_path extension
    if save_path.endswith('.pth'):
        if DQNAgent is None:
            raise RuntimeError('DQN requested but dqn_agent not available (install torch)')
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
        use_dqn = True
    else:
        agent = QLearningAgent()

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
            fi.write(f'device: {dev if save_path.endswith(".pth") else "cpu"}\n')
            fi.write(f'episodes: {episodes}\n')
    except Exception:
        pass

    # CSV for episode metrics
    csv_path = os.path.join(logs_dir, 'rewards.csv')
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['episode', 'reward', 'avg_last50', 'eps'])

    # write run info
    try:
        info_path = os.path.join(logs_dir, 'run_info.txt')
        with open(info_path, 'w', encoding='utf-8') as infof:
            infof.write(f'timestamp: {timestamp}\n')
            infof.write(f'episodes: {episodes}\n')
            infof.write(f'max_steps: {max_steps}\n')
            infof.write(f'save_path: {save_path}\n')
            infof.write(f'use_dqn: {use_dqn}\n')
            infof.write(f'device: {device}\n')
    except Exception:
        pass

    rewards = []
    for ep in range(1, episodes + 1):
        state = env.reset()
        # if a level file was provided via env, derive nicer name
        if hasattr(env, 'level_name') and env.level_name:
            level_name = env.level_name
        total_reward = 0
        for t in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, info = env.play_step(action)
            total_reward += reward
            if use_dqn:
                agent.push(state, action, reward, next_state, done)
                agent.update()
                if t % 100 == 0:
                    agent.sync_target()
            else:
                agent.learn(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

        rewards.append(total_reward)
        if ep % 50 == 0:
            avg = np.mean(rewards[-50:])
            print(f'Episode {ep}/{episodes}  avg_reward_last50={avg:.2f}  eps={agent.eps:.3f}')

        # write metrics to CSV per-episode
        avg50 = float(np.mean(rewards[-50:])) if len(rewards) >= 1 else float(total_reward)
        csv_writer.writerow([ep, float(total_reward), avg50, float(getattr(agent, 'eps', 0.0))])

        # periodic checkpoint
        if ep % 200 == 0:
            base = 'model'
            ext = '.pth' if use_dqn or save_path.endswith('.pth') else '.npy'
            ckpt_path = os.path.join(logs_dir, f'{base}_ep{ep}{ext}')
            try:
                agent.save(ckpt_path)
                print('Saved checkpoint to', ckpt_path)
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
    parser.add_argument('--level', type=str, default='')
    parser.add_argument('--device', type=str, default='auto', help="Device for DQN: 'auto','cpu' or 'cuda'")
    parser.add_argument('--level', type=str, default='')
    parser.add_argument('--checkpoint-interval', type=int, default=200)
    parser.add_argument('--save', type=str, default='q_table.npy')
    parser.add_argument('--algo', choices=['tabular', 'dqn'], default='tabular')
    args = parser.parse_args()

    # override save extension for dqn if requested
    save_path = args.save
    if args.algo == 'dqn':
        if save_path.endswith('.npy'):
            save_path = save_path.replace('.npy', '.pth')
    # if level provided, pass into environment via env.level_name attribute by setting
    if args.level:
        os.environ['SNAKE_LEVEL_PATH'] = args.level
    train(episodes=args.episodes, save_path=save_path, device=args.device)
