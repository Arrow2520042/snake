"""Replay a saved episode CSV visually in the Snake GUI.

Reads a rewards.csv from a training run and replays the last (or chosen)
episode by re-running the environment with the saved checkpoint.

Usage:
    python replay_viewer.py logs/20260308-033349/v1.pth
    python replay_viewer.py model.pth --level levels/test1.json --speed 10
"""

import argparse
import sys
import pygame
from game import SnakeGameAI


def replay(checkpoint, level=None, speed=10, board_size=20, episodes=5):
    from dqn_agent import DQNAgent

    env = SnakeGameAI(render=True, speed=speed, board_blocks=board_size)

    if level:
        import json
        with open(level, 'r', encoding='utf-8') as f:
            data = json.load(f)
        walls = set()
        for item in data:
            cx, cy = int(item[0]), int(item[1])
            if 0 <= cx < env.board_blocks and 0 <= cy < env.board_blocks:
                walls.add((cx, cy))
        env.walls = walls

    agent = DQNAgent()
    agent.load(checkpoint)
    agent.eps = 0.0
    agent.policy_net.eval()

    print(f'Replaying with checkpoint: {checkpoint}')
    print(f'Speed: {speed} FPS | Episodes: {episodes}')

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.VIDEORESIZE:
                    env.resize_window(event.w, event.h)

            action = agent.act(state)
            state, reward, done, info = env.play_step(action, skip_events=True)
            total_reward += reward
            steps += 1

        print(f'Episode {ep}: score={env.score} reward={total_reward:.1f} steps={steps}')

        # Brief pause between episodes
        pause = 60
        while pause > 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
            pause -= 1
            if env.clock:
                env.clock.tick(30)

    pygame.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replay Snake agent from checkpoint')
    parser.add_argument('checkpoint', help='Path to .pth checkpoint')
    parser.add_argument('--level', type=str, default=None, help='Level JSON path')
    parser.add_argument('--speed', type=int, default=10, help='Replay FPS')
    parser.add_argument('--board-size', type=int, default=20)
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to replay')
    args = parser.parse_args()

    replay(args.checkpoint, level=args.level, speed=args.speed,
           board_size=args.board_size, episodes=args.episodes)
