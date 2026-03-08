import pygame
import random
import os
import sys
import json
import subprocess
from enum import Enum
from collections import deque
import numpy as np

from game_layout import (
    recompute_layout as _layout_recompute_layout,
    resize_window as _layout_resize_window,
    get_left_control_rects as _layout_get_left_control_rects,
)
from game_render import (
    update_ui as _render_update_ui,
    draw_panel_box as _render_draw_panel_box,
    fit_text as _render_fit_text,
    wrap_text as _render_wrap_text,
    draw_footer_block as _render_draw_footer_block,
)
from game_designer import level_designer as _designer_level_designer


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


DIR_VECTORS = {
    Direction.RIGHT: (1, 0),
    Direction.LEFT: (-1, 0),
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
}

# Colours
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
PANEL_BG = (24, 24, 24)
PANEL_BORDER = (115, 115, 115)
BOARD_BORDER = (185, 185, 185)
FOOTER_BG = (28, 28, 28)
FOOTER_BORDER = (105, 105, 105)

BLOCK_SIZE = 20
DEFAULT_SPEED = 30
MAX_EPISODE_MOVES = 15000

BOARD_BLOCKS = 20
MIN_WINDOW_W = 1000
MIN_WINDOW_H = 700
LEFT_PANEL_W = 300
UI_MARGIN = 10
FOOTER_H = 112
PANEL_MIN_W = 280
PANEL_MAX_W = 460
MIN_BLOCK_PIXELS = 10


class SnakeGameAI:
    """Snake environment with cell-based internal state.

    All game entities (head, snake, food, walls) are stored as (cx, cy) tuples
    within a logical grid of size ``board_blocks x board_blocks``.  Pixel
    coordinates are computed only for rendering.
    """

    def __init__(self, w=640, h=480, render=True, seed=None, speed=DEFAULT_SPEED,
                 max_episode_steps=MAX_EPISODE_MOVES, board_blocks=BOARD_BLOCKS):
        self.w = max(w, MIN_WINDOW_W)
        self.h = max(h, MIN_WINDOW_H)
        self.render = render
        self.speed = speed
        self.max_episode_steps = max(1, int(max_episode_steps))
        if seed is not None:
            random.seed(seed)

        if self.render:
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h), pygame.RESIZABLE)
            pygame.display.set_caption('Snake AI - Projekt')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(pygame.font.get_default_font(), 25)
            self.small_font = pygame.font.Font(pygame.font.get_default_font(), 18)
        else:
            self.display = None
            self.clock = None
            self.font = None
            self.small_font = None

        self.layout_cfg = {
            'board_blocks': board_blocks,
            'min_window_w': MIN_WINDOW_W,
            'min_window_h': MIN_WINDOW_H,
            'left_panel_w': LEFT_PANEL_W,
            'ui_margin': UI_MARGIN,
            'footer_h': FOOTER_H,
            'panel_min_w': PANEL_MIN_W,
            'panel_max_w': PANEL_MAX_W,
            'min_block_pixels': MIN_BLOCK_PIXELS,
        }
        self.theme = {
            'white': WHITE, 'red': RED, 'blue1': BLUE1, 'blue2': BLUE2,
            'black': BLACK, 'panel_bg': PANEL_BG, 'panel_border': PANEL_BORDER,
            'board_border': BOARD_BORDER, 'footer_bg': FOOTER_BG, 'footer_border': FOOTER_BORDER,
        }

        self.board_blocks = board_blocks
        self.left_panel_width = LEFT_PANEL_W
        self.footer_height = FOOTER_H
        self.left_panel_rect = pygame.Rect(0, 0, self.left_panel_width, self.h)
        self.footer_rect = pygame.Rect(0, self.h - self.footer_height, self.w, self.footer_height)
        self.bs = BLOCK_SIZE
        self.board_w = self.bs * self.board_blocks
        self.board_h = self.bs * self.board_blocks
        self.board_x = 0
        self.board_y = 0
        self.walls = set()

        self._recompute_layout()
        self.reset()

    # -- layout delegates ------------------------------------------------
    def _recompute_layout(self):
        _layout_recompute_layout(self)

    def resize_window(self, w, h):
        _layout_resize_window(self, w, h)

    def _get_left_control_rects(self, panel_h=260):
        return _layout_get_left_control_rects(self, panel_h=panel_h)

    # -- game state ------------------------------------------------------
    def reset(self):
        self.direction = Direction.RIGHT
        center = self.board_blocks // 2
        self.head = (center, center)
        self.snake = [self.head, (center - 1, center), (center - 2, center)]
        self.snake_body_set = set(self.snake[1:])
        self.score = 0
        self.food = None
        self.no_food_slots = False
        self._place_food()
        self.frame_iteration = 0
        self._recent_positions = deque(maxlen=16)
        self._prev_food_dist = self._manhattan_to_food()
        return self.get_state()

    def _manhattan_to_food(self):
        if self.food is None:
            return 0
        return abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])

    def _place_food(self):
        occupied = set(self.snake)
        if self.walls:
            occupied |= self.walls
        free = [
            (cx, cy)
            for cy in range(self.board_blocks)
            for cx in range(self.board_blocks)
            if (cx, cy) not in occupied
        ]
        if not free:
            self.food = self.head
            self.no_food_slots = True
            return False
        self.food = random.choice(free)
        self.no_food_slots = False
        return True

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        cx, cy = pt
        if cx < 0 or cx >= self.board_blocks or cy < 0 or cy >= self.board_blocks:
            return True
        if pt in self.snake_body_set:
            return True
        if self.walls and pt in self.walls:
            return True
        return False

    # -- step ------------------------------------------------------------
    def play_step(self, action, skip_events=False):
        self.frame_iteration += 1

        if self.render and not skip_events:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return self.get_state(), 0, True, {"quit": True}

        if self.no_food_slots:
            return self.get_state(), 0, True, {
                "score": self.score, "board_filled": True, "reason": "no_food_slots"}

        act_idx = self._parse_action(action)
        self._move(act_idx)
        self.snake.insert(0, self.head)
        self.snake_body_set = set(self.snake[1:])

        reward = -0.01
        done = False

        if self.is_collision():
            done = True
            reward = -10
            return self.get_state(), reward, done, {"score": self.score, "reason": "collision"}

        if self.frame_iteration >= self.max_episode_steps:
            done = True
            reward = -10
            return self.get_state(), reward, done, {"score": self.score, "reason": "max_steps"}

        # Distance-based reward shaping
        new_food_dist = self._manhattan_to_food()

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            self._prev_food_dist = self._manhattan_to_food()
        else:
            self.snake.pop()
            self.snake_body_set = set(self.snake[1:])
            if new_food_dist < self._prev_food_dist:
                reward += 0.1
            elif new_food_dist > self._prev_food_dist:
                reward -= 0.1
            self._prev_food_dist = new_food_dist

        # Anti-loop penalty
        if self.head in self._recent_positions:
            reward -= 0.3
        self._recent_positions.append(self.head)

        if self.render:
            self._update_ui()
            if self.clock:
                self.clock.tick(self.speed)

        return self.get_state(), reward, done, {"score": self.score, "reason": "running"}

    # -- render delegates ------------------------------------------------
    def _update_ui(self):
        return _render_update_ui(self)

    def _draw_panel_box(self, rect):
        return _render_draw_panel_box(self, rect)

    def _fit_text(self, text, font, max_width):
        return _render_fit_text(self, text, font, max_width)

    def _wrap_text(self, text, font, max_width):
        return _render_wrap_text(self, text, font, max_width)

    def _draw_footer_block(self, lines):
        return _render_draw_footer_block(self, lines)

    def render_screen(self, mode='human'):
        if not self.render:
            return None
        if mode == 'human':
            return None
        if mode == 'rgb_array':
            arr = pygame.surfarray.array3d(self.display)
            return np.transpose(arr, (1, 0, 2))

    def level_designer(self):
        return _designer_level_designer(self)

    # -- movement --------------------------------------------------------
    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if action == 0:
            new_dir = clock_wise[idx]
        elif action == 1:
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]
        self.direction = new_dir
        dx, dy = DIR_VECTORS[self.direction]
        self.head = (self.head[0] + dx, self.head[1] + dy)

    def _next_cell(self, pos, direction):
        dx, dy = DIR_VECTORS[direction]
        return (pos[0] + dx, pos[1] + dy)

    def _parse_action(self, action):
        if isinstance(action, int):
            if action in (0, 1, 2):
                return action
            raise ValueError('action int must be 0 (straight), 1 (right) or 2 (left)')
        if isinstance(action, (list, tuple)):
            if len(action) == 3:
                if action[0] == 1:
                    return 0
                if action[1] == 1:
                    return 1
                return 2
        if isinstance(action, np.ndarray):
            return self._parse_action(action.tolist())
        raise ValueError('Unsupported action format: %r' % (action,))

    # -- observations ----------------------------------------------------
    def get_state(self):
        """18-feature vector for the RL agent.

        [danger_s, danger_r, danger_l,
         dir_up, dir_down, dir_left, dir_right,
         food_dx, food_dy,
         wall_dist_s, wall_dist_r, wall_dist_l, wall_dist_b,
         body_dist_s, body_dist_r, body_dist_l, body_dist_b,
         snake_length_norm]
        """
        head = self.head
        bb = self.board_blocks

        cw = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = cw.index(self.direction)
        dir_s = cw[idx]
        dir_r = cw[(idx + 1) % 4]
        dir_l = cw[(idx - 1) % 4]
        dir_b = cw[(idx + 2) % 4]

        danger_s = int(self.is_collision(self._next_cell(head, dir_s)))
        danger_r = int(self.is_collision(self._next_cell(head, dir_r)))
        danger_l = int(self.is_collision(self._next_cell(head, dir_l)))

        dir_up = int(self.direction == Direction.UP)
        dir_down = int(self.direction == Direction.DOWN)
        dir_left = int(self.direction == Direction.LEFT)
        dir_right = int(self.direction == Direction.RIGHT)

        food_dx = (self.food[0] - head[0]) / bb
        food_dy = (self.food[1] - head[1]) / bb

        def ray_wall(direction):
            dx, dy = DIR_VECTORS[direction]
            cx, cy = head
            dist = 0
            while True:
                cx += dx
                cy += dy
                dist += 1
                if cx < 0 or cx >= bb or cy < 0 or cy >= bb:
                    return dist / bb
                if self.walls and (cx, cy) in self.walls:
                    return dist / bb

        def ray_body(direction):
            dx, dy = DIR_VECTORS[direction]
            cx, cy = head
            dist = 0
            while True:
                cx += dx
                cy += dy
                dist += 1
                if cx < 0 or cx >= bb or cy < 0 or cy >= bb:
                    return 1.0
                if (cx, cy) in self.snake_body_set:
                    return dist / bb

        wall_s = ray_wall(dir_s)
        wall_r = ray_wall(dir_r)
        wall_l = ray_wall(dir_l)
        wall_b = ray_wall(dir_b)

        body_s = ray_body(dir_s)
        body_r = ray_body(dir_r)
        body_l = ray_body(dir_l)
        body_b = ray_body(dir_b)

        snake_len = len(self.snake) / (bb * bb)

        return [
            danger_s, danger_r, danger_l,
            dir_up, dir_down, dir_left, dir_right,
            food_dx, food_dy,
            wall_s, wall_r, wall_l, wall_b,
            body_s, body_r, body_l, body_b,
            snake_len,
        ]

    def get_grid_state(self):
        """4-channel grid for CNN agent: [head, body, food, walls].
        Returns np.ndarray of shape (4, board_blocks, board_blocks).
        """
        bb = self.board_blocks
        grid = np.zeros((4, bb, bb), dtype=np.float32)
        grid[0, self.head[1], self.head[0]] = 1.0
        for cell in self.snake[1:]:
            grid[1, cell[1], cell[0]] = 1.0
        if self.food:
            grid[2, self.food[1], self.food[0]] = 1.0
        for cell in self.walls:
            grid[3, cell[1], cell[0]] = 1.0
        return grid


# ===================================================================
# Main entry-point / GUI
# ===================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run SnakeGameAI example')
    parser.add_argument('--no-render', action='store_true', help='Run without opening a window')
    args = parser.parse_args()

    g = SnakeGameAI(render=not args.no_render)
    print('Starting game (render=%s)...' % (g.render,))

    if not g.render:
        while True:
            action = random.randint(0, 2)
            state, reward, done, info = g.play_step(action)
            if done:
                print('Game over. Score:', g.score)
                break
    else:
        # ----- GUI mode ------------------------------------------------
        btn_w = 360
        btn_h = 64
        center_x = g.w // 2
        btn1_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 - 150, btn_w, btn_h)
        btn2_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 - 60, btn_w, btn_h)
        btn3_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 + 30, btn_w, btn_h)
        btn4_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 + 120, btn_w, btn_h)
        btn5_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 + 210, btn_w, btn_h)

        g.current_level_name = None
        g.current_level_path = None
        g.current_checkpoint_path = None

        # -- helper: file dialog ----------------------------------------
        def pick_file_dialog(filetypes, fallback_exts=('.pth',),
                             fallback_dirs=('.', 'logs'),
                             fallback_title='Select file - Esc to cancel'):
            try:
                import tkinter as _tk
                from tkinter import filedialog
                root = _tk.Tk()
                root.withdraw()
                path = filedialog.askopenfilename(filetypes=filetypes)
                root.destroy()
                return path
            except Exception:
                def _pygame_file_picker(exts, search_dirs, title):
                    files = []
                    for d in search_dirs:
                        if os.path.isdir(d):
                            for root_dir, _, fns in os.walk(d):
                                for fn in fns:
                                    if any(fn.lower().endswith(e.lower()) for e in exts):
                                        files.append(os.path.join(root_dir, fn))
                    files = sorted(set(files))
                    if not files:
                        return ''
                    screen = pygame.display.get_surface()
                    created_temp = False
                    if screen is None:
                        pygame.display.init()
                        screen = pygame.display.set_mode((640, 400))
                        created_temp = True
                    font_small = pygame.font.SysFont(None, 20)
                    running = True
                    selected = ''
                    while running:
                        for ev in pygame.event.get():
                            if ev.type == pygame.QUIT:
                                running = False
                            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                                running = False
                            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                                mx, my = ev.pos
                                for i, fp in enumerate(files):
                                    rect = pygame.Rect(20, 50 + i * 30,
                                                        screen.get_width() - 40, 28)
                                    if rect.collidepoint(mx, my):
                                        selected = fp
                                        running = False
                                        break
                        screen.fill((30, 30, 30))
                        title_s = font_small.render(title, True, (255, 255, 255))
                        screen.blit(title_s, (20, 10))
                        for i, fp in enumerate(files):
                            y = 50 + i * 30
                            color = (200, 200, 200) if i % 2 == 0 else (170, 170, 170)
                            pygame.draw.rect(screen, color, (18, y,
                                             screen.get_width() - 36, 28))
                            s = font_small.render(os.path.basename(fp), True, (0, 0, 0))
                            screen.blit(s, (25, y + 6))
                        pygame.display.flip()
                        pygame.time.wait(30)
                    if created_temp:
                        pygame.display.quit()
                    return selected

                return _pygame_file_picker(fallback_exts, fallback_dirs, fallback_title)

        # -- helper: torch check ----------------------------------------
        def detect_torch_backend():
            info = {'installed': False, 'version': 'not installed'}
            try:
                import torch as _torch
                info['installed'] = True
                info['version'] = getattr(_torch, '__version__', 'unknown')
            except Exception:
                pass
            return info

        # -- live training loop -----------------------------------------
        def live_train(env, max_episodes=10000, max_steps=MAX_EPISODE_MOVES,
                       init_ckpt=None, eval_only=False):
            AgentDQN = None
            try:
                from dqn_agent import DQNAgent as AgentDQN
            except Exception:
                AgentDQN = None

            if AgentDQN is None:
                msg = 'DQN not available (torch missing)'
                info_timer = 90
                while info_timer > 0:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit(0)
                    env.display.fill(BLACK)
                    env.display.blit(env.font.render(msg, True, WHITE), (10, 40))
                    pygame.display.flip()
                    info_timer -= 1
                    if env.clock:
                        env.clock.tick(30)
                return

            # load level
            if getattr(env, 'current_level_path', None):
                try:
                    with open(env.current_level_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    walls = set()
                    for item in data:
                        cx, cy = int(item[0]), int(item[1])
                        if 0 <= cx < env.board_blocks and 0 <= cy < env.board_blocks:
                            walls.add((cx, cy))
                    env.walls = walls
                    env.level_name = os.path.splitext(
                        os.path.basename(env.current_level_path))[0]
                except Exception:
                    pass

            # create agent
            try:
                agent = AgentDQN()
            except Exception as e:
                msg = f'Failed to create DQN agent: {e}'
                info_timer = 90
                while info_timer > 0:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit(0)
                    env.display.fill(BLACK)
                    env.display.blit(env.font.render(msg, True, WHITE), (10, 40))
                    pygame.display.flip()
                    info_timer -= 1
                    if env.clock:
                        env.clock.tick(30)
                return

            if init_ckpt:
                try:
                    agent.load(init_ckpt)
                except Exception:
                    pass

            if eval_only:
                agent.eps = 0.0
                agent.policy_net.eval()

            max_cells = env.board_blocks * env.board_blocks
            action_names = {0: 'STRAIGHT', 1: 'RIGHT TURN', 2: 'LEFT TURN'}
            recent_scores = []
            recent_steps = []

            torch_mod = None
            try:
                import torch as torch_mod
            except Exception:
                torch_mod = None

            def format_q_values(q_vals):
                if not q_vals or len(q_vals) < 3:
                    return 'Q[S,R,L]: n/a'
                return f'Q[S,R,L]: {q_vals[0]:.2f}, {q_vals[1]:.2f}, {q_vals[2]:.2f}'

            def choose_action_with_debug(cur_state):
                eps_value = float(getattr(agent, 'eps', 0.0))
                explore = random.random() < eps_value
                if hasattr(agent, 'steps'):
                    agent.steps += 1
                if explore:
                    return random.randrange(agent.n_actions), 'explore', eps_value, None
                try:
                    if torch_mod is not None:
                        with torch_mod.no_grad():
                            st = torch_mod.as_tensor(
                                cur_state, dtype=torch_mod.float32).unsqueeze(0)
                            q_out = agent.policy_net(st).squeeze(0).tolist()
                        q_values = [float(v) for v in q_out]
                        best = int(max(range(len(q_values)), key=lambda i: q_values[i]))
                        return best, 'greedy', eps_value, q_values
                except Exception:
                    pass
                return agent.act(cur_state), 'agent_fallback', eps_value, None

            pause_requested = False
            ep = 0
            running = True
            while running and ep < max_episodes:
                ep += 1
                state = env.reset()
                total_reward = 0.0
                paused = False
                panel_bg, btn_pause, btn_plus, btn_minus = env._get_left_control_rects(
                    panel_h=260)

                last_action = None
                last_mode = 'n/a'
                last_eps = float(getattr(agent, 'eps', 0.0))
                last_q_values = None
                ep_steps = 0
                step_info = {}

                for t in range(max_steps):
                    panel_bg, btn_pause, btn_plus, btn_minus = \
                        env._get_left_control_rects(panel_h=260)

                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit(0)
                        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                            running = False
                            break
                        if ev.type == pygame.VIDEORESIZE:
                            env.resize_window(ev.w, ev.h)
                            panel_bg, btn_pause, btn_plus, btn_minus = \
                                env._get_left_control_rects(panel_h=260)
                            max_cells = env.board_blocks * env.board_blocks
                        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                            mx, my = ev.pos
                            if btn_pause.collidepoint(mx, my):
                                if paused:
                                    paused = False
                                else:
                                    pause_requested = True
                            if btn_plus.collidepoint(mx, my):
                                env.speed += 10
                            if btn_minus.collidepoint(mx, my):
                                env.speed = max(1, env.speed - 10)
                    if not running:
                        break

                    transition_ready = False
                    if not paused:
                        action, last_mode, last_eps, last_q_values = \
                            choose_action_with_debug(state)
                        last_action = action
                        next_state, reward, done, step_info = env.play_step(
                            action, skip_events=True)
                        total_reward += reward
                        transition_ready = True
                        ep_steps += 1
                    else:
                        next_state, reward, done, step_info = state, 0, False, {}

                    if transition_ready and not eval_only:
                        agent.push(state, action, reward, next_state, done)
                        agent.update()

                    state = next_state

                    # info panel
                    try:
                        env._draw_panel_box(panel_bg)
                        move_txt = action_names.get(last_action, 'n/a')
                        mode_txt = f'{last_mode} (eps={last_eps:.3f})'
                        q_txt = format_q_values(last_q_values)
                        run_mode_txt = ('Run mode: EVAL' if eval_only
                                        else 'Run mode: TRAIN')
                        if recent_scores:
                            n = min(50, len(recent_scores))
                            avg_s = sum(recent_scores[-50:]) / n
                            avg_t = sum(recent_steps[-50:]) / n
                            avg_txt = f'Avg50: {avg_s:.2f} / {avg_t:.1f}'
                        else:
                            avg_txt = 'Avg50: n/a'

                        lines = [
                            f'Episode: {ep} | Step: {t}',
                            f'Score: {env.score} | Reward: {total_reward:.1f}',
                            avg_txt,
                            f'Speed: {env.speed} FPS | {run_mode_txt}',
                            f'Move: {move_txt} | {mode_txt}',
                            q_txt,
                        ]
                        info_font = env.small_font or env.font
                        line_h = info_font.get_height() + 6
                        for i, ln in enumerate(lines):
                            clipped = env._fit_text(ln, info_font,
                                                    panel_bg.width - 12)
                            s = info_font.render(clipped, True, WHITE)
                            env.display.blit(
                                s, (panel_bg.x + 6, panel_bg.y + 6 + i * line_h))

                        # buttons
                        pygame.draw.rect(env.display,
                                         (180, 180, 100) if paused else (100, 180, 100),
                                         btn_pause)
                        pygame.draw.rect(env.display, PANEL_BORDER, btn_pause, 2)
                        env.display.blit(
                            env.font.render(
                                'Pause (ep)' if not paused else 'Resume',
                                True, BLACK),
                            (btn_pause.x + 8, btn_pause.y + 4))
                        for btn in (btn_plus, btn_minus):
                            pygame.draw.rect(env.display, (140, 140, 140), btn)
                            pygame.draw.rect(env.display, PANEL_BORDER, btn, 2)
                        env.display.blit(env.font.render('+', True, BLACK),
                                         (btn_plus.x + 10, btn_plus.y + 4))
                        env.display.blit(env.font.render('-', True, BLACK),
                                         (btn_minus.x + 12, btn_minus.y + 4))

                        if eval_only:
                            env._draw_footer_block([
                                'Esc: menu | Pause: after episode ends',
                                'Eval mode: eps=0, no weight updates',
                                'Speed +/-: change FPS',
                            ])
                        else:
                            env._draw_footer_block([
                                'Esc: menu | Pause: after episode ends',
                                'Speed +/-: change FPS',
                                'Panel shows model decision process',
                            ])
                        pygame.display.flip()
                    except Exception:
                        pass

                    if paused and env.clock:
                        env.clock.tick(30)

                    occupied = len(env.snake) + len(env.walls)
                    if occupied >= max_cells:
                        try:
                            env.display.blit(
                                env.font.render(
                                    f'Episode {ep} filled board!', True, WHITE),
                                (10, 10))
                            pygame.display.flip()
                        except Exception:
                            pass
                        running = False
                        break

                    if done:
                        if step_info.get('board_filled'):
                            running = False
                        pause_cnt = 30
                        while pause_cnt > 0 and running:
                            for ev in pygame.event.get():
                                if ev.type == pygame.QUIT:
                                    pygame.quit()
                                    sys.exit(0)
                                if (ev.type == pygame.KEYDOWN
                                        and ev.key == pygame.K_ESCAPE):
                                    running = False
                                    break
                            pause_cnt -= 1
                            if env.clock:
                                env.clock.tick(30)
                        break

                recent_scores.append(float(env.score))
                recent_steps.append(float(ep_steps))

                # pause between episodes
                if pause_requested:
                    paused = True
                    pause_requested = False
                    while paused and running:
                        panel_bg, btn_pause, btn_plus, btn_minus = \
                            env._get_left_control_rects(panel_h=260)
                        for ev in pygame.event.get():
                            if ev.type == pygame.QUIT:
                                pygame.quit()
                                sys.exit(0)
                            if ev.type == pygame.VIDEORESIZE:
                                env.resize_window(ev.w, ev.h)
                            if (ev.type == pygame.MOUSEBUTTONDOWN
                                    and ev.button == 1):
                                mx, my = ev.pos
                                if btn_pause.collidepoint(mx, my):
                                    paused = False
                                    break
                                if btn_plus.collidepoint(mx, my):
                                    env.speed += 10
                                if btn_minus.collidepoint(mx, my):
                                    env.speed = max(1, env.speed - 10)
                        try:
                            env._draw_panel_box(panel_bg)
                            n = min(50, len(recent_scores)) if recent_scores else 0
                            if n:
                                avg_s = sum(recent_scores[-50:]) / n
                                avg_t = sum(recent_steps[-50:]) / n
                                avg_txt = f'Avg50: {avg_s:.2f} / {avg_t:.1f}'
                            else:
                                avg_txt = 'Avg50: n/a'
                            lines = [
                                f'Paused after episode: {ep}',
                                f'Score: {env.score} | Reward: {total_reward:.1f}',
                                avg_txt,
                                f'Speed: {env.speed} FPS',
                                f'Last move: {action_names.get(last_action, "n/a")}',
                                format_q_values(last_q_values),
                            ]
                            info_font = env.small_font or env.font
                            line_h = info_font.get_height() + 6
                            for i, ln in enumerate(lines):
                                clipped = env._fit_text(
                                    ln, info_font, panel_bg.width - 12)
                                s = info_font.render(clipped, True, WHITE)
                                env.display.blit(
                                    s, (panel_bg.x + 6,
                                        panel_bg.y + 6 + i * line_h))
                            pygame.draw.rect(env.display, (100, 180, 100),
                                             btn_pause)
                            pygame.draw.rect(env.display, PANEL_BORDER,
                                             btn_pause, 2)
                            env.display.blit(
                                env.font.render('Resume', True, BLACK),
                                (btn_pause.x + 8, btn_pause.y + 4))
                            for btn in (btn_plus, btn_minus):
                                pygame.draw.rect(env.display, (140, 140, 140),
                                                 btn)
                                pygame.draw.rect(env.display, PANEL_BORDER,
                                                 btn, 2)
                            env.display.blit(env.font.render('+', True, BLACK),
                                             (btn_plus.x + 10, btn_plus.y + 4))
                            env.display.blit(env.font.render('-', True, BLACK),
                                             (btn_minus.x + 12, btn_minus.y + 4))
                            env._draw_footer_block([
                                'Paused between episodes',
                                'Resume to continue | Speed +/- works',
                            ])
                            pygame.display.flip()
                        except Exception:
                            pass
                        if env.clock:
                            env.clock.tick(30)

            env.reset()

        # ----- main menu loop ------------------------------------------
        running = True
        while running:
            choice = None
            menu_running = True
            while menu_running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit(0)
                    if event.type == pygame.VIDEORESIZE:
                        g.resize_window(event.w, event.h)
                        center_x = g.w // 2
                        btn1_rect = pygame.Rect(center_x - btn_w // 2,
                                                g.h // 2 - 150, btn_w, btn_h)
                        btn2_rect = pygame.Rect(center_x - btn_w // 2,
                                                g.h // 2 - 60, btn_w, btn_h)
                        btn3_rect = pygame.Rect(center_x - btn_w // 2,
                                                g.h // 2 + 30, btn_w, btn_h)
                        btn4_rect = pygame.Rect(center_x - btn_w // 2,
                                                g.h // 2 + 120, btn_w, btn_h)
                        btn5_rect = pygame.Rect(center_x - btn_w // 2,
                                                g.h // 2 + 210, btn_w, btn_h)
                    if (event.type == pygame.MOUSEBUTTONDOWN
                            and event.button == 1):
                        mx, my = event.pos
                        for tag, rect in [('1', btn1_rect), ('2', btn2_rect),
                                          ('3', btn3_rect), ('4', btn4_rect),
                                          ('5', btn5_rect)]:
                            if rect.collidepoint(mx, my):
                                choice = tag
                                menu_running = False
                                break

                g.display.fill(BLACK)
                title = g.font.render('Snake AI - Menu', True, WHITE)
                g.display.blit(title, (center_x - title.get_width() // 2,
                                       g.h // 2 - 180))
                for rect, label in [
                    (btn1_rect, 'Level Design'),
                    (btn2_rect, 'Classic Snake'),
                    (btn3_rect, 'Train on Level'),
                    (btn4_rect, 'Load checkpoint (.pth)'),
                    (btn5_rect, 'Load level (.json)'),
                ]:
                    pygame.draw.rect(g.display, (50, 50, 50), rect)
                    t = g.font.render(label, True, WHITE)
                    g.display.blit(t, (rect.x + 12,
                                       rect.y + btn_h // 2 - t.get_height() // 2))
                if g.current_level_name:
                    s = g.font.render(f'Level: {g.current_level_name}', True, WHITE)
                    g.display.blit(s, (10, 10))
                if g.current_checkpoint_path:
                    s = g.font.render(
                        f'Checkpoint: {os.path.basename(g.current_checkpoint_path)}',
                        True, WHITE)
                    g.display.blit(s, (10, 36))
                pygame.display.flip()
                if g.clock:
                    g.clock.tick(30)

            # -- handle choice ------------------------------------------
            if choice == '1':
                res = g.level_designer()
                if res is None:
                    info_msg = 'Level edit cancelled.'
                else:
                    g.current_level_name = os.path.basename(res)
                    g.current_level_path = res
                    info_msg = f'Saved level: {g.current_level_name}'
                info_timer = 45
                while info_timer > 0:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit(0)
                    g.display.fill(BLACK)
                    g.display.blit(g.font.render(info_msg, True, WHITE), (10, 40))
                    pygame.display.flip()
                    info_timer -= 1
                    if g.clock:
                        g.clock.tick(30)

            elif choice == '2':
                running_demo = True
                paused = False
                action_names_demo = {0: 'STRAIGHT', 1: 'RIGHT', 2: 'LEFT'}
                panel_bg, btn_pause, btn_plus, btn_minus = \
                    g._get_left_control_rects(panel_h=220)
                step = 0
                last_action_d = None
                state_d = g.get_state()
                while running_demo:
                    panel_bg, btn_pause, btn_plus, btn_minus = \
                        g._get_left_control_rects(panel_h=220)
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit(0)
                        if ev.type == pygame.VIDEORESIZE:
                            g.resize_window(ev.w, ev.h)
                        if (ev.type == pygame.KEYDOWN
                                and ev.key == pygame.K_ESCAPE):
                            running_demo = False
                            break
                        if (ev.type == pygame.MOUSEBUTTONDOWN
                                and ev.button == 1):
                            mx, my = ev.pos
                            if btn_pause.collidepoint(mx, my):
                                paused = not paused
                            if btn_plus.collidepoint(mx, my):
                                g.speed += 10
                            if btn_minus.collidepoint(mx, my):
                                g.speed = max(1, g.speed - 10)
                    if not paused:
                        act = random.randint(0, 2)
                        last_action_d = act
                        state_d, reward, done, info = g.play_step(act)
                        step += 1
                    try:
                        g._draw_panel_box(panel_bg)
                        lines = [
                            f'Step: {step} | Score: {g.score}',
                            f'Speed: {g.speed} FPS',
                            f'Move: {action_names_demo.get(last_action_d, "n/a")}',
                            'Mode: random_policy',
                        ]
                        info_font = g.small_font or g.font
                        lh = info_font.get_height() + 6
                        for i, ln in enumerate(lines):
                            cl = g._fit_text(ln, info_font, panel_bg.width - 12)
                            s = info_font.render(cl, True, WHITE)
                            g.display.blit(
                                s, (panel_bg.x + 6, panel_bg.y + 6 + i * lh))
                        pygame.draw.rect(g.display,
                                         (180, 180, 100) if paused
                                         else (100, 180, 100), btn_pause)
                        pygame.draw.rect(g.display, PANEL_BORDER, btn_pause, 2)
                        g.display.blit(
                            g.font.render('Pause' if not paused else 'Resume',
                                          True, BLACK),
                            (btn_pause.x + 8, btn_pause.y + 4))
                        for btn in (btn_plus, btn_minus):
                            pygame.draw.rect(g.display, (140, 140, 140), btn)
                            pygame.draw.rect(g.display, PANEL_BORDER, btn, 2)
                        g.display.blit(g.font.render('+', True, BLACK),
                                       (btn_plus.x + 10, btn_plus.y + 4))
                        g.display.blit(g.font.render('-', True, BLACK),
                                       (btn_minus.x + 12, btn_minus.y + 4))
                        g._draw_footer_block([
                            'Esc: menu | Pause/Resume with button',
                            'Speed +/-: change FPS',
                            'Classic mode uses random moves',
                        ])
                        pygame.display.flip()
                    except Exception:
                        pass
                    if paused and g.clock:
                        g.clock.tick(30)
                    if not paused and done:
                        pause_cnt = 30
                        while pause_cnt > 0:
                            for ev in pygame.event.get():
                                if ev.type == pygame.QUIT:
                                    pygame.quit()
                                    sys.exit(0)
                            pause_cnt -= 1
                            if g.clock:
                                g.clock.tick(30)
                        running_demo = False
                g.reset()

            elif choice == '4':
                path = pick_file_dialog(
                    [('PyTorch', '*.pth'), ('All files', '*.*')],
                    fallback_exts=('.pth',), fallback_dirs=('.', 'logs'),
                    fallback_title='Select checkpoint (.pth)')
                if path:
                    g.current_checkpoint_path = path
                    info_msg = f'Checkpoint: {os.path.basename(path)}'
                else:
                    info_msg = 'No checkpoint selected.'
                info_timer = 45
                while info_timer > 0:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit(0)
                    g.display.fill(BLACK)
                    g.display.blit(g.font.render(info_msg, True, WHITE), (10, 40))
                    pygame.display.flip()
                    info_timer -= 1
                    if g.clock:
                        g.clock.tick(30)

            elif choice == '5':
                path = pick_file_dialog(
                    [('JSON level', '*.json'), ('All files', '*.*')],
                    fallback_exts=('.json',), fallback_dirs=('levels', '.'),
                    fallback_title='Select level (.json)')
                if path:
                    g.current_level_path = path
                    g.current_level_name = os.path.basename(path)
                    info_msg = f'Level: {g.current_level_name}'
                else:
                    info_msg = 'No level selected.'
                info_timer = 45
                while info_timer > 0:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit(0)
                    g.display.fill(BLACK)
                    g.display.blit(g.font.render(info_msg, True, WHITE), (10, 40))
                    pygame.display.flip()
                    info_timer -= 1
                    if g.clock:
                        g.clock.tick(30)

            elif choice == '3':
                sub_w = 420
                sub_h = 320
                headless = True
                eval_only = False
                torch_info = detect_torch_backend()
                submenu = True
                while submenu:
                    sx = g.w // 2 - sub_w // 2
                    sy = g.h // 2 - sub_h // 2
                    sub_rect = pygame.Rect(sx, sy, sub_w, sub_h)
                    btn_headless = pygame.Rect(sx + 20, sy + 40, 180, 40)
                    btn_live = pygame.Rect(sx + 220, sy + 40, 180, 40)
                    btn_eval = pygame.Rect(sx + 20, sy + 100, 380, 40)
                    btn_start = pygame.Rect(sx + 20, sy + 210, 180, 50)
                    btn_back = pygame.Rect(sx + 220, sy + 210, 180, 50)

                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit(0)
                        if ev.type == pygame.VIDEORESIZE:
                            g.resize_window(ev.w, ev.h)
                        if (ev.type == pygame.MOUSEBUTTONDOWN
                                and ev.button == 1):
                            mx, my = ev.pos
                            if btn_headless.collidepoint(mx, my):
                                headless = True
                            elif btn_live.collidepoint(mx, my):
                                headless = False
                            elif btn_eval.collidepoint(mx, my):
                                eval_only = not eval_only
                            elif btn_start.collidepoint(mx, my):
                                cmd = [sys.executable, 'train.py']
                                if g.current_level_path:
                                    cmd += ['--level', g.current_level_path]
                                if g.current_checkpoint_path:
                                    cmd += ['--init-checkpoint',
                                            g.current_checkpoint_path]
                                if headless:
                                    try:
                                        subprocess.Popen(cmd)
                                        info_msg = 'Background training started.'
                                    except Exception as e:
                                        info_msg = f'Failed: {e}'
                                    submenu = False
                                else:
                                    try:
                                        live_train(
                                            g, max_episodes=10000,
                                            max_steps=MAX_EPISODE_MOVES,
                                            init_ckpt=g.current_checkpoint_path,
                                            eval_only=eval_only)
                                        info_msg = 'Live training finished.'
                                    except Exception as e:
                                        info_msg = f'Live training failed: {e}'
                                    submenu = False
                            elif btn_back.collidepoint(mx, my):
                                submenu = False

                    g.display.fill((20, 20, 20))
                    pygame.draw.rect(g.display, (60, 60, 60), sub_rect)
                    title = g.font.render('Train on Level', True, WHITE)
                    g.display.blit(title,
                                   (sx + sub_w // 2 - title.get_width() // 2,
                                    sy + 8))
                    pygame.draw.rect(
                        g.display,
                        (100, 180, 100) if headless else (150, 150, 150),
                        btn_headless)
                    pygame.draw.rect(
                        g.display,
                        (100, 180, 100) if not headless else (150, 150, 150),
                        btn_live)
                    g.display.blit(g.font.render('Headless', True, BLACK),
                                   (btn_headless.x + 12, btn_headless.y + 8))
                    g.display.blit(g.font.render('Live (demo)', True, BLACK),
                                   (btn_live.x + 12, btn_live.y + 8))

                    eval_ok = not headless
                    ebg = ((100, 180, 100) if (eval_only and eval_ok)
                           else (150, 150, 150))
                    if not eval_ok:
                        ebg = (120, 120, 120)
                    pygame.draw.rect(g.display, ebg, btn_eval)
                    elbl = g._fit_text(
                        f'Eval only (no learning): '
                        f'{"ON" if eval_only else "OFF"}',
                        g.font, btn_eval.width - 16)
                    g.display.blit(g.font.render(elbl, True, BLACK),
                                   (btn_eval.x + 8, btn_eval.y + 8))

                    status_font = g.small_font or g.font
                    line1 = (f"Torch: {torch_info['version']}"
                             if torch_info['installed']
                             else 'Torch: not installed')
                    line2 = 'Algorithm: Double DQN with PER'
                    line3 = 'Eval applies in Live mode only.'
                    for i, ln in enumerate([line1, line2, line3]):
                        cl = g._fit_text(ln, status_font, sub_w - 24)
                        sf = status_font.render(cl, True, WHITE)
                        g.display.blit(
                            sf, (sx + 12,
                                 sy + 152 + i * (status_font.get_height() + 4)))

                    pygame.draw.rect(g.display, (80, 200, 120), btn_start)
                    pygame.draw.rect(g.display, (200, 80, 80), btn_back)
                    g.display.blit(g.font.render('Start', True, BLACK),
                                   (btn_start.x + 60, btn_start.y + 12))
                    g.display.blit(g.font.render('Back', True, BLACK),
                                   (btn_back.x + 68, btn_back.y + 14))
                    pygame.display.flip()
                    if g.clock:
                        g.clock.tick(30)

                if 'info_msg' in locals() and info_msg:
                    info_timer = 45
                    while info_timer > 0:
                        for ev in pygame.event.get():
                            if ev.type == pygame.QUIT:
                                pygame.quit()
                                sys.exit(0)
                        g.display.fill(BLACK)
                        g.display.blit(
                            g.font.render(info_msg, True, WHITE), (10, 40))
                        pygame.display.flip()
                        info_timer -= 1
                        if g.clock:
                            g.clock.tick(30)
