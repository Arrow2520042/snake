import pygame
import random
import os
import sys
import json
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

_CW = (Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP)
_CW_IDX = {d: i for i, d in enumerate(_CW)}


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
                 max_episode_steps=MAX_EPISODE_MOVES, board_blocks=BOARD_BLOCKS,
                 state_mode='features', simple_rewards=False):
        self.w = max(w, MIN_WINDOW_W)
        self.h = max(h, MIN_WINDOW_H)
        self.render = render
        self.speed = speed
        self.max_episode_steps = max(1, int(max_episode_steps))
        self.state_mode = state_mode      # 'features' or 'grid'
        self.simple_rewards = simple_rewards
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
        self._steps_since_food = 0
        self._recent_positions = deque(maxlen=64)
        self._recent_set = set()
        self._prev_food_dist = self._manhattan_to_food()
        self._just_ate = False
        self._cached_flood = None
        return self._get_obs()

    def _get_obs(self):
        """Return observation matching the current state_mode."""
        if self.state_mode == 'grid':
            return self.get_grid_state()
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
                    return self._get_obs(), 0, True, {"quit": True}

        if self.no_food_slots:
            return self._get_obs(), 0, True, {
                "score": self.score, "board_filled": True, "reason": "no_food_slots"}

        act_idx = action if isinstance(action, int) and 0 <= action <= 2 else self._parse_action(action)
        self._move(act_idx)

        # Incremental body set: old head becomes body, add to set
        old_head = self.snake[0] if self.snake else None
        self.snake.insert(0, self.head)
        if old_head is not None:
            self.snake_body_set.add(old_head)

        reward = -0.01
        done = False
        self._steps_since_food += 1
        self._just_ate = False

        if self.is_collision():
            done = True
            reward = -10
            return self._get_obs(), reward, done, {"score": self.score, "reason": "collision"}

        if self.frame_iteration >= self.max_episode_steps:
            done = True
            reward = -10
            return self._get_obs(), reward, done, {"score": self.score, "reason": "max_steps"}

        # Per-food timeout: kill episode if snake stalls too long without eating
        food_timeout = 4 * self.board_blocks * self.board_blocks
        if self._steps_since_food > food_timeout:
            done = True
            reward = -10
            return self._get_obs(), reward, done, {"score": self.score, "reason": "food_timeout"}

        # Distance-based reward shaping (reduced to not punish necessary detours)
        new_food_dist = abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])

        if self.head == self.food:
            self._just_ate = True
            self.score += 1
            self._steps_since_food = 0
            self._place_food()
            self._prev_food_dist = abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])
            if self.simple_rewards:
                reward = 10
            else:
                # Scale food reward by post-eat safety; penalise trap-food
                post_eat_ratio = self._flood_fill_ratio()
                if post_eat_ratio < 0.10:
                    reward = -5
                elif post_eat_ratio < 0.15:
                    reward = 2
                elif post_eat_ratio < 0.3:
                    reward = 5
                elif post_eat_ratio < 0.5:
                    reward = 8
                else:
                    reward = 10
        else:
            tail = self.snake.pop()
            self.snake_body_set.discard(tail)
            if not self.simple_rewards:
                if new_food_dist < self._prev_food_dist:
                    reward += 0.03
                elif new_food_dist > self._prev_food_dist:
                    reward -= 0.03
            self._prev_food_dist = new_food_dist

        if not self.simple_rewards:
            # Space-awareness penalty: light, no body_ratio scaling
            reachable_ratio = self._flood_fill_ratio()
            self._cached_flood = reachable_ratio  # cache for get_state
            if reachable_ratio < 0.2:
                reward -= 0.3
            elif reachable_ratio < 0.4:
                reward -= 0.1

            # Hunger penalty: escalating cost for circling without eating
            if self._steps_since_food > self.board_blocks * 2:
                hunger_ratio = self._steps_since_food / food_timeout
                reward -= 0.1 * hunger_ratio

            # Anti-loop penalty – larger window (64) and scaled by snake length
            if self.head in self._recent_set:
                loop_penalty = 0.3 + 0.2 * (len(self.snake) / (self.board_blocks ** 2))
                reward -= loop_penalty
            self._recent_positions.append(self.head)
            self._recent_set = set(self._recent_positions)
        else:
            # simple_rewards mode
            # Hunger penalty: escalating cost for idleness (stronger than complex mode)
            if self._steps_since_food > self.board_blocks:
                hunger_ratio = self._steps_since_food / food_timeout
                reward -= 0.3 * hunger_ratio
            if self.state_mode == 'features':
                self._cached_flood = self._flood_fill_ratio()

        if self.render:
            self._update_ui()
            if self.clock:
                self.clock.tick(self.speed)

        return self._get_obs(), reward, done, {"score": self.score, "reason": "running"}

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


    def level_designer(self):
        return _designer_level_designer(self)

    # -- movement --------------------------------------------------------
    def _move(self, action):
        idx = _CW_IDX[self.direction]
        if action == 1:
            idx = (idx + 1) & 3
        elif action == 2:
            idx = (idx - 1) & 3
        self.direction = _CW[idx]
        dx, dy = DIR_VECTORS[self.direction]
        self.head = (self.head[0] + dx, self.head[1] + dy)


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
        """22-feature vector for the RL agent."""
        hx, hy = self.head
        bb = self.board_blocks
        inv_bb = 1.0 / bb

        idx = _CW_IDX[self.direction]
        dir_s = _CW[idx]
        dir_r = _CW[(idx + 1) & 3]
        dir_l = _CW[(idx - 1) & 3]
        dir_b = _CW[(idx + 2) & 3]
        # Check for dangers in each direction
        # Tail tip: safe only if snake did NOT just eat (tail moves on next step)
        tail_tip = None
        if not self._just_ate and len(self.snake) > 1:
            tail_tip = self.snake[-1]
        dxs, dys = DIR_VECTORS[dir_s]
        dxr, dyr = DIR_VECTORS[dir_r]
        dxl, dyl = DIR_VECTORS[dir_l]
        danger_s = int(self._is_blocked_excluding(hx + dxs, hy + dys, tail_tip))
        danger_r = int(self._is_blocked_excluding(hx + dxr, hy + dyr, tail_tip))
        danger_l = int(self._is_blocked_excluding(hx + dxl, hy + dyl, tail_tip))
        # Food delta is normalized to [-1, 1] range by dividing by board_blocks (max possible distance)
        food_dx = (self.food[0] - hx) * inv_bb
        food_dy = (self.food[1] - hy) * inv_bb

        body_set = self.snake_body_set
        walls = self.walls

        def ray_wall(d):
            dx, dy = DIR_VECTORS[d]
            cx, cy = hx + dx, hy + dy
            dist = 1
            while 0 <= cx < bb and 0 <= cy < bb:
                if walls and (cx, cy) in walls:
                    return dist * inv_bb
                cx += dx; cy += dy; dist += 1
            return dist * inv_bb

        def ray_body(d):
            dx, dy = DIR_VECTORS[d]
            cx, cy = hx + dx, hy + dy
            dist = 1
            while 0 <= cx < bb and 0 <= cy < bb:
                if (cx, cy) in body_set and (cx, cy) != tail_tip:
                    return dist * inv_bb
                cx += dx; cy += dy; dist += 1
            return 1.0
        # Flood fill ratio: use cached value from play_step if available
        flood_ratio = getattr(self, '_cached_flood', None)
        if flood_ratio is None:
            flood_ratio = self._flood_fill_ratio()

        # Food safety: simulated flood_fill if snake were 1 longer (ate food)
        food_safety = self._flood_fill_ratio(extra_block=self.food)

        # Tail direction (normalised vector from head to tail)
        tail = self.snake[-1]
        tail_dx = (tail[0] - hx) * inv_bb
        tail_dy = (tail[1] - hy) * inv_bb
        # Tail-chase distance (Manhattan, normalized) – key for long snake survival
        tail_dist = (abs(tail[0] - hx) + abs(tail[1] - hy)) * inv_bb

        # Per-action flood fill: look-ahead reachable space for each move
        action_flood_s = self._action_flood_fill(0)
        action_flood_r = self._action_flood_fill(1)
        action_flood_l = self._action_flood_fill(2)

        # Order: 3 danger, 4 direction, 2 food delta, 4 ray-wall, 4 ray-body,
        #        1 normalized length, 1 flood_fill, 1 food_safety, 2 tail_dir,
        #        1 tail_dist, 3 action_flood = 26
        return [
            danger_s, danger_r, danger_l,
            int(self.direction == Direction.UP),
            int(self.direction == Direction.DOWN),
            int(self.direction == Direction.LEFT),
            int(self.direction == Direction.RIGHT),
            food_dx, food_dy,
            ray_wall(dir_s), ray_wall(dir_r), ray_wall(dir_l), ray_wall(dir_b),
            ray_body(dir_s), ray_body(dir_r), ray_body(dir_l), ray_body(dir_b),
            len(self.snake) / (bb * bb),
            flood_ratio,
            food_safety,
            tail_dx, tail_dy,
            tail_dist,
            action_flood_s, action_flood_r, action_flood_l,
        ]


    def _is_blocked_excluding(self, cx, cy, exclude=None):
        """Collision check that treats *exclude* cell as passable (for tail)."""
        if cx < 0 or cx >= self.board_blocks or cy < 0 or cy >= self.board_blocks:
            return True
        if (cx, cy) in self.snake_body_set and (cx, cy) != exclude:
            return True
        if self.walls and (cx, cy) in self.walls:
            return True
        return False

    def _flood_fill_ratio(self, extra_block=None):
        """BFS from head – returns fraction of free cells reachable.

        If *extra_block* is given, treat it as an additional body cell
        (used to simulate the effect of eating food).
        """
        bb = self.board_blocks
        body_set = self.snake_body_set
        walls = self.walls
        visited = set()
        queue = deque()
        queue.append(self.head)
        visited.add(self.head)
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in visited:
                    continue
                if nx < 0 or nx >= bb or ny < 0 or ny >= bb:
                    continue
                if (nx, ny) in body_set:
                    continue
                if extra_block and (nx, ny) == extra_block:
                    continue
                if walls and (nx, ny) in walls:
                    continue
                visited.add((nx, ny))
                queue.append((nx, ny))
        extra_count = 1 if extra_block and extra_block not in body_set else 0
        total_free = bb * bb - len(body_set) - extra_count - (len(walls) if walls else 0)
        if total_free <= 0:
            return 0.0
        return len(visited) / total_free

    def _action_flood_fill(self, action_idx):
        """Flood fill ratio after simulating action (0=straight, 1=right, 2=left).

        Predicts reachable space one step ahead so the agent can avoid
        moves that box it in.
        """
        idx = _CW_IDX[self.direction]
        if action_idx == 1:
            idx = (idx + 1) & 3
        elif action_idx == 2:
            idx = (idx - 1) & 3
        d = _CW[idx]
        dx, dy = DIR_VECTORS[d]
        nx, ny = self.head[0] + dx, self.head[1] + dy
        bb = self.board_blocks
        # Immediate collision → 0
        if nx < 0 or nx >= bb or ny < 0 or ny >= bb:
            return 0.0
        if self.walls and (nx, ny) in self.walls:
            return 0.0
        tail_tip = self.snake[-1] if len(self.snake) > 1 else None
        if (nx, ny) in self.snake_body_set and (nx, ny) != tail_tip:
            return 0.0
        # BFS from (nx, ny); old head becomes body, tail moves away
        body_set = self.snake_body_set
        walls = self.walls
        head = self.head
        visited = set()
        queue = deque()
        queue.append((nx, ny))
        visited.add((nx, ny))
        while queue:
            cx, cy = queue.popleft()
            for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ax, ay = cx + ddx, cy + ddy
                if (ax, ay) in visited:
                    continue
                if ax < 0 or ax >= bb or ay < 0 or ay >= bb:
                    continue
                if (ax, ay) == head:  # old head is now body
                    continue
                if (ax, ay) in body_set and (ax, ay) != tail_tip:
                    continue
                if walls and (ax, ay) in walls:
                    continue
                visited.add((ax, ay))
                queue.append((ax, ay))
        # blocked = body_set + old head - tail (tail moves away)
        blocked_body = len(body_set) + 1
        if tail_tip is not None:
            blocked_body -= 1
        total_free = bb * bb - blocked_body - (len(walls) if walls else 0)
        if total_free <= 0:
            return 0.0
        return len(visited) / total_free

    def get_grid_state(self):
        """Grid state for CNN agent.

        Returns flat float32 array: [4 channels * board^2, dir_onehot(4), norm_len(1)]
        Channels:
          0: body_age (float 0–1, newest body segment ~1.0, tail ~0.0)
          1: head (binary)
          2: food (binary)
          3: walls (binary)
        """
        bb = self.board_blocks
        grid = np.zeros((4, bb, bb), dtype=np.float32)

        # Channel 0: body age (normalized, head excluded — separate channel)
        n = len(self.snake)
        if n > 1:
            for i in range(1, n):
                cell = self.snake[i]
                grid[0, cell[1], cell[0]] = 1.0 - (i / n)

        # Channel 1: head (guard against out-of-bounds on collision frame)
        hx, hy = self.head
        if 0 <= hx < bb and 0 <= hy < bb:
            grid[1, hy, hx] = 1.0

        # Channel 2: food
        if self.food:
            grid[2, self.food[1], self.food[0]] = 1.0

        # Channel 3: walls
        if self.walls:
            for cell in self.walls:
                grid[3, cell[1], cell[0]] = 1.0

        # Auxiliary features (appended after flattened grid)
        dir_oh = [
            float(self.direction == Direction.UP),
            float(self.direction == Direction.DOWN),
            float(self.direction == Direction.LEFT),
            float(self.direction == Direction.RIGHT),
        ]
        norm_len = n / (bb * bb)

        flat = np.empty(4 * bb * bb + 5, dtype=np.float32)
        flat[:4 * bb * bb] = grid.ravel()
        flat[4 * bb * bb:] = dir_oh + [norm_len]
        return flat


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
        btn1_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 - 120, btn_w, btn_h)
        btn2_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 - 30, btn_w, btn_h)
        btn3_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 + 60, btn_w, btn_h)
        btn4_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 + 150, btn_w, btn_h)

        # -- helper: session persistence --------------------------------
        _SESSION_CFG = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'session_cfg.json')

        def _load_session_cfg():
            try:
                with open(_SESSION_CFG, 'r', encoding='utf-8') as _f:
                    return json.load(_f)
            except Exception:
                return {}

        def _save_session_cfg(**kwargs):
            try:
                _d = _load_session_cfg()
                _d.update(kwargs)
                with open(_SESSION_CFG, 'w', encoding='utf-8') as _f:
                    json.dump(_d, _f)
            except Exception:
                pass

        g.current_level_name = None
        g.current_level_path = None
        g.current_checkpoint_path = None
        _sess = _load_session_cfg()
        if _sess.get('level_path') and os.path.isfile(_sess['level_path']):
            g.current_level_path = _sess['level_path']
            g.current_level_name = os.path.basename(_sess['level_path'])
        if _sess.get('checkpoint_path') and os.path.isfile(_sess['checkpoint_path']):
            g.current_checkpoint_path = _sess['checkpoint_path']

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

        # -- visualization loop (eval only) --------------------------------
        def visualize_agent(env, max_episodes=10000, max_steps=MAX_EPISODE_MOVES,
                            init_ckpt=None):
            import torch as torch_mod

            # --- detect agent type from checkpoint -----------------------
            agent = None
            agent_type = 'dqn'
            if init_ckpt:
                try:
                    data = torch_mod.load(init_ckpt, map_location='cpu', weights_only=False)
                    if isinstance(data, dict) and 'board_size' in data:
                        agent_type = 'cnn'
                except Exception:
                    pass

            if agent_type == 'cnn':
                from cnn_agent import CNNAgent
                agent = CNNAgent(board_size=env.board_blocks)
                env.state_mode = 'grid'
            else:
                from dqn_agent import DQNAgent
                agent = DQNAgent()
                env.state_mode = 'features'

            if init_ckpt:
                try:
                    agent.load(init_ckpt)
                except Exception as e:
                    msg = f'Failed to load checkpoint: {e}'
                    info_timer = 120
                    while info_timer > 0:
                        for ev in pygame.event.get():
                            if ev.type == pygame.QUIT:
                                pygame.quit()
                                sys.exit(0)
                        env.display.fill(BLACK)
                        for i, ln in enumerate(env._wrap_text(msg, env.font, env.w - 40)):
                            env.display.blit(env.font.render(ln, True, WHITE), (10, 40 + i * 30))
                        pygame.display.flip()
                        info_timer -= 1
                        if env.clock:
                            env.clock.tick(30)
                    return

            agent.eps = 0.0
            agent.policy_net.eval()

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
                except Exception:
                    pass

            max_cells = env.board_blocks * env.board_blocks
            n_walls = len(getattr(env, 'walls', set()))
            max_score = max_cells - n_walls - 3
            action_names = {0: 'STRAIGHT', 1: 'RIGHT TURN', 2: 'LEFT TURN'}
            recent_scores = []
            recent_steps = []
            device = getattr(agent, 'device', torch_mod.device('cpu'))

            def format_q_values(q_vals):
                if not q_vals or len(q_vals) < 3:
                    return 'Q[S,R,L]: n/a'
                return f'Q[S,R,L]: {q_vals[0]:.2f}, {q_vals[1]:.2f}, {q_vals[2]:.2f}'

            def choose_action_with_debug(cur_state):
                try:
                    with torch_mod.no_grad():
                        st = torch_mod.as_tensor(
                            cur_state, dtype=torch_mod.float32
                        ).unsqueeze(0).to(device)
                        q_out = agent.policy_net(st).squeeze(0).cpu().tolist()
                    q_values = [float(v) for v in q_out]
                    best = int(max(range(len(q_values)), key=lambda i: q_values[i]))
                    return best, q_values
                except Exception:
                    return agent.act(cur_state), None

            def _wait_for_resume(env, ep, score, total_reward,
                                 recent_scores, recent_steps,
                                 last_action, last_q_values,
                                 action_names, reason_text):
                while True:
                    panel_bg, btn_pause, btn_plus, btn_minus, btn_stop = \
                        env._get_left_control_rects(panel_h=260)
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit(0)
                        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                            return False
                        if ev.type == pygame.VIDEORESIZE:
                            env.resize_window(ev.w, ev.h)
                        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                            mx, my = ev.pos
                            if btn_pause.collidepoint(mx, my):
                                return True
                            if btn_stop.collidepoint(mx, my):
                                return False
                            if btn_plus.collidepoint(mx, my):
                                env.speed += 10
                            if btn_minus.collidepoint(mx, my):
                                env.speed = max(10, env.speed - 10)
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
                            f'** {reason_text} **',
                            f'Episode: {ep}  Score: {score}/{max_score}',
                            f'Reward: {total_reward:.1f}',
                            avg_txt,
                            f'Speed: {env.speed} FPS',
                            f'Last move: {action_names.get(last_action, "n/a")}',
                            format_q_values(last_q_values),
                            '',
                            'Click Resume for next episode',
                        ]
                        info_font = env.small_font or env.font
                        line_h = info_font.get_height() + 4
                        y_off = panel_bg.y + 6
                        max_y = panel_bg.bottom - line_h
                        for ln in lines:
                            if y_off > max_y:
                                break
                            for wrapped_ln in env._wrap_text(
                                    ln, info_font, panel_bg.width - 12):
                                if y_off > max_y:
                                    break
                                s = info_font.render(wrapped_ln, True, WHITE)
                                env.display.blit(s, (panel_bg.x + 6, y_off))
                                y_off += line_h
                        pygame.draw.rect(env.display, (100, 180, 100), btn_pause)
                        pygame.draw.rect(env.display, PANEL_BORDER, btn_pause, 2)
                        env.display.blit(
                            env.font.render('Resume', True, BLACK),
                            (btn_pause.x + 8, btn_pause.y + 4))
                        for btn in (btn_plus, btn_minus):
                            pygame.draw.rect(env.display, (140, 140, 140), btn)
                            pygame.draw.rect(env.display, PANEL_BORDER, btn, 2)
                        env.display.blit(env.font.render('+', True, BLACK),
                                         (btn_plus.x + 10, btn_plus.y + 4))
                        env.display.blit(env.font.render('-', True, BLACK),
                                         (btn_minus.x + 12, btn_minus.y + 4))
                        pygame.draw.rect(env.display, (200, 80, 80), btn_stop)
                        pygame.draw.rect(env.display, PANEL_BORDER, btn_stop, 2)
                        env.display.blit(
                            env.font.render('Stop', True, BLACK),
                            (btn_stop.x + 8, btn_stop.y + 4))
                        env._draw_footer_block([
                            f'{reason_text} - Click Resume for next episode',
                            'Stop: end | Esc: menu',
                        ])
                        pygame.display.flip()
                    except Exception:
                        pass
                    if env.clock:
                        env.clock.tick(30)

            ep = 0
            running = True
            while running and ep < max_episodes:
                ep += 1
                state = env.reset()
                total_reward = 0.0
                paused = False
                panel_bg, btn_pause, btn_plus, btn_minus, btn_stop = env._get_left_control_rects(
                    panel_h=260)

                last_action = None
                last_q_values = None
                last_abs_dir = None
                ep_steps = 0
                step_info = {}
                do_step = False

                for t in range(max_steps):
                    panel_bg, btn_pause, btn_plus, btn_minus, btn_stop = \
                        env._get_left_control_rects(panel_h=260)
                    btn_step = pygame.Rect(
                        btn_stop.x, btn_stop.bottom + 6, btn_stop.width, 32)

                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit(0)
                        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                            running = False
                            break
                        if ev.type == pygame.VIDEORESIZE:
                            env.resize_window(ev.w, ev.h)
                            panel_bg, btn_pause, btn_plus, btn_minus, btn_stop = \
                                env._get_left_control_rects(panel_h=260)
                            btn_step = pygame.Rect(
                                btn_stop.x, btn_stop.bottom + 6, btn_stop.width, 32)
                            max_cells = env.board_blocks * env.board_blocks
                        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                            mx, my = ev.pos
                            if btn_pause.collidepoint(mx, my):
                                paused = not paused
                            elif btn_stop.collidepoint(mx, my):
                                running = False
                                break
                            elif btn_plus.collidepoint(mx, my):
                                env.speed += 10
                            elif btn_minus.collidepoint(mx, my):
                                env.speed = max(10, env.speed - 10)
                            elif paused and btn_step.collidepoint(mx, my):
                                do_step = True
                    if not running:
                        break

                    if not paused or do_step:
                        action, last_q_values = choose_action_with_debug(state)
                        last_action = action
                        next_state, reward, done, step_info = env.play_step(
                            action, skip_events=True)
                        total_reward += reward
                        ep_steps += 1
                        last_abs_dir = env.direction.name
                        do_step = False
                    else:
                        next_state, reward, done, step_info = state, 0, False, {}

                    state = next_state

                    # info panel
                    try:
                        env._draw_panel_box(panel_bg)
                        move_txt = last_abs_dir if last_abs_dir else 'n/a'
                        q_txt = format_q_values(last_q_values)
                        if recent_scores:
                            n = min(50, len(recent_scores))
                            avg_s = sum(recent_scores[-50:]) / n
                            avg_t = sum(recent_steps[-50:]) / n
                            avg_txt = f'Avg50: {avg_s:.2f} / {avg_t:.1f}'
                        else:
                            avg_txt = 'Avg50: n/a'

                        lines = [
                            f'Episode: {ep}  Step: {ep_steps}',
                            f'Score: {env.score}/{max_score}  Reward: {total_reward:.1f}',
                            avg_txt,
                            f'Speed: {env.speed} FPS',
                            f'Agent: {agent_type.upper()} (eval)',
                            f'Move: {move_txt}',
                            q_txt,
                        ]
                        info_font = env.small_font or env.font
                        line_h = info_font.get_height() + 4
                        y_off = panel_bg.y + 6
                        max_y = panel_bg.bottom - line_h
                        for ln in lines:
                            if y_off > max_y:
                                break
                            for wrapped_ln in env._wrap_text(ln, info_font,
                                                             panel_bg.width - 12):
                                if y_off > max_y:
                                    break
                                s = info_font.render(wrapped_ln, True, WHITE)
                                env.display.blit(s, (panel_bg.x + 6, y_off))
                                y_off += line_h

                        # buttons
                        pygame.draw.rect(env.display,
                                         (180, 180, 100) if paused else (100, 180, 100),
                                         btn_pause)
                        pygame.draw.rect(env.display, PANEL_BORDER, btn_pause, 2)
                        env.display.blit(
                            env.font.render(
                                'Pause' if not paused else 'Resume',
                                True, BLACK),
                            (btn_pause.x + 8, btn_pause.y + 4))
                        for btn in (btn_plus, btn_minus):
                            pygame.draw.rect(env.display, (140, 140, 140), btn)
                            pygame.draw.rect(env.display, PANEL_BORDER, btn, 2)
                        env.display.blit(env.font.render('+', True, BLACK),
                                         (btn_plus.x + 10, btn_plus.y + 4))
                        env.display.blit(env.font.render('-', True, BLACK),
                                         (btn_minus.x + 12, btn_minus.y + 4))
                        pygame.draw.rect(env.display, (200, 80, 80), btn_stop)
                        pygame.draw.rect(env.display, PANEL_BORDER, btn_stop, 2)
                        env.display.blit(
                            env.font.render('Stop', True, BLACK),
                            (btn_stop.x + 8, btn_stop.y + 4))
                        if paused:
                            pygame.draw.rect(
                                env.display, (80, 140, 200), btn_step)
                            pygame.draw.rect(
                                env.display, PANEL_BORDER, btn_step, 2)
                            env.display.blit(
                                env.font.render('Step', True, BLACK),
                                (btn_step.x + 8, btn_step.y + 4))

                        env._draw_footer_block([
                            'Esc: menu | Pause: toggle | Stop: end',
                            'Speed +/-: change FPS | Step: single step (when paused)',
                        ])
                        pygame.display.flip()
                    except Exception:
                        pass

                    if paused and env.clock:
                        env.clock.tick(30)

                    occupied = len(env.snake) + len(env.walls)
                    if occupied >= max_cells:
                        _wait_for_resume(env, ep, env.score, total_reward,
                                         recent_scores, recent_steps,
                                         last_action, last_q_values,
                                         action_names, 'Board filled!')
                        running = False
                        break

                    if done:
                        if step_info.get('board_filled'):
                            reason_text = 'Board filled!'
                        elif step_info.get('reason') == 'collision':
                            reason_text = 'Snake died (collision)'
                        elif step_info.get('reason') == 'max_steps':
                            reason_text = 'Max steps reached'
                        elif step_info.get('reason') == 'food_timeout':
                            reason_text = 'Stalled (food timeout)'
                        else:
                            reason_text = 'Episode ended'
                        resume = _wait_for_resume(
                            env, ep, env.score, total_reward,
                            recent_scores, recent_steps,
                            last_action, last_q_values,
                            action_names, reason_text)
                        if not resume:
                            running = False
                        break

                recent_scores.append(float(env.score))
                recent_steps.append(float(ep_steps))

            env.state_mode = 'features'
            env.reset()

        # ----- main menu loop ------------------------------------------
        _notif_msg = ''
        _notif_frames = 0
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
                                                g.h // 2 - 120, btn_w, btn_h)
                        btn2_rect = pygame.Rect(center_x - btn_w // 2,
                                                g.h // 2 - 30, btn_w, btn_h)
                        btn3_rect = pygame.Rect(center_x - btn_w // 2,
                                                g.h // 2 + 60, btn_w, btn_h)
                        btn4_rect = pygame.Rect(center_x - btn_w // 2,
                                                g.h // 2 + 150, btn_w, btn_h)
                    if (event.type == pygame.MOUSEBUTTONDOWN
                            and event.button == 1):
                        mx, my = event.pos
                        for tag, rect in [('1', btn1_rect), ('2', btn2_rect),
                                          ('3', btn3_rect), ('4', btn4_rect)]:
                            if rect.collidepoint(mx, my):
                                choice = tag
                                menu_running = False
                                break

                g.display.fill(BLACK)
                title = g.font.render('Snake AI - Menu', True, WHITE)
                g.display.blit(title, (center_x - title.get_width() // 2,
                                       g.h // 2 - 160))
                for rect, label in [
                    (btn1_rect, 'Level Design'),
                    (btn2_rect, 'Visualization'),
                    (btn3_rect, 'Load checkpoint (.pth)'),
                    (btn4_rect, 'Load level (.json)'),
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
                if _notif_frames > 0:
                    _nf = g.small_font or g.font
                    _ns = _nf.render(_notif_msg, True, WHITE)
                    _nx, _ny = 10, g.h - _ns.get_height() - 24
                    _nw, _nh = _ns.get_width() + 20, _ns.get_height() + 16
                    pygame.draw.rect(g.display, (30, 30, 30), (_nx, _ny, _nw, _nh))
                    pygame.draw.rect(g.display, (140, 140, 140), (_nx, _ny, _nw, _nh), 1)
                    g.display.blit(_ns, (_nx + 10, _ny + 8))
                    _notif_frames -= 1
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
                _notif_msg = info_msg
                _notif_frames = 90

            elif choice == '3':
                path = pick_file_dialog(
                    [('PyTorch', '*.pth'), ('All files', '*.*')],
                    fallback_exts=('.pth',), fallback_dirs=('.', 'logs'),
                    fallback_title='Select checkpoint (.pth)')
                if path:
                    g.current_checkpoint_path = path
                    _save_session_cfg(checkpoint_path=path)
                    info_msg = f'Checkpoint: {os.path.basename(path)}'
                else:
                    info_msg = 'No checkpoint selected.'
                _notif_msg = info_msg
                _notif_frames = 90

            elif choice == '4':
                path = pick_file_dialog(
                    [('JSON level', '*.json'), ('All files', '*.*')],
                    fallback_exts=('.json',), fallback_dirs=('levels', '.'),
                    fallback_title='Select level (.json)')
                if path:
                    g.current_level_path = path
                    g.current_level_name = os.path.basename(path)
                    _save_session_cfg(level_path=path)
                    info_msg = f'Level: {g.current_level_name}'
                else:
                    info_msg = 'No level selected.'
                _notif_msg = info_msg
                _notif_frames = 90

            elif choice == '2':
                sub_w = 460
                sub_h = 300
                board_size_str = str(g.board_blocks)
                bs_cursor = len(board_size_str)
                active_input = None
                blink_timer = 0
                submenu = True
                while submenu:
                    blink_timer = (blink_timer + 1) % 60
                    show_cursor = blink_timer < 30
                    sx = g.w // 2 - sub_w // 2
                    sy = g.h // 2 - sub_h // 2
                    sub_rect = pygame.Rect(sx, sy, sub_w, sub_h)
                    inp_bs_rect = pygame.Rect(sx + 20, sy + 80, 420, 32)
                    btn_start = pygame.Rect(sx + 20, sy + 200, 200, 50)
                    btn_back = pygame.Rect(sx + 240, sy + 200, 200, 50)

                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit(0)
                        if ev.type == pygame.VIDEORESIZE:
                            g.resize_window(ev.w, ev.h)
                        if (ev.type == pygame.MOUSEBUTTONDOWN
                                and ev.button == 1):
                            mx, my = ev.pos
                            if inp_bs_rect.collidepoint(mx, my):
                                active_input = 'boardsize'
                                blink_timer = 0
                            elif btn_start.collidepoint(mx, my):
                                active_input = None
                                bsize = max(5, int(board_size_str)) if board_size_str.isdigit() else g.board_blocks
                                if bsize != g.board_blocks:
                                    g.board_blocks = bsize
                                    g.layout_cfg['board_blocks'] = bsize
                                    g._recompute_layout()
                                try:
                                    visualize_agent(
                                        g,
                                        max_steps=MAX_EPISODE_MOVES,
                                        init_ckpt=g.current_checkpoint_path)
                                    info_msg = 'Visualization finished.'
                                except Exception as e:
                                    info_msg = f'Visualization failed: {e}'
                                submenu = False
                            elif btn_back.collidepoint(mx, my):
                                submenu = False
                                active_input = None
                            else:
                                active_input = None
                        if ev.type == pygame.KEYDOWN and active_input:
                            blink_timer = 0
                            if ev.key in (pygame.K_ESCAPE, pygame.K_RETURN):
                                active_input = None
                            elif ev.key == pygame.K_LEFT:
                                bs_cursor = max(0, bs_cursor - 1)
                            elif ev.key == pygame.K_RIGHT:
                                bs_cursor = min(len(board_size_str), bs_cursor + 1)
                            elif ev.key == pygame.K_HOME:
                                bs_cursor = 0
                            elif ev.key == pygame.K_END:
                                bs_cursor = len(board_size_str)
                            elif ev.key == pygame.K_BACKSPACE:
                                if bs_cursor > 0:
                                    board_size_str = board_size_str[:bs_cursor-1] + board_size_str[bs_cursor:]
                                    bs_cursor -= 1
                            elif ev.key == pygame.K_DELETE:
                                if bs_cursor < len(board_size_str):
                                    board_size_str = board_size_str[:bs_cursor] + board_size_str[bs_cursor+1:]
                            else:
                                ch = ev.unicode
                                if ch and ch.isdigit():
                                    board_size_str = board_size_str[:bs_cursor] + ch + board_size_str[bs_cursor:]
                                    bs_cursor += 1

                    status_font = g.small_font or g.font
                    g.display.fill((20, 20, 20))
                    pygame.draw.rect(g.display, (60, 60, 60), sub_rect)
                    title = g.font.render('Visualization', True, WHITE)
                    g.display.blit(title,
                                   (sx + sub_w // 2 - title.get_width() // 2,
                                    sy + 8))

                    # Board size input
                    bs_label = status_font.render('Board size:', True, WHITE)
                    g.display.blit(bs_label, (sx + 20, sy + 60))
                    bs_border = (255, 220, 50) if active_input == 'boardsize' else (150, 150, 150)
                    pygame.draw.rect(g.display, (40, 40, 40), inp_bs_rect)
                    pygame.draw.rect(g.display, bs_border, inp_bs_rect, 2)
                    bs_txt = status_font.render(board_size_str, True, WHITE)
                    g.display.blit(bs_txt, (inp_bs_rect.x + 6, inp_bs_rect.y + 6))
                    if active_input == 'boardsize' and show_cursor:
                        _cx = inp_bs_rect.x + 6 + status_font.size(board_size_str[:bs_cursor])[0]
                        pygame.draw.line(g.display, WHITE,
                                         (_cx, inp_bs_rect.y + 4),
                                         (_cx, inp_bs_rect.bottom - 4))

                    # Status lines
                    level_status = (f'Level: {g.current_level_name}'
                                    if g.current_level_name
                                    else 'Level: empty map (classic snake)')
                    ckpt_status = (f'Checkpoint: {os.path.basename(g.current_checkpoint_path)}'
                                   if g.current_checkpoint_path else 'No checkpoint loaded')
                    info_y = sy + 130
                    for ln in [level_status, ckpt_status]:
                        cl = g._fit_text(ln, status_font, sub_w - 24)
                        sf = status_font.render(cl, True, WHITE)
                        g.display.blit(sf, (sx + 12, info_y))
                        info_y += status_font.get_height() + 4

                    pygame.draw.rect(g.display, (80, 200, 120), btn_start)
                    pygame.draw.rect(g.display, (200, 80, 80), btn_back)
                    g.display.blit(g.font.render('Start', True, BLACK),
                                   (btn_start.x + 62, btn_start.y + 12))
                    g.display.blit(g.font.render('Back', True, BLACK),
                                   (btn_back.x + 70, btn_back.y + 14))
                    pygame.display.flip()
                    if g.clock:
                        g.clock.tick(30)

                if 'info_msg' in locals() and info_msg:
                    _notif_msg = info_msg
                    _notif_frames = 90
