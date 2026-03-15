"""Snake game model (MVC: Model layer).

This module contains the environment state, transition dynamics, observations,
and reward shaping used by training and evaluation scripts.
"""

from collections import deque
from enum import Enum
import random

import numpy as np
import pygame

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


# UI palette used by renderer helpers.
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
    inside a logical grid of size board_blocks x board_blocks. Pixel coordinates
    are computed only in the view layer.
    """

    def __init__(
            self,
            w=640,
            h=480,
            render=True,
            seed=None,
            speed=DEFAULT_SPEED,
            max_episode_steps=MAX_EPISODE_MOVES,
            board_blocks=BOARD_BLOCKS,
            state_mode='features',
            simple_rewards=False,
            reward_mode='complex',
            reward_switch_start=0.25,
            reward_switch_end=0.35):
        self.w = max(w, MIN_WINDOW_W)
        self.h = max(h, MIN_WINDOW_H)
        self.render = render
        self.speed = speed
        self.max_episode_steps = max(1, int(max_episode_steps))
        self.state_mode = state_mode      # 'features' or 'grid'
        self.simple_rewards = simple_rewards
        self.reward_mode = str(reward_mode).lower()
        if self.simple_rewards:
            self.reward_mode = 'simple'
        if self.reward_mode not in ('simple', 'complex', 'blend'):
            self.reward_mode = 'complex'
        rs = max(0.0, min(0.99, float(reward_switch_start)))
        re = max(rs + 1e-6, min(1.0, float(reward_switch_end)))
        self.reward_switch_start = rs
        self.reward_switch_end = re
        if seed is not None:
            random.seed(seed)

        if self.render:
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h), pygame.RESIZABLE)
            pygame.display.set_caption('Snake AI - Project')
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
            'white': WHITE,
            'red': RED,
            'blue1': BLUE1,
            'blue2': BLUE2,
            'black': BLACK,
            'panel_bg': PANEL_BG,
            'panel_border': PANEL_BORDER,
            'board_border': BOARD_BORDER,
            'footer_bg': FOOTER_BG,
            'footer_border': FOOTER_BORDER,
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

    # -- Layout delegates (View support) -------------------------------
    def _recompute_layout(self):
        _layout_recompute_layout(self)

    def resize_window(self, w, h):
        _layout_resize_window(self, w, h)

    def _get_left_control_rects(self, panel_h=260):
        return _layout_get_left_control_rects(self, panel_h=panel_h)

    # -- Core environment state ----------------------------------------
    def reset(self):
        self.direction = Direction.RIGHT
        center = self.board_blocks // 2
        self.head = (center, center)
        self.snake = [self.head, (center - 1, center), (center - 2, center)]

        # Stable segment IDs: each grown segment receives an immutable ID.
        self.snake_segment_ids = [1, 2, 3]
        self._next_segment_id = 4

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
        """Return an observation vector matching the selected state_mode."""
        if self.state_mode == 'grid':
            return self.get_grid_state()
        return self.get_state()

    def _manhattan_to_food(self):
        if self.food is None:
            return 0
        return abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])

    def _occupancy_ratio(self):
        total = self.board_blocks * self.board_blocks
        if total <= 0:
            return 0.0
        return len(self.snake) / float(total)

    def _reward_blend_weight(self, occupancy_ratio):
        """Compute blend weight for simple/complex reward interpolation."""
        if self.reward_mode == 'simple':
            return 0.0
        if self.reward_mode == 'complex':
            return 1.0
        if occupancy_ratio <= self.reward_switch_start:
            return 0.0
        if occupancy_ratio >= self.reward_switch_end:
            return 1.0
        span = self.reward_switch_end - self.reward_switch_start
        return (occupancy_ratio - self.reward_switch_start) / span

    def _place_food(self):
        """Place food on any free cell that is not occupied by snake or walls."""
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

    # -- Transition function (MDP step) --------------------------------
    def play_step(self, action, skip_events=False):
        self.frame_iteration += 1

        if self.render and not skip_events:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return self._get_obs(), 0, True, {'quit': True}

        if self.no_food_slots:
            return self._get_obs(), 0, True, {
                'score': self.score,
                'board_filled': True,
                'reason': 'no_food_slots',
            }

        act_idx = action if isinstance(action, int) and 0 <= action <= 2 else self._parse_action(action)
        self._move(act_idx)

        # Incremental body set update: old head becomes body after movement.
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
            return self._get_obs(), reward, done, {'score': self.score, 'reason': 'collision'}

        if self.frame_iteration >= self.max_episode_steps:
            done = True
            reward = -10
            return self._get_obs(), reward, done, {'score': self.score, 'reason': 'max_steps'}

        # Episode timeout if no food is eaten for too long.
        food_timeout = 4 * self.board_blocks * self.board_blocks
        if self._steps_since_food > food_timeout:
            done = True
            reward = -10
            return self._get_obs(), reward, done, {'score': self.score, 'reason': 'food_timeout'}

        # Distance shaping reference value.
        new_food_dist = abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])
        simple_reward = -0.01
        complex_reward = -0.01

        if self.head == self.food:
            self._just_ate = True
            self.score += 1
            self._steps_since_food = 0

            # Growth adds one tail segment with a permanent ID.
            if len(self.snake_segment_ids) < len(self.snake):
                self.snake_segment_ids.append(self._next_segment_id)
                self._next_segment_id += 1

            self._place_food()
            self._prev_food_dist = abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])
            simple_reward = 10.0

            # Complex reward: post-eat free-space safety estimate.
            post_eat_ratio = self._flood_fill_ratio()
            if post_eat_ratio < 0.10:
                complex_reward = -5.0
            elif post_eat_ratio < 0.15:
                complex_reward = 2.0
            elif post_eat_ratio < 0.3:
                complex_reward = 5.0
            elif post_eat_ratio < 0.5:
                complex_reward = 8.0
            else:
                complex_reward = 10.0
        else:
            tail = self.snake.pop()
            self.snake_body_set.discard(tail)

            # Potential-based shaping on Manhattan distance to food.
            if new_food_dist < self._prev_food_dist:
                complex_reward += 0.03
            elif new_food_dist > self._prev_food_dist:
                complex_reward -= 0.03
            self._prev_food_dist = new_food_dist

        occupancy = self._occupancy_ratio()
        blend_weight = self._reward_blend_weight(occupancy)
        simple_weight = 1.0 - blend_weight
        complex_active = (self.reward_mode == 'complex') or (
            self.reward_mode == 'blend' and blend_weight > 0.0
        )

        # Flood fill is one of the heaviest operations per step.
        if self.state_mode == 'features' or complex_active:
            reachable_ratio = self._flood_fill_ratio()
            self._cached_flood = reachable_ratio
        else:
            reachable_ratio = self._cached_flood if self._cached_flood is not None else 1.0

        if complex_active:
            # Space awareness penalty: avoid entering tight regions.
            if reachable_ratio < 0.2:
                complex_reward -= 0.3
            elif reachable_ratio < 0.4:
                complex_reward -= 0.1

            # Hunger penalty (complex branch).
            hunger_threshold = self.board_blocks * 2 + len(self.snake)
            if self._steps_since_food > hunger_threshold:
                hunger_ratio = self._steps_since_food / food_timeout
                complex_reward -= 0.1 * hunger_ratio

            # Anti-loop penalty to discourage short cyclic trajectories.
            if self.head in self._recent_set:
                loop_penalty = 0.3 + 0.2 * (len(self.snake) / (self.board_blocks ** 2))
                complex_reward -= loop_penalty

            self._recent_positions.append(self.head)
            self._recent_set = set(self._recent_positions)

        # Hunger penalty (simple branch).
        if self.reward_mode in ('simple', 'blend'):
            hunger_threshold = self.board_blocks + len(self.snake)
            if self._steps_since_food > hunger_threshold:
                hunger_ratio = self._steps_since_food / food_timeout
                simple_reward -= 0.3 * hunger_ratio

        reward = simple_weight * simple_reward + blend_weight * complex_reward

        if self.render:
            self._update_ui()
            if self.clock:
                self.clock.tick(self.speed)

        return self._get_obs(), reward, done, {
            'score': self.score,
            'reason': 'running',
            'reward_blend': blend_weight,
            'occupancy': occupancy,
        }

    # -- Render delegates (View API) -----------------------------------
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

    # -- Movement / action parsing -------------------------------------
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

    def get_safe_action_mask(self):
        """Return [straight, right, left] mask for immediate-death filtering.

        True means the action is safe for a one-step lookahead. If all actions
        are unsafe (rare edge case), all actions are re-enabled as fallback.
        """
        idx = _CW_IDX[self.direction]
        tail_tip = None
        if not self._just_ate and len(self.snake) > 1:
            tail_tip = self.snake[-1]

        mask = [True, True, True]
        for action in (0, 1, 2):
            nidx = idx
            if action == 1:
                nidx = (nidx + 1) & 3
            elif action == 2:
                nidx = (nidx - 1) & 3
            ndir = _CW[nidx]
            dx, dy = DIR_VECTORS[ndir]
            nx = self.head[0] + dx
            ny = self.head[1] + dy
            mask[action] = not self._is_blocked_excluding(nx, ny, tail_tip)

        if not any(mask):
            return [True, True, True]
        return mask

    # -- Feature observations ------------------------------------------
    def get_state(self):
        """Return 26 hand-crafted features for the tabular/MLP DQN policy."""
        hx, hy = self.head
        bb = self.board_blocks
        inv_bb = 1.0 / bb

        idx = _CW_IDX[self.direction]
        dir_s = _CW[idx]
        dir_r = _CW[(idx + 1) & 3]
        dir_l = _CW[(idx - 1) & 3]
        dir_b = _CW[(idx + 2) & 3]

        # Tail tip is passable only when the snake did not just eat.
        tail_tip = None
        if not self._just_ate and len(self.snake) > 1:
            tail_tip = self.snake[-1]

        dxs, dys = DIR_VECTORS[dir_s]
        dxr, dyr = DIR_VECTORS[dir_r]
        dxl, dyl = DIR_VECTORS[dir_l]
        danger_s = int(self._is_blocked_excluding(hx + dxs, hy + dys, tail_tip))
        danger_r = int(self._is_blocked_excluding(hx + dxr, hy + dyr, tail_tip))
        danger_l = int(self._is_blocked_excluding(hx + dxl, hy + dyl, tail_tip))

        # Normalized food offset in [-1, 1] approx.
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
                cx += dx
                cy += dy
                dist += 1
            return dist * inv_bb

        def ray_body(d):
            dx, dy = DIR_VECTORS[d]
            cx, cy = hx + dx, hy + dy
            dist = 1
            while 0 <= cx < bb and 0 <= cy < bb:
                if (cx, cy) in body_set and (cx, cy) != tail_tip:
                    return dist * inv_bb
                cx += dx
                cy += dy
                dist += 1
            return 1.0

        flood_ratio = getattr(self, '_cached_flood', None)
        if flood_ratio is None:
            flood_ratio = self._flood_fill_ratio()

        # Simulate the board as if food cell were additionally blocked (future body growth).
        food_safety = self._flood_fill_ratio(extra_block=self.food)

        tail = self.snake[-1]
        tail_dx = (tail[0] - hx) * inv_bb
        tail_dy = (tail[1] - hy) * inv_bb
        tail_dist = (abs(tail[0] - hx) + abs(tail[1] - hy)) * inv_bb

        # One-step reachable-space estimates for each action.
        action_flood_s = self._action_flood_fill(0)
        action_flood_r = self._action_flood_fill(1)
        action_flood_l = self._action_flood_fill(2)

        return [
            danger_s,
            danger_r,
            danger_l,
            int(self.direction == Direction.UP),
            int(self.direction == Direction.DOWN),
            int(self.direction == Direction.LEFT),
            int(self.direction == Direction.RIGHT),
            food_dx,
            food_dy,
            ray_wall(dir_s),
            ray_wall(dir_r),
            ray_wall(dir_l),
            ray_wall(dir_b),
            ray_body(dir_s),
            ray_body(dir_r),
            ray_body(dir_l),
            ray_body(dir_b),
            len(self.snake) / (bb * bb),
            flood_ratio,
            food_safety,
            tail_dx,
            tail_dy,
            tail_dist,
            action_flood_s,
            action_flood_r,
            action_flood_l,
        ]

    def _is_blocked_excluding(self, cx, cy, exclude=None):
        """Collision test that treats one cell (typically moving tail) as passable."""
        if cx < 0 or cx >= self.board_blocks or cy < 0 or cy >= self.board_blocks:
            return True
        if (cx, cy) in self.snake_body_set and (cx, cy) != exclude:
            return True
        if self.walls and (cx, cy) in self.walls:
            return True
        return False

    def _flood_fill_ratio(self, extra_block=None):
        """Run BFS from the head and return reachable/free cell ratio."""
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
        """Estimate reachable-space ratio after simulating a one-step action."""
        idx = _CW_IDX[self.direction]
        if action_idx == 1:
            idx = (idx + 1) & 3
        elif action_idx == 2:
            idx = (idx - 1) & 3
        d = _CW[idx]
        dx, dy = DIR_VECTORS[d]
        nx, ny = self.head[0] + dx, self.head[1] + dy
        bb = self.board_blocks

        if nx < 0 or nx >= bb or ny < 0 or ny >= bb:
            return 0.0
        if self.walls and (nx, ny) in self.walls:
            return 0.0

        tail_tip = self.snake[-1] if len(self.snake) > 1 else None
        if (nx, ny) in self.snake_body_set and (nx, ny) != tail_tip:
            return 0.0

        # BFS after transition assumptions: old head becomes body, tail moves.
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
                if (ax, ay) == head:
                    continue
                if (ax, ay) in body_set and (ax, ay) != tail_tip:
                    continue
                if walls and (ax, ay) in walls:
                    continue
                visited.add((ax, ay))
                queue.append((ax, ay))

        blocked_body = len(body_set) + 1
        if tail_tip is not None:
            blocked_body -= 1
        total_free = bb * bb - blocked_body - (len(walls) if walls else 0)
        if total_free <= 0:
            return 0.0
        return len(visited) / total_free

    # -- Grid observation for CNN --------------------------------------
    def get_grid_state(self):
        """Return flattened CNN input: 4 channels + 5 auxiliary features.

        Channels:
          0: body_age (newer segments have larger values)
          1: head (binary)
          2: food (binary)
          3: walls (binary)
        Auxiliary features:
          direction one-hot (4), normalized snake length (1)
        """
        bb = self.board_blocks
        grid = np.zeros((4, bb, bb), dtype=np.float32)

        # Channel 0: body age (head excluded, rendered in its own channel).
        n = len(self.snake)
        if n > 1:
            for i in range(1, n):
                cell = self.snake[i]
                grid[0, cell[1], cell[0]] = 1.0 - (i / n)

        # Channel 1: head position.
        hx, hy = self.head
        if 0 <= hx < bb and 0 <= hy < bb:
            grid[1, hy, hx] = 1.0

        # Channel 2: food.
        if self.food:
            grid[2, self.food[1], self.food[0]] = 1.0

        # Channel 3: static walls.
        if self.walls:
            for cell in self.walls:
                grid[3, cell[1], cell[0]] = 1.0

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


__all__ = [
    'Direction',
    'SnakeGameAI',
    'DIR_VECTORS',
    'WHITE',
    'RED',
    'BLUE1',
    'BLUE2',
    'BLACK',
    'PANEL_BG',
    'PANEL_BORDER',
    'BOARD_BORDER',
    'FOOTER_BG',
    'FOOTER_BORDER',
    'BLOCK_SIZE',
    'DEFAULT_SPEED',
    'MAX_EPISODE_MOVES',
    'BOARD_BLOCKS',
    'MIN_WINDOW_W',
    'MIN_WINDOW_H',
    'LEFT_PANEL_W',
    'UI_MARGIN',
    'FOOTER_H',
    'PANEL_MIN_W',
    'PANEL_MAX_W',
    'MIN_BLOCK_PIXELS',
]
