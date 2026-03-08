import pygame
import random
import os
import sys
import json
import subprocess
from enum import Enum
from collections import namedtuple
from game_layout import (
    point_to_cell as _layout_point_to_cell,
    cell_to_point as _layout_cell_to_point,
    remap_entities_after_layout_change as _layout_remap_entities_after_layout_change,
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
try:
    import numpy as _np
except Exception:
    _np = None

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Kolory używane w grze
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
DEFAULT_SPEED = 30 # Możesz to zmieniać, żeby przyspieszyć lub zwolnić symulację
MAX_EPISODE_MOVES = 15000

# Board and window policy
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
    def __init__(self, w=640, h=480, render=True, seed=None, speed=DEFAULT_SPEED, max_episode_steps=MAX_EPISODE_MOVES):
        # enforce minimum window size
        self.w = max(w, MIN_WINDOW_W)
        self.h = max(h, MIN_WINDOW_H)
        self.render = render
        self.speed = speed
        # Hard cap for steps in a single episode.
        self.max_episode_steps = max(1, int(max_episode_steps))
        if seed is not None:
            random.seed(seed)

        # Inicjalizacja okna gry tylko jeśli renderujemy
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

        self.point_factory = Point
        self.layout_cfg = {
            'board_blocks': BOARD_BLOCKS,
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

        # board layout parameters (logical blocks)
        self.board_blocks = self.layout_cfg['board_blocks']
        self.left_panel_width = self.layout_cfg['left_panel_w']
        self.footer_height = self.layout_cfg['footer_h']
        self.left_panel_rect = pygame.Rect(0, 0, self.left_panel_width, self.h)
        self.footer_rect = pygame.Rect(0, self.h - self.footer_height, self.w, self.footer_height)
        self.bs = BLOCK_SIZE
        self.board_w = self.bs * self.board_blocks
        self.board_h = self.bs * self.board_blocks
        self.board_x = 0
        self.board_y = 0

        # centralized layout setup
        self._recompute_layout(preserve_state=False)

        self.reset()

    def _point_to_cell(self, pt, board_x=None, board_y=None, block_size=None):
        return _layout_point_to_cell(self, pt, board_x=board_x, board_y=board_y, block_size=block_size)

    def _cell_to_point(self, cx, cy):
        return _layout_cell_to_point(self, cx, cy)

    def _remap_entities_after_layout_change(self, old_board_x, old_board_y, old_bs):
        return _layout_remap_entities_after_layout_change(self, old_board_x, old_board_y, old_bs)

    def _recompute_layout(self, preserve_state=False):
        _layout_recompute_layout(self, preserve_state=preserve_state)
        globals()['BLOCK_SIZE'] = self.bs

    def resize_window(self, w, h, preserve_state=True):
        return _layout_resize_window(self, w, h, preserve_state=preserve_state)

    def _get_left_control_rects(self, panel_h=260):
        return _layout_get_left_control_rects(self, panel_h=panel_h)
        
    def reset(self):
        # Stan początkowy gry (reset po śmierci węża)
        self.direction = Direction.RIGHT
        # Set head in center of logical board (in pixel coords)
        center_cell = self.board_blocks // 2
        cx = self.board_x + center_cell * self.bs
        cy = self.board_y + center_cell * self.bs
        self.head = Point(int(cx), int(cy))
        self.snake = [self.head,
                  Point(self.head.x - self.bs, self.head.y),
                  Point(self.head.x - 2 * self.bs, self.head.y)]
        
        self.score = 0
        self.food = None
        self.no_food_slots = (not self._place_food())
        self.frame_iteration = 0
        # zwróć obserwację (przydatne dla RL)
        return self.get_state()
        
    def _place_food(self):
        # Random free cell inside logical board (free = not snake, not wall).
        occupied = set(self.snake)
        if hasattr(self, 'walls') and self.walls:
            occupied.update(self.walls)

        for _ in range(1000):
            cx = random.randint(0, self.board_blocks - 1)
            cy = random.randint(0, self.board_blocks - 1)
            x = self.board_x + cx * self.bs
            y = self.board_y + cy * self.bs
            p = Point(int(x), int(y))
            if p not in occupied:
                self.food = p
                return True

        # Deterministic fallback: scan all board cells and pick first free.
        for cy in range(self.board_blocks):
            for cx in range(self.board_blocks):
                x = self.board_x + cx * self.bs
                y = self.board_y + cy * self.bs
                p = Point(int(x), int(y))
                if p not in occupied:
                    self.food = p
                    return True

        # No free cells left (board filled by snake/walls).
        self.food = self.head
        return False

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Collision with board borders (board glued to right)
        if pt.x > self.board_x + self.board_w - self.bs or pt.x < self.board_x or pt.y > self.board_y + self.board_h - self.bs or pt.y < self.board_y:
            return True
        # Uderzenie w ogon
        if pt in self.snake[1:]:
            return True
        # Uderzenie w bloki ścian
        if hasattr(self, 'walls') and self.walls:
            if pt in self.walls:
                return True
        return False
            
    def play_step(self, action, skip_events=False):
        """
        Wykonuje krok symulacji.
        Zwraca: (observation, reward, done, info)
        action może być:
         - int: 0=straight, 1=right, 2=left
         - lista/tuple długości 3: [1,0,0] itp. (dla kompatybilności)
        """
        self.frame_iteration += 1

        # 1. Obsługa eventów tylko jeśli renderujemy i nie pominięto eventów
        if self.render and not skip_events:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return self.get_state(), 0, True, {"quit": True}

        # No free slot for any next fruit -> terminal episode.
        if getattr(self, 'no_food_slots', False):
            return self.get_state(), 0, True, {"score": self.score, "board_filled": True, "reason": "no_food_slots"}

        # 2. Wykonaj ruch bazując na akcji od AI
        act_idx = self._parse_action(action)
        self._move(act_idx)  # aktualizuje self.head
        self.snake.insert(0, self.head)

        # 3. Sprawdź zakończenie i nagrody
        reward = 0
        # mała kara za krok, by zapobiec bezcelowemu krążeniu
        reward -= 0.01
        done = False

        if self.is_collision():
            done = True
            reward = -10
            return self.get_state(), reward, done, {"score": self.score, "reason": "collision"}

        if self.frame_iteration >= self.max_episode_steps:
            done = True
            reward = -10
            return self.get_state(), reward, done, {"score": self.score, "reason": "max_steps"}

        # 4. Jedzenie
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.no_food_slots = (not self._place_food())
            if self.no_food_slots:
                return self.get_state(), reward, True, {"score": self.score, "board_filled": True, "reason": "board_filled"}
        else:
            self.snake.pop()

        # 5. Render i zegar
        if self.render:
            self._update_ui()
            if self.clock:
                self.clock.tick(self.speed)

        return self.get_state(), reward, done, {"score": self.score, "reason": "running"}
        
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
        """Renderuje ekran. Jeśli mode=='rgb_array' zwraca numpy array (H,W,3) gdy numpy dostępne."""
        if not self.render:
            return None
        if mode == 'human':
            # już wyrenderowane w _update_ui
            return None
        if mode == 'rgb_array':
            if _np is None:
                raise RuntimeError('numpy required for rgb_array mode')
            arr = pygame.surfarray.array3d(self.display)
            # Konwertuj do (H,W,3)
            return _np.transpose(arr, (1, 0, 2))

    def level_designer(self):
        return _designer_level_designer(self)

    def _move(self, action):
        # action: 0=straight,1=right,2=left
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action == 0:
            new_dir = clock_wise[idx]
        elif action == 1:
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # action == 2
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = int(self.head.x)
        y = int(self.head.y)
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(int(x), int(y))

    def _move_point(self, pt, direction):
        x, y = pt.x, pt.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        return Point(int(x), int(y))

    def _parse_action(self, action):
        # Akceptuje int 0/1/2 lub jednowymiarową listę/tuple [1,0,0]
        if isinstance(action, int):
            if action in (0, 1, 2):
                return action
            raise ValueError('action int must be 0 (straight),1 (right) or 2 (left)')
        if isinstance(action, (list, tuple)):
            if len(action) == 3:
                if action[0] == 1:
                    return 0
                if action[1] == 1:
                    return 1
                return 2
        # Fallback: try to interpret numpy array-like
        try:
            if _np is not None and isinstance(action, _np.ndarray):
                a = action.tolist()
                return self._parse_action(a)
        except Exception:
            pass
        raise ValueError('Unsupported action format: %r' % (action,))

    def get_state(self):
        """Zwraca wektor cech (feature vector) użyteczny dla agenta RL.
        Format (float list): [danger_straight, danger_right, danger_left,
        dir_up, dir_down, dir_left, dir_right, food_dx, food_dy]
        """
        head = self.head
        point_l = None
        point_r = None
        point_s = None

        # Kierunki względem aktualnego kierunku
        if self.direction == Direction.RIGHT:
            point_s = self._move_point(head, Direction.RIGHT)
            point_r = self._move_point(head, Direction.DOWN)
            point_l = self._move_point(head, Direction.UP)
        elif self.direction == Direction.LEFT:
            point_s = self._move_point(head, Direction.LEFT)
            point_r = self._move_point(head, Direction.UP)
            point_l = self._move_point(head, Direction.DOWN)
        elif self.direction == Direction.UP:
            point_s = self._move_point(head, Direction.UP)
            point_r = self._move_point(head, Direction.RIGHT)
            point_l = self._move_point(head, Direction.LEFT)
        else:  # DOWN
            point_s = self._move_point(head, Direction.DOWN)
            point_r = self._move_point(head, Direction.LEFT)
            point_l = self._move_point(head, Direction.RIGHT)

        danger_s = int(self.is_collision(point_s))
        danger_r = int(self.is_collision(point_r))
        danger_l = int(self.is_collision(point_l))

        dir_up = int(self.direction == Direction.UP)
        dir_down = int(self.direction == Direction.DOWN)
        dir_left = int(self.direction == Direction.LEFT)
        dir_right = int(self.direction == Direction.RIGHT)

        # wektor do jedzenia (znormalizowany względem realnego pola planszy)
        board_w = max(1, self.board_w)
        board_h = max(1, self.board_h)
        food_dx = (self.food.x - head.x) / board_w
        food_dy = (self.food.y - head.y) / board_h

        state = [danger_s, danger_r, danger_l,
                 dir_up, dir_down, dir_left, dir_right,
                 food_dx, food_dy]
        return state


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run SnakeGameAI example')
    parser.add_argument('--no-render', action='store_true', help='Run without opening a window')
    args = parser.parse_args()

    import sys

    g = SnakeGameAI(render=not args.no_render)
    print('Starting game (render=%s)...' % (g.render,))
    if not g.render:
        # headless: run simple random simulation
        try:
            while True:
                action = random.randint(0, 2)
                state, reward, done, info = g.play_step(action)
                if done:
                    print('Game over. Score:', g.score)
                    break
        finally:
            pass
    else:
        # Main menu loop (keep returning to menu until user closes)
        btn_w = 360
        btn_h = 64
        center_x = g.w // 2
        btn1_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 - 150, btn_w, btn_h)
        btn2_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 - 60, btn_w, btn_h)
        btn3_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 + 30, btn_w, btn_h)
        btn4_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 + 120, btn_w, btn_h)
        btn5_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 + 210, btn_w, btn_h)

        # stateful selections stored on game instance
        g.current_level_name = getattr(g, 'current_level_name', None)
        g.current_level_path = getattr(g, 'current_level_path', None)
        g.current_checkpoint_path = getattr(g, 'current_checkpoint_path', None)

        def pick_file_dialog(
            filetypes,
            fallback_exts=('.pth',),
            fallback_dirs=('.', 'logs'),
            fallback_title='Select file - Esc to cancel'
        ):
            # Try native tkinter file dialog first, fallback to simple pygame picker
            try:
                import tkinter as _tk
                from tkinter import filedialog
                root = _tk.Tk()
                root.withdraw()
                path = filedialog.askopenfilename(filetypes=filetypes)
                root.destroy()
                return path
            except Exception:
                # Fallback picker for environments without tkinter.
                def _pygame_file_picker(
                    exts=('.pth',),
                    search_dirs=('.', 'logs'),
                    title='Select file - Esc to cancel'
                ):
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
                                # list area starting y=50
                                for i, fp in enumerate(files):
                                    rect = pygame.Rect(20, 50 + i*30, screen.get_width() - 40, 28)
                                    if rect.collidepoint(mx, my):
                                        selected = fp
                                        running = False
                                        break
                        screen.fill((30, 30, 30))
                        title_s = font_small.render(title, True, (255,255,255))
                        screen.blit(title_s, (20, 10))
                        for i, fp in enumerate(files):
                            y = 50 + i*30
                            color = (200,200,200) if i % 2 == 0 else (170,170,170)
                            pygame.draw.rect(screen, color, (18, y, screen.get_width()-36, 28))
                            short = os.path.basename(fp)
                            s = font_small.render(short, True, (0,0,0))
                            screen.blit(s, (25, y+6))
                        pygame.display.flip()
                        pygame.time.wait(30)
                    if created_temp:
                        pygame.display.quit()
                    return selected

                return _pygame_file_picker(
                    exts=fallback_exts,
                    search_dirs=fallback_dirs,
                    title=fallback_title
                )

        def detect_torch_backend():
            info = {
                'installed': False,
                'version': 'not installed',
                'cuda_build': 'none',
                'cuda_available': False,
                'device_name': 'CPU',
                'error': '',
            }
            try:
                import torch as _torch
                info['installed'] = True
                info['version'] = getattr(_torch, '__version__', 'unknown')
                info['cuda_build'] = getattr(_torch.version, 'cuda', None) or 'cpu-only'
                info['cuda_available'] = bool(_torch.cuda.is_available())
                if info['cuda_available']:
                    try:
                        info['device_name'] = _torch.cuda.get_device_name(0)
                    except Exception:
                        info['device_name'] = 'CUDA device'
            except Exception as e:
                info['error'] = str(e)
            return info

        def live_train(env, max_episodes=10000, max_steps=MAX_EPISODE_MOVES, init_ckpt=None, eval_only=False):
            """Run live training inside the game loop, rendering every step.
            Stops when user presses Esc/Quit or when an episode fills the board.
            If eval_only is True, no learning updates are applied.
            """
            # lazy imports
            AgentDQN = None
            try:
                from dqn_agent import DQNAgent as AgentDQN
            except Exception:
                AgentDQN = None

            if AgentDQN is None:
                # inform user
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

            # load level into env if provided
            if getattr(env, 'current_level_path', None):
                try:
                    with open(env.current_level_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    # data may be stored as cell coords [cx,cy]; convert to pixel coords
                    conv = set()
                    for item in data:
                        try:
                            cx, cy = int(item[0]), int(item[1])
                            px = env.board_x + cx * env.bs
                            py = env.board_y + cy * env.bs
                            conv.add(Point(int(px), int(py)))
                        except Exception:
                            pass
                    env.walls = conv
                    env.level_name = os.path.splitext(os.path.basename(env.current_level_path))[0]
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
            # optional init checkpoint
            if init_ckpt:
                try:
                    agent.load(init_ckpt)
                except Exception:
                    pass

            # Optional evaluation mode: disable exploration and weight updates.
            if eval_only:
                try:
                    if hasattr(agent, 'eps'):
                        agent.eps = 0.0
                except Exception:
                    pass
                try:
                    agent.policy_net.eval()
                except Exception:
                    pass

            max_cells = (env.board_blocks) * (env.board_blocks)
            action_names = {0: 'STRAIGHT', 1: 'RIGHT TURN', 2: 'LEFT TURN'}
            recent_scores = []
            recent_steps = []
            torch_mod = None
            try:
                import torch as torch_mod
            except Exception:
                torch_mod = None

            def format_q_values(q_values):
                if not q_values or len(q_values) < 3:
                    return 'Q[S,R,L]: n/a'
                return f'Q[S,R,L]: {q_values[0]:.2f}, {q_values[1]:.2f}, {q_values[2]:.2f}'

            def choose_action_with_debug(cur_state):
                eps_value = float(getattr(agent, 'eps', 0.0))
                explore = random.random() < eps_value

                # mirror act() side effect to keep diagnostics consistent
                if hasattr(agent, 'steps'):
                    agent.steps += 1

                if explore:
                    return random.randrange(agent.n_actions), 'explore', eps_value, None, None

                    try:
                        if torch_mod is not None:
                            with torch_mod.no_grad():
                                state_t = torch_mod.as_tensor(
                                    cur_state,
                                    dtype=torch_mod.float32,
                                    device=agent.device
                                ).unsqueeze(0)
                                q_out = agent.policy_net(state_t).squeeze(0).detach().cpu().tolist()
                            q_values = [float(v) for v in q_out]
                            action_idx = int(max(range(len(q_values)), key=lambda i: q_values[i]))
                            return action_idx, 'greedy', eps_value, q_values, None
                    except Exception:
                        pass

                    # Fallback when torch diagnostics are unavailable
                    action_idx = agent.act(cur_state)
                    return action_idx, 'agent_fallback', eps_value, None, None

            pause_requested = False
            ep = 0
            running = True
            while running and ep < max_episodes:
                ep += 1
                state = env.reset()
                total_reward = 0.0
                # control state
                paused = False
                panel_bg, btn_pause, btn_plus, btn_minus = env._get_left_control_rects(panel_h=260)

                last_action = None
                last_mode = 'n/a'
                last_eps = float(getattr(agent, 'eps', 0.0))
                last_q_values = None
                last_state_idx = None
                ep_steps = 0
                step_info = {}

                for t in range(max_steps):
                    panel_bg, btn_pause, btn_plus, btn_minus = env._get_left_control_rects(panel_h=260)

                    # handle events (allow Esc to quit live training)
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit(0)
                        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                            running = False
                            break
                        if ev.type == pygame.VIDEORESIZE:
                            env.resize_window(ev.w, ev.h, preserve_state=True)
                            panel_bg, btn_pause, btn_plus, btn_minus = env._get_left_control_rects(panel_h=260)
                            max_cells = (env.board_blocks) * (env.board_blocks)
                        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                            mx, my = ev.pos
                            # pause requested: take effect after current episode finishes
                            if btn_pause.collidepoint(mx, my):
                                if paused:
                                    # resume immediately if currently paused
                                    paused = False
                                else:
                                    pause_requested = True
                            if btn_plus.collidepoint(mx, my):
                                env.speed = env.speed + 10
                            if btn_minus.collidepoint(mx, my):
                                env.speed = max(1, env.speed - 10)
                    if not running:
                        break

                    # action selection and step (skip internal event processing)
                    transition_ready = False
                    if not paused:
                        action, last_mode, last_eps, last_q_values, last_state_idx = choose_action_with_debug(state)
                        last_action = action
                        next_state, reward, done, step_info = env.play_step(action, skip_events=True)
                        total_reward += reward
                        transition_ready = True
                        ep_steps += 1
                    else:
                        # when paused, just handle events and sleep tick
                        next_state, reward, done, step_info = state, 0, False, {}

                    # learning updates
                    if transition_ready and (not eval_only):
                        agent.push(state, action, reward, next_state, done)
                        agent.update()
                        if t % 100 == 0:
                            try:
                                agent.sync_target()
                            except Exception:
                                pass

                    state = next_state

                    # info panel at top-left
                    try:
                        env._draw_panel_box(panel_bg)

                        move_txt = action_names.get(last_action, 'n/a')
                        mode_txt = f'{last_mode} (eps={last_eps:.3f})'
                        q_txt = format_q_values(last_q_values)
                        idx_txt = ''
                        compute_txt = f'Compute: {str(getattr(agent, "device", "cpu")).upper()}'
                        run_mode_txt = 'Run mode: EVAL (no learning updates)' if eval_only else 'Run mode: TRAIN (online updates)'
                        if recent_scores:
                            avg_score50 = sum(recent_scores[-50:]) / min(50, len(recent_scores))
                            avg_steps50 = sum(recent_steps[-50:]) / min(50, len(recent_steps))
                            avg_stats_txt = f'Avg50 score/steps: {avg_score50:.2f} / {avg_steps50:.1f}'
                        else:
                            avg_stats_txt = 'Avg50 score/steps: n/a'

                        lines = [
                            f'Episode: {ep} | Step: {t}',
                            f'Score: {env.score} | Reward: {total_reward:.1f}',
                            avg_stats_txt,
                            f'Game speed: {env.speed} FPS',
                            compute_txt,
                            run_mode_txt,
                            f'Move selected: {move_txt}',
                            f'Selection mode: {mode_txt}',
                            q_txt,
                        ]
                        info_font = env.small_font or env.font
                        line_h = info_font.get_height() + 6
                        for i, ln in enumerate(lines):
                            clipped_ln = env._fit_text(ln, info_font, panel_bg.width - 12)
                            s = info_font.render(clipped_ln, True, WHITE)
                            env.display.blit(s, (panel_bg.x + 6, panel_bg.y + 6 + i * line_h))

                        # draw buttons
                        pygame.draw.rect(env.display, (180,180,100) if paused else (100,180,100), btn_pause)
                        pygame.draw.rect(env.display, PANEL_BORDER, btn_pause, 2)
                        env.display.blit(env.font.render('Pause (ep)' if not paused else 'Resume', True, BLACK), (btn_pause.x + 8, btn_pause.y + 4))
                        pygame.draw.rect(env.display, (140,140,140), btn_plus)
                        pygame.draw.rect(env.display, (140,140,140), btn_minus)
                        pygame.draw.rect(env.display, PANEL_BORDER, btn_plus, 2)
                        pygame.draw.rect(env.display, PANEL_BORDER, btn_minus, 2)
                        env.display.blit(env.font.render('+', True, BLACK), (btn_plus.x + 10, btn_plus.y + 4))
                        env.display.blit(env.font.render('-', True, BLACK), (btn_minus.x + 12, btn_minus.y + 4))

                        if eval_only:
                            env._draw_footer_block([
                                'Esc: back to menu | Pause: after current episode',
                                'Eval mode: no learning updates and eps=0',
                                'Speed +/-: change simulation FPS'
                            ])
                        else:
                            env._draw_footer_block([
                                'Esc: back to menu | Pause: after current episode',
                                'Speed +/-: change simulation FPS',
                                'Panel shows model move selection process'
                            ])

                        pygame.display.flip()
                    except Exception:
                        pass

                    if paused and env.clock:
                        env.clock.tick(30)

                    # check goal: filled board
                    occupied_cells = len(env.snake) + len(getattr(env, 'walls', set()) or set())
                    if occupied_cells >= max_cells:
                        # success
                        msg = f'Episode {ep} filled board!'
                        try:
                            env.display.blit(env.font.render(msg, True, WHITE), (10, 10))
                            pygame.display.flip()
                        except Exception:
                            pass
                        running = False
                        break

                    if done:
                        if step_info.get('board_filled'):
                            running = False
                        # short pause to show score
                        pause = 30
                        while pause > 0 and running:
                            for ev in pygame.event.get():
                                if ev.type == pygame.QUIT:
                                    pygame.quit()
                                    sys.exit(0)
                                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                                    running = False
                                    break
                            pause -= 1
                            if env.clock:
                                env.clock.tick(30)
                        break

                    recent_scores.append(float(env.score))
                    recent_steps.append(float(ep_steps))

                # one episode finished; continue to next unless stopped
                # after episode: if pause was requested, enter paused state until user resumes
                if pause_requested:
                    paused = True
                    pause_requested = False
                    # show paused panel and wait for resume
                    while paused and running:
                        panel_bg, btn_pause, btn_plus, btn_minus = env._get_left_control_rects(panel_h=260)
                        for ev in pygame.event.get():
                            if ev.type == pygame.QUIT:
                                pygame.quit()
                                sys.exit(0)
                            if ev.type == pygame.VIDEORESIZE:
                                env.resize_window(ev.w, ev.h, preserve_state=True)
                                panel_bg, btn_pause, btn_plus, btn_minus = env._get_left_control_rects(panel_h=260)
                            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                                mx, my = ev.pos
                                if btn_pause.collidepoint(mx, my):
                                    paused = False
                                    break
                                if btn_plus.collidepoint(mx, my):
                                    env.speed = env.speed + 10
                                if btn_minus.collidepoint(mx, my):
                                    env.speed = max(1, env.speed - 10)
                        # draw paused panel
                        try:
                            env._draw_panel_box(panel_bg)
                            lines = [
                                f'Episode paused after: {ep}',
                                f'Score: {env.score} | Reward: {total_reward:.1f}',
                                (f'Avg50 score/steps: '
                                 f'{(sum(recent_scores[-50:]) / min(50, len(recent_scores))):.2f} / '
                                 f'{(sum(recent_steps[-50:]) / min(50, len(recent_steps))):.1f}')
                                if recent_scores else 'Avg50 score/steps: n/a',
                                f'Game speed: {env.speed} FPS',
                                f'Compute: {str(getattr(agent, "device", "cpu")).upper()}',
                                f'Run mode: {"EVAL (no learning updates)" if eval_only else "TRAIN (online updates)"}',
                                f'Last move: {action_names.get(last_action, "n/a")}',
                                f'Last mode: {last_mode} (eps={last_eps:.3f})',
                                format_q_values(last_q_values)
                            ]
                            info_font = env.small_font or env.font
                            line_h = info_font.get_height() + 6
                            for i, ln in enumerate(lines):
                                clipped_ln = env._fit_text(ln, info_font, panel_bg.width - 12)
                                s = info_font.render(clipped_ln, True, WHITE)
                                env.display.blit(s, (panel_bg.x + 6, panel_bg.y + 6 + i * line_h))
                            pygame.draw.rect(env.display, (100,180,100), btn_pause)
                            pygame.draw.rect(env.display, PANEL_BORDER, btn_pause, 2)
                            env.display.blit(env.font.render('Resume', True, BLACK), (btn_pause.x + 8, btn_pause.y + 4))
                            pygame.draw.rect(env.display, (140,140,140), btn_plus)
                            pygame.draw.rect(env.display, (140,140,140), btn_minus)
                            pygame.draw.rect(env.display, PANEL_BORDER, btn_plus, 2)
                            pygame.draw.rect(env.display, PANEL_BORDER, btn_minus, 2)
                            env.display.blit(env.font.render('+', True, BLACK), (btn_plus.x + 10, btn_plus.y + 4))
                            env.display.blit(env.font.render('-', True, BLACK), (btn_minus.x + 12, btn_minus.y + 4))

                            if eval_only:
                                env._draw_footer_block([
                                    'Paused between episodes',
                                    'Resume to continue evaluation',
                                    'Speed +/- works while paused'
                                ])
                            else:
                                env._draw_footer_block([
                                    'Paused between episodes',
                                    'Resume to continue training',
                                    'Speed +/- works while paused'
                                ])

                            pygame.display.flip()
                        except Exception:
                            pass
                        if env.clock:
                            env.clock.tick(30)
            # live training finished; leave env in a clean reset state
            env.reset()

        running = True
        while running:
            # main menu
            choice = None
            menu_running = True
            while menu_running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit(0)
                    if event.type == pygame.VIDEORESIZE:
                        g.resize_window(event.w, event.h, preserve_state=True)
                        center_x = g.w // 2
                        btn1_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 - 150, btn_w, btn_h)
                        btn2_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 - 60, btn_w, btn_h)
                        btn3_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 + 30, btn_w, btn_h)
                        btn4_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 + 120, btn_w, btn_h)
                        btn5_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 + 210, btn_w, btn_h)
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        mx, my = event.pos
                        if btn1_rect.collidepoint(mx, my):
                            choice = '1'
                            menu_running = False
                        elif btn2_rect.collidepoint(mx, my):
                            choice = '2'
                            menu_running = False
                        elif btn3_rect.collidepoint(mx, my):
                            choice = '3'
                            menu_running = False
                        elif btn4_rect.collidepoint(mx, my):
                            choice = '4'
                            menu_running = False
                        elif btn5_rect.collidepoint(mx, my):
                            choice = '5'
                            menu_running = False

                # draw menu (do not render game in background)
                g.display.fill(BLACK)
                title = g.font.render('Snake AI - Menu', True, WHITE)
                g.display.blit(title, (center_x - title.get_width() // 2, g.h // 2 - 180))

                # buttons
                pygame.draw.rect(g.display, (50, 50, 50), btn1_rect)
                pygame.draw.rect(g.display, (50, 50, 50), btn2_rect)
                pygame.draw.rect(g.display, (50, 50, 50), btn3_rect)
                pygame.draw.rect(g.display, (50, 50, 50), btn4_rect)
                pygame.draw.rect(g.display, (50, 50, 50), btn5_rect)
                t1 = g.font.render('Level Design', True, WHITE)
                t2 = g.font.render('Classic Snake', True, WHITE)
                t3 = g.font.render('Train on Level', True, WHITE)
                t4 = g.font.render('Load checkpoint (.pth)', True, WHITE)
                t5 = g.font.render('Load level (.json)', True, WHITE)
                g.display.blit(t1, (btn1_rect.x + 12, btn1_rect.y + btn_h // 2 - t1.get_height() // 2))
                g.display.blit(t2, (btn2_rect.x + 12, btn2_rect.y + btn_h // 2 - t2.get_height() // 2))
                g.display.blit(t3, (btn3_rect.x + 12, btn3_rect.y + btn_h // 2 - t3.get_height() // 2))
                g.display.blit(t4, (btn4_rect.x + 12, btn4_rect.y + btn_h // 2 - t4.get_height() // 2))
                g.display.blit(t5, (btn5_rect.x + 12, btn5_rect.y + btn_h // 2 - t5.get_height() // 2))

                # show current selections
                if g.current_level_name:
                    lvl_s = g.font.render(f'Current level: {g.current_level_name}', True, WHITE)
                    g.display.blit(lvl_s, (10, 10))
                if g.current_checkpoint_path:
                    cp_s = g.font.render(f'Checkpoint: {os.path.basename(g.current_checkpoint_path)}', True, WHITE)
                    g.display.blit(cp_s, (10, 36))

                pygame.display.flip()
                if g.clock:
                    g.clock.tick(30)

            # handle choice
            if choice == '1':
                # Level Design: open editor (designer saves and returns path or None)
                res = g.level_designer()
                if res is None:
                    info_msg = 'Level edit cancelled.'
                else:
                    # designer already saved and returned the path
                    level_path = res
                    level_name = os.path.basename(level_path)
                    g.current_level_name = level_name
                    g.current_level_path = level_path
                    info_msg = f'Saved level: {level_name}'

                # show brief info and return to main menu
                info_timer = 45
                while info_timer > 0:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit(0)
                    g.display.fill(BLACK)
                    info_surf = g.font.render(info_msg, True, WHITE)
                    g.display.blit(info_surf, (10, 40))
                    pygame.display.flip()
                    info_timer -= 1
                    if g.clock:
                        g.clock.tick(30)
            elif choice == '2':
                # Classic Snake: run a demo until user returns (Esc) or game over
                try:
                    running_demo = True
                    paused = False
                    action_names = {0: 'STRAIGHT', 1: 'RIGHT TURN', 2: 'LEFT TURN'}
                    panel_bg, btn_pause, btn_plus, btn_minus = g._get_left_control_rects(panel_h=220)
                    step = 0
                    last_action = None
                    selection_mode = 'random_policy'
                    state = g.get_state()
                    while running_demo:
                        panel_bg, btn_pause, btn_plus, btn_minus = g._get_left_control_rects(panel_h=220)
                        for ev in pygame.event.get():
                            if ev.type == pygame.QUIT:
                                pygame.quit()
                                sys.exit(0)
                            if ev.type == pygame.VIDEORESIZE:
                                g.resize_window(ev.w, ev.h, preserve_state=True)
                                panel_bg, btn_pause, btn_plus, btn_minus = g._get_left_control_rects(panel_h=220)
                            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                                running_demo = False
                                break
                            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                                mx, my = ev.pos
                                if btn_pause.collidepoint(mx, my):
                                    paused = not paused
                                if btn_plus.collidepoint(mx, my):
                                    g.speed += 10
                                if btn_minus.collidepoint(mx, my):
                                    g.speed = max(1, g.speed - 10)
                        if not paused:
                            action = random.randint(0, 2)
                            last_action = action
                            state, reward, done, info = g.play_step(action)
                            step += 1
                        else:
                            state, reward, done, info = state, 0, False, {}
                        # draw demo left info panel
                        try:
                            pg = g.display
                            g._draw_panel_box(panel_bg)
                            lines = [
                                f'Step: {step} | Score: {g.score}',
                                f'Game speed: {g.speed} FPS',
                                f'Move selected: {action_names.get(last_action, "n/a")}',
                                f'Selection mode: {selection_mode}',
                                'Q[S,R,L]: n/a (random demo)'
                            ]
                            info_font = g.small_font or g.font
                            line_h = info_font.get_height() + 6
                            for i, ln in enumerate(lines):
                                clipped_ln = g._fit_text(ln, info_font, panel_bg.width - 12)
                                s = info_font.render(clipped_ln, True, WHITE)
                                pg.blit(s, (panel_bg.x + 6, panel_bg.y + 6 + i * line_h))
                            pygame.draw.rect(pg, (180,180,100) if paused else (100,180,100), btn_pause)
                            pygame.draw.rect(pg, PANEL_BORDER, btn_pause, 2)
                            pg.blit(g.font.render('Pause' if not paused else 'Resume', True, BLACK), (btn_pause.x + 8, btn_pause.y + 4))
                            pygame.draw.rect(pg, (140,140,140), btn_plus)
                            pygame.draw.rect(pg, (140,140,140), btn_minus)
                            pygame.draw.rect(pg, PANEL_BORDER, btn_plus, 2)
                            pygame.draw.rect(pg, PANEL_BORDER, btn_minus, 2)
                            pg.blit(g.font.render('+', True, BLACK), (btn_plus.x + 10, btn_plus.y + 4))
                            pg.blit(g.font.render('-', True, BLACK), (btn_minus.x + 12, btn_minus.y + 4))

                            g._draw_footer_block([
                                'Esc: back to menu | Pause/Resume with button',
                                'Speed +/-: change simulation FPS',
                                'Classic mode uses random next-move selection'
                            ])

                            pygame.display.flip()
                        except Exception:
                            pass

                        if paused and g.clock:
                            g.clock.tick(30)

                        if done:
                            # short pause to show score
                            pause = 30
                            while pause > 0:
                                for ev in pygame.event.get():
                                    if ev.type == pygame.QUIT:
                                        pygame.quit()
                                        sys.exit(0)
                                pause -= 1
                                if g.clock:
                                    g.clock.tick(30)
                            running_demo = False
                    # reset game to clean state
                    g.reset()
                finally:
                    pass
            elif choice == '4':
                # Load checkpoint (.pth) using picker with fallback
                path = pick_file_dialog(
                    [('PyTorch','*.pth'), ('All files','*.*')],
                    fallback_exts=('.pth',),
                    fallback_dirs=('.', 'logs'),
                    fallback_title='Select checkpoint (.pth) - Esc to cancel'
                )
                if path:
                    g.current_checkpoint_path = path
                    info_msg = f'Checkpoint selected: {os.path.basename(path)}'
                else:
                    info_msg = 'No checkpoint selected.'
                # brief info
                info_timer = 45
                while info_timer > 0:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit(0)
                    g.display.fill(BLACK)
                    info_surf = g.font.render(info_msg, True, WHITE)
                    g.display.blit(info_surf, (10, 40))
                    pygame.display.flip()
                    info_timer -= 1
                    if g.clock:
                        g.clock.tick(30)
            elif choice == '5':
                # Load level layout (.json) using picker with fallback
                path = pick_file_dialog(
                    [('JSON level','*.json'), ('All files','*.*')],
                    fallback_exts=('.json',),
                    fallback_dirs=('levels', '.'),
                    fallback_title='Select level (.json) - Esc to cancel'
                )
                if path:
                    g.current_level_path = path
                    g.current_level_name = os.path.basename(path)
                    info_msg = f'Level selected: {g.current_level_name}'
                else:
                    info_msg = 'No level selected.'
                # brief info
                info_timer = 45
                while info_timer > 0:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit(0)
                    g.display.fill(BLACK)
                    info_surf = g.font.render(info_msg, True, WHITE)
                    g.display.blit(info_surf, (10, 40))
                    pygame.display.flip()
                    info_timer -= 1
                    if g.clock:
                        g.clock.tick(30)
            elif choice == '3':
                # Train on Level: simple submenu to choose headless vs live and algorithm
                # UI elements
                sub_w = 420
                sub_h = 430
                headless = True
                algo = 'dqn'
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
                    btn_start = pygame.Rect(sx + 20, sy + 270, 180, 50)
                    btn_back = pygame.Rect(sx + 220, sy + 270, 180, 50)

                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit(0)
                        if ev.type == pygame.VIDEORESIZE:
                            g.resize_window(ev.w, ev.h, preserve_state=True)
                            sx = g.w // 2 - sub_w // 2
                            sy = g.h // 2 - sub_h // 2
                            sub_rect = pygame.Rect(sx, sy, sub_w, sub_h)
                            btn_headless = pygame.Rect(sx + 20, sy + 40, 180, 40)
                            btn_live = pygame.Rect(sx + 220, sy + 40, 180, 40)
                            btn_eval = pygame.Rect(sx + 20, sy + 100, 380, 40)
                            btn_start = pygame.Rect(sx + 20, sy + 270, 180, 50)
                            btn_back = pygame.Rect(sx + 220, sy + 270, 180, 50)
                        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                            mx, my = ev.pos
                            if btn_headless.collidepoint(mx, my):
                                headless = True
                            elif btn_live.collidepoint(mx, my):
                                headless = False
                            elif btn_eval.collidepoint(mx, my):
                                eval_only = not eval_only
                            elif btn_start.collidepoint(mx, my):
                                # start training (headless uses subprocess)
                                cmd = [sys.executable, 'train.py']
                                if getattr(g, 'current_level_path', None):
                                    cmd += ['--level', g.current_level_path]
                                if getattr(g, 'current_checkpoint_path', None):
                                    cmd += ['--init-checkpoint', g.current_checkpoint_path]
                                if headless:
                                    try:
                                        subprocess.Popen(cmd)
                                        info_msg = 'Background training started (DQN).'
                                    except Exception as e:
                                        info_msg = f'Failed to start: {e}'
                                    submenu = False
                                else:
                                    # launch live in-process trainer that renders episodes
                                    try:
                                        live_train(
                                            g,
                                            max_episodes=10000,
                                            max_steps=MAX_EPISODE_MOVES,
                                            init_ckpt=getattr(g, 'current_checkpoint_path', None),
                                            eval_only=eval_only
                                        )
                                        info_msg = 'Live training finished.'
                                    except Exception as e:
                                        info_msg = f'Live training failed: {e}'
                                    submenu = False
                            elif btn_back.collidepoint(mx, my):
                                submenu = False

                    # draw submenu
                    g.display.fill((20, 20, 20))
                    pygame.draw.rect(g.display, (60,60,60), sub_rect)
                    title = g.font.render('Train on Level', True, WHITE)
                    g.display.blit(title, (sx + sub_w//2 - title.get_width()//2, sy + 8))
                    # mode buttons
                    pygame.draw.rect(g.display, (100,180,100) if headless else (150,150,150), btn_headless)
                    pygame.draw.rect(g.display, (100,180,100) if not headless else (150,150,150), btn_live)
                    g.display.blit(g.font.render('Headless', True, BLACK), (btn_headless.x + 12, btn_headless.y + 8))
                    g.display.blit(g.font.render('Live (demo)', True, BLACK), (btn_live.x + 12, btn_live.y + 8))

                    # eval toggle for live mode
                    eval_enabled_for_mode = (not headless)
                    eval_bg = (100, 180, 100) if (eval_only and eval_enabled_for_mode) else (150, 150, 150)
                    if not eval_enabled_for_mode:
                        eval_bg = (120, 120, 120)
                    pygame.draw.rect(g.display, eval_bg, btn_eval)
                    eval_text = f'Evaluate checkpoint only (no learning): {"ON" if eval_only else "OFF"}'
                    eval_label = g._fit_text(eval_text, g.font, btn_eval.width - 16)
                    g.display.blit(g.font.render(eval_label, True, BLACK), (btn_eval.x + 8, btn_eval.y + 8))

                    # runtime backend status
                    status_font = g.small_font or g.font
                    if torch_info['installed']:
                        line1 = f"Torch: {torch_info['version']} (CUDA build: {torch_info['cuda_build']})"
                        if torch_info['cuda_available']:
                            line2 = f"CUDA: ON ({torch_info['device_name']})"
                        else:
                            line2 = 'CUDA: OFF (CPU-only torch build or unavailable)'
                    else:
                        line1 = 'Torch: not installed'
                        line2 = 'CUDA: OFF'
                    line3 = 'Algorithm: DQN (Deep Q-Network)'
                    line4 = 'Eval toggle applies in Live mode; headless always trains.'
                    status_lines = [line1, line2, line3, line4]
                    for i, ln in enumerate(status_lines):
                        clipped = g._fit_text(ln, status_font, sub_w - 24)
                        surf = status_font.render(clipped, True, WHITE)
                        g.display.blit(surf, (sx + 12, sy + 212 + i * (status_font.get_height() + 4)))

                    # start/back
                    pygame.draw.rect(g.display, (80,200,120), btn_start)
                    pygame.draw.rect(g.display, (200,80,80), btn_back)
                    g.display.blit(g.font.render('Start', True, BLACK), (btn_start.x + 60, btn_start.y + 12))
                    g.display.blit(g.font.render('Back', True, BLACK), (btn_back.x + 68, btn_back.y + 14))
                    pygame.display.flip()
                    if g.clock:
                        g.clock.tick(30)
                # show info (from start/back)
                if 'info_msg' in locals():
                    info_timer = 45
                    while info_timer > 0:
                        for ev in pygame.event.get():
                            if ev.type == pygame.QUIT:
                                pygame.quit()
                                sys.exit(0)
                        g.display.fill(BLACK)
                        info_surf = g.font.render(info_msg, True, WHITE)
                        g.display.blit(info_surf, (10, 40))
                        pygame.display.flip()
                        info_timer -= 1
                        if g.clock:
                            g.clock.tick(30)