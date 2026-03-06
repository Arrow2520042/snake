import pygame
import random
import os
import sys
import json
import datetime
import subprocess
from enum import Enum
from collections import namedtuple
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

BLOCK_SIZE = 20
DEFAULT_SPEED = 200 # Możesz to zmieniać, żeby przyspieszyć lub zwolnić symulację

class SnakeGameAI:
    def __init__(self, w=640, h=480, render=True, seed=None, speed=DEFAULT_SPEED):
        self.w = w
        self.h = h
        self.render = render
        self.speed = speed
        if seed is not None:
            random.seed(seed)

        # Inicjalizacja okna gry tylko jeśli renderujemy
        if self.render:
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake AI - Projekt')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(pygame.font.get_default_font(), 25)
        else:
            self.display = None
            self.clock = None
            self.font = None

        self.reset()
        
    def reset(self):
        # Stan początkowy gry (reset po śmierci węża)
        self.direction = Direction.RIGHT
        # Wyrównanie pozycji głowy do siatki (całkowite współrzędne)
        cx = (self.w // 2)
        cy = (self.h // 2)
        cx = (cx // BLOCK_SIZE) * BLOCK_SIZE
        cy = (cy // BLOCK_SIZE) * BLOCK_SIZE
        self.head = Point(int(cx), int(cy))
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        # zwróć obserwację (przydatne dla RL)
        return self.get_state()
        
    def _place_food(self):
        # Losowanie pozycji jedzenia na siatce, bez rekurencji
        max_x = (self.w - BLOCK_SIZE) // BLOCK_SIZE
        max_y = (self.h - BLOCK_SIZE) // BLOCK_SIZE
        for _ in range(1000):
            x = random.randint(0, max_x) * BLOCK_SIZE
            y = random.randint(0, max_y) * BLOCK_SIZE
            p = Point(int(x), int(y))
            if p not in self.snake:
                self.food = p
                return
        # Fallback: jeżeli nie znaleziono pozycji (bardzo mała plansza), ustaw losowo bez gwarancji
        self.food = Point(0, 0)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Uderzenie w ścianę
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
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

        # 2. Wykonaj ruch bazując na akcji od AI
        act_idx = self._parse_action(action)
        self._move(act_idx)  # aktualizuje self.head
        self.snake.insert(0, self.head)

        # 3. Sprawdź zakończenie i nagrody
        reward = 0
        # mała kara za krok, by zapobiec bezcelowemu krążeniu
        reward -= 0.01
        done = False

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            done = True
            reward = -10
            return self.get_state(), reward, done, {"score": self.score}

        # 4. Jedzenie
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. Render i zegar
        if self.render:
            self._update_ui()
            if self.clock:
                self.clock.tick(self.speed)

        return self.get_state(), reward, done, {"score": self.score}
        
    def _update_ui(self):
        if not self.render:
            return
        self.display.fill(BLACK)
        
        # Rysowanie węża
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        # Rysowanie jedzenia
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        # Rysowanie bloków ścian
        if hasattr(self, 'walls') and self.walls:
            for w in self.walls:
                pygame.draw.rect(self.display, (100, 100, 100), pygame.Rect(w.x, w.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Wyświetlanie wyniku (z tłem, żeby uniknąć migotania)
        score_text = f"Score: {self.score}"
        text = self.font.render(score_text, True, WHITE)
        # background rect
        self.display.fill(BLACK, rect=pygame.Rect(0, 0, text.get_width() + 8, text.get_height() + 8))
        self.display.blit(text, (4, 4))
        # Wyświetlanie generacji/epizodu w prawym górnym rogu (jeśli dostępne)
        if hasattr(self, 'generation'):
            gen_text = self.font.render(f'Gen: {self.generation}', True, WHITE)
            gr = pygame.Rect(self.w - gen_text.get_width() - 12, 4, gen_text.get_width() + 8, gen_text.get_height() + 4)
            self.display.fill(BLACK, rect=gr)
            self.display.blit(gen_text, (gr.x + 4, gr.y))
        pygame.display.flip()

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
        """Prosty edytor poziomów: kliknij, aby dodać/usunąć blok. Naciśnij Enter, aby zakończyć."""
        if not self.render:
            print('Level designer wymaga trybu render (GUI).')
            return set()
        if not hasattr(self, 'walls'):
            self.walls = set()

        import json
        designing = True
        status_msg = ''
        status_timer = 0
        while designing:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return set()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        designing = False
                    if event.key == pygame.K_s:
                        # save to level.json
                        data = [[p.x, p.y] for p in sorted(self.walls, key=lambda x:(x.x,x.y))]
                        try:
                            with open('level.json','w',encoding='utf-8') as f:
                                json.dump(data, f)
                            status_msg = 'Saved level.json'
                        except Exception as e:
                            status_msg = f'Save error: {e}'
                        status_timer = 120
                    if event.key == pygame.K_l:
                        # load from level.json
                        try:
                            with open('level.json','r',encoding='utf-8') as f:
                                data = json.load(f)
                            self.walls = set(Point(int(x),int(y)) for x,y in data)
                            status_msg = 'Loaded level.json'
                        except Exception as e:
                            status_msg = f'Load error: {e}'
                        status_timer = 120
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    gx = (mx // BLOCK_SIZE) * BLOCK_SIZE
                    gy = (my // BLOCK_SIZE) * BLOCK_SIZE
                    p = Point(gx, gy)
                    if p in self.walls:
                        self.walls.remove(p)
                    else:
                        self.walls.add(p)

            # rysuj
            self._update_ui()
            # dodatkowe instrukcje
            info_text = self.font.render('Level Designer: click to toggle walls, Enter to finish (S=save, L=load)', True, WHITE)
            self.display.blit(info_text, [10, self.h - 30])
            if status_timer > 0 and status_msg:
                status_surf = self.font.render(status_msg, True, WHITE)
                self.display.blit(status_surf, [10, self.h - 60])
                status_timer -= 1
            pygame.display.flip()
            if self.clock:
                self.clock.tick(30)

        return self.walls

    def run_ai_visualization(self, walls=None, episodes=5):
        """Prosta wizualizacja AI na zdefiniowanym poziomie.
        Jeśli przekażesz argument `agent` (obiekt z metodą `act(state)`),
        będzie on sterował wężem. W przeciwnym razie użyte będą losowe akcje.
        Pokazuje numer epizodu/generacji w rogu ekranu.
        """
        def _ensure_walls(w):
            return set(w) if w is not None else set()

        if walls is None:
            walls = set()
        self.walls = _ensure_walls(walls)
        for e in range(1, episodes + 1):
            self.generation = e
            self.reset()
            self.walls = _ensure_walls(walls)
            done = False
            while not done:
                # choose action via provided agent if present, else random
                if hasattr(self, 'visual_agent') and self.visual_agent is not None:
                    state = self.get_state()
                    action = self.visual_agent.act(state)
                else:
                    action = random.randint(0, 2)
                state, reward, done, info = self.play_step(action)
            # krótka pauza między epizodami
            if self.render:
                pygame.time.wait(500)

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

        # pomocnicza funkcja do obliczenia punktu w danym kierunku
        def move_point(pt, direction):
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

        # Kierunki względem aktualnego kierunku
        if self.direction == Direction.RIGHT:
            point_s = move_point(head, Direction.RIGHT)
            point_r = move_point(head, Direction.DOWN)
            point_l = move_point(head, Direction.UP)
        elif self.direction == Direction.LEFT:
            point_s = move_point(head, Direction.LEFT)
            point_r = move_point(head, Direction.UP)
            point_l = move_point(head, Direction.DOWN)
        elif self.direction == Direction.UP:
            point_s = move_point(head, Direction.UP)
            point_r = move_point(head, Direction.RIGHT)
            point_l = move_point(head, Direction.LEFT)
        else:  # DOWN
            point_s = move_point(head, Direction.DOWN)
            point_r = move_point(head, Direction.LEFT)
            point_l = move_point(head, Direction.RIGHT)

        danger_s = int(self.is_collision(point_s))
        danger_r = int(self.is_collision(point_r))
        danger_l = int(self.is_collision(point_l))

        dir_up = int(self.direction == Direction.UP)
        dir_down = int(self.direction == Direction.DOWN)
        dir_left = int(self.direction == Direction.LEFT)
        dir_right = int(self.direction == Direction.RIGHT)

        # wektor do jedzenia (znormalizowany względem planszy)
        food_dx = (self.food.x - head.x) / self.w
        food_dy = (self.food.y - head.y) / self.h

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
        # GUI menu inside the Pygame window (non-blocking)
        menu_running = True
        choice = None

        btn_w = 360
        btn_h = 64
        center_x = g.w // 2
        btn1_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 - 150, btn_w, btn_h)
        btn2_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 - 60, btn_w, btn_h)
        btn3_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 + 30, btn_w, btn_h)
        btn4_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 + 120, btn_w, btn_h)

        while menu_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
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

            # rysuj menu
            g.display.fill(BLACK)
            title = g.font.render('Snake AI - Menu', True, WHITE)
            g.display.blit(title, (center_x - title.get_width() // 2, g.h // 2 - 180))

            # przyciski
            pygame.draw.rect(g.display, (50, 50, 50), btn1_rect)
            pygame.draw.rect(g.display, (50, 50, 50), btn2_rect)
            pygame.draw.rect(g.display, (50, 50, 50), btn3_rect)
            pygame.draw.rect(g.display, (50, 50, 50), btn4_rect)
            t1 = g.font.render('Level Designer -> AI Visualization', True, WHITE)
            t2 = g.font.render('Play classic Snake (keyboard)', True, WHITE)
            t3 = g.font.render('Train DQN on Level', True, WHITE)
            t4 = g.font.render('Demo Agent (load model)', True, WHITE)
            g.display.blit(t1, (btn1_rect.x + 12, btn1_rect.y + btn_h // 2 - t1.get_height() // 2))
            g.display.blit(t2, (btn2_rect.x + 12, btn2_rect.y + btn_h // 2 - t2.get_height() // 2))
            g.display.blit(t3, (btn3_rect.x + 12, btn3_rect.y + btn_h // 2 - t3.get_height() // 2))
            g.display.blit(t4, (btn4_rect.x + 12, btn4_rect.y + btn_h // 2 - t4.get_height() // 2))

            pygame.display.flip()
            if g.clock:
                g.clock.tick(30)

        try:
            if choice == '1':
                walls = g.level_designer()
                g.run_ai_visualization(walls=walls, episodes=5)
            elif choice == '2':
                # manual play
                if not hasattr(g, 'walls'):
                    g.walls = set()
                done = False
                while not done:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            done = True
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_LEFT and g.direction != Direction.RIGHT:
                                g.direction = Direction.LEFT
                            if event.key == pygame.K_RIGHT and g.direction != Direction.LEFT:
                                g.direction = Direction.RIGHT
                            if event.key == pygame.K_UP and g.direction != Direction.DOWN:
                                g.direction = Direction.UP
                            if event.key == pygame.K_DOWN and g.direction != Direction.UP:
                                g.direction = Direction.DOWN
                    # move forward (action=0 => straight)
                    state, reward, done_step, info = g.play_step(0, skip_events=True)
                    if done_step:
                        # show final score briefly
                        print('Game over. Score:', g.score)
                        pygame.time.wait(800)
                        done = True
            elif choice == '4':
                # Demo agent: try to load model.pth (DQN) or q_table.npy (tabular)
                walls = g.level_designer()
                model_loaded = False
                # prefer DQN model
                try:
                    from dqn_agent import DQNAgent
                except Exception:
                    DQNAgent = None

                import os
                # try model.pth
                if os.path.exists('model.pth') and DQNAgent is not None:
                    try:
                        agent = DQNAgent()
                        agent.load('model.pth')
                        g.visual_agent = agent
                        model_loaded = True
                    except Exception:
                        model_loaded = False
                # else try any model_ep*.pth
                if not model_loaded and DQNAgent is not None:
                    files = [f for f in os.listdir('.') if f.startswith('model_ep') and f.endswith('.pth')]
                    if files:
                        try:
                            agent = DQNAgent()
                            agent.load(files[-1])
                            g.visual_agent = agent
                            model_loaded = True
                        except Exception:
                            model_loaded = False

                # try tabular
                if not model_loaded and os.path.exists('q_table.npy'):
                    try:
                        from rl_agent import QLearningAgent
                        q = QLearningAgent()
                        q.q = __import__('numpy').load('q_table.npy')
                        g.visual_agent = q
                        model_loaded = True
                    except Exception:
                        model_loaded = False

                if not model_loaded:
                    # show message until keypress
                    msg = 'No model found (model.pth or q_table.npy) or required libs missing.'
                    showing = True
                    while showing:
                        for ev in pygame.event.get():
                            if ev.type == pygame.QUIT:
                                pygame.quit()
                                sys.exit(0)
                            if ev.type == pygame.KEYDOWN:
                                showing = False
                        g._update_ui()
                        err_surf = g.font.render(msg, True, WHITE)
                        g.display.blit(err_surf, (10, 40))
                        pygame.display.flip()
                        if g.clock:
                            g.clock.tick(10)
                else:
                    # run visualization with loaded agent
                    g.visual_agent = getattr(g, 'visual_agent', None)
                    g.run_ai_visualization(walls=walls, episodes=5)
            elif choice == '3':
                # Train DQN on designed level: let user choose Background or Live training
                walls = g.level_designer()

                # chooser UI
                center_x = g.w // 2
                btn_w = 360
                btn_h = 64
                bg_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 - 40, btn_w, btn_h)
                live_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 + 40, btn_w, btn_h)

                choosing = True
                mode = None
                while choosing:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit(0)
                        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                            mx, my = ev.pos
                            if bg_rect.collidepoint(mx, my):
                                mode = 'background'
                                choosing = False
                            if live_rect.collidepoint(mx, my):
                                mode = 'live'
                                choosing = False
                        if ev.type == pygame.KEYDOWN:
                            if ev.key == pygame.K_1:
                                mode = 'background'
                                choosing = False
                            if ev.key == pygame.K_2:
                                mode = 'live'
                                choosing = False

                    # draw chooser
                    g._update_ui()
                    pygame.draw.rect(g.display, (60, 60, 60), bg_rect)
                    pygame.draw.rect(g.display, (60, 60, 60), live_rect)
                    tbg = g.font.render('Start background trainer (recommended)', True, WHITE)
                    tlive = g.font.render('Live training in this window (slower)', True, WHITE)
                    g.display.blit(tbg, (bg_rect.x + 12, bg_rect.y + btn_h // 2 - tbg.get_height() // 2))
                    g.display.blit(tlive, (live_rect.x + 12, live_rect.y + btn_h // 2 - tlive.get_height() // 2))
                    hint = g.font.render('Press 1 = background, 2 = live', True, WHITE)
                    g.display.blit(hint, (center_x - hint.get_width() // 2, live_rect.y + 90))
                    pygame.display.flip()
                    if g.clock:
                        g.clock.tick(30)

                if mode == 'background':
                    # save the designed level to a timestamped file so train.py can load it
                    levels_dir = os.path.join('levels')
                    os.makedirs(levels_dir, exist_ok=True)
                    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                    level_name = f'level_{ts}.json'
                    level_path = os.path.join(levels_dir, level_name)
                    try:
                        data = [[p.x, p.y] for p in sorted(walls, key=lambda x: (x.x, x.y))]
                        with open(level_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f)
                    except Exception:
                        level_path = 'level.json'

                    # prefer launching train.py as an external process so logs are created by train.py
                    python_exe = sys.executable or 'python'
                    save_name = f'model_{ts}.pth'
                    cmd = [python_exe, 'train.py', '--episodes', '200', '--algo', 'dqn', '--save', save_name, '--level', level_path]
                    # if local environment has CUDA, instruct trainer to use it
                    try:
                        import torch as _torch
                        if _torch.cuda.is_available():
                            cmd += ['--device', 'cuda']
                    except Exception:
                        pass
                    try:
                        subprocess.Popen(cmd)
                        # inform the user in GUI briefly
                        info_msg = f'Started background training: {save_name}'
                        info_timer = 180
                        while info_timer > 0:
                            for ev in pygame.event.get():
                                if ev.type == pygame.QUIT:
                                    pygame.quit()
                                    sys.exit(0)
                            g._update_ui()
                            info_surf = g.font.render(info_msg, True, WHITE)
                            g.display.blit(info_surf, (10, 40))
                            pygame.display.flip()
                            info_timer -= 1
                            if g.clock:
                                g.clock.tick(10)
                    except Exception as e:
                        msg = f'Failed to start trainer: {e}'
                        showing = True
                        while showing:
                            for ev in pygame.event.get():
                                if ev.type == pygame.QUIT:
                                    pygame.quit()
                                    sys.exit(0)
                                if ev.type == pygame.KEYDOWN:
                                    showing = False
                            g._update_ui()
                            err_surf = g.font.render(msg, True, WHITE)
                            g.display.blit(err_surf, (10, 40))
                            pygame.display.flip()
                            if g.clock:
                                g.clock.tick(10)

                elif mode == 'live':
                    # In-process live training (slower, but visible). Recreate earlier in-loop trainer.
                    # import agent and detect CUDA availability
                    try:
                        from dqn_agent import DQNAgent
                        import torch as _torch
                        cuda_avail = _torch.cuda.is_available()
                    except Exception as e:
                        DQNAgent = None
                        cuda_avail = False
                        msg = f'DQN unavailable: {e}. Install PyTorch.'
                    if DQNAgent is None:
                        showing = True
                        while showing:
                            for ev in pygame.event.get():
                                if ev.type == pygame.QUIT:
                                    pygame.quit()
                                    sys.exit(0)
                                if ev.type == pygame.KEYDOWN:
                                    showing = False
                            g._update_ui()
                            err_surf = g.font.render(msg, True, WHITE)
                            g.display.blit(err_surf, (10, 40))
                            pygame.display.flip()
                            if g.clock:
                                g.clock.tick(10)
                    else:
                        # choose device for live training: prefer CUDA if available
                        dev = 'cuda' if cuda_avail else 'cpu'
                        agent = DQNAgent(device=dev)
                        episodes = 200
                        max_steps = 1000
                        sync_every = 500
                        save_every = 50
                        steps_per_frame = 1

                        g.walls = set(walls)
                        state = g.reset()
                        g.walls = set(walls)
                        ep = 0
                        step = 0
                        ep_reward = 0
                        recent_rewards = []
                        training = True
                        paused = False
                        while training:
                            for ev in pygame.event.get():
                                if ev.type == pygame.QUIT:
                                    pygame.quit()
                                    sys.exit(0)
                                if ev.type == pygame.KEYDOWN:
                                    if ev.key == pygame.K_p:
                                        paused = not paused
                                    if ev.key == pygame.K_s:
                                        agent.save('model.pth')
                                    if ev.key == pygame.K_q:
                                        training = False

                            if not paused:
                                for _ in range(steps_per_frame):
                                    action = agent.act(state)
                                    next_state, reward, done, info = g.play_step(action, skip_events=True)
                                    agent.push(state, action, reward, next_state, done)
                                    agent.update()
                                    state = next_state
                                    ep_reward += reward
                                    step += 1
                                    if step % sync_every == 0:
                                        agent.sync_target()
                                    if done or step >= max_steps:
                                        ep += 1
                                        recent_rewards.append(ep_reward)
                                        if len(recent_rewards) > 100:
                                            recent_rewards.pop(0)
                                        ep_reward = 0
                                        step = 0
                                        state = g.reset()
                                        g.walls = set(walls)
                                        if ep % save_every == 0:
                                            try:
                                                agent.save(f'model_ep{ep}.pth')
                                            except Exception:
                                                pass
                                        if ep >= episodes:
                                            training = False

                            # overlay metrics
                            avg = sum(recent_rewards)/len(recent_rewards) if recent_rewards else 0.0
                            eps = getattr(agent, 'eps', 0.0)
                            info_s = g.font.render(f'Ep: {ep}/{episodes}  EPS: {eps:.3f}  AvgR: {avg:.2f}', True, WHITE)
                            g._update_ui()
                            g.display.blit(info_s, (10, 40))
                            pygame.display.flip()
                            if g.clock:
                                g.clock.tick(30)
        finally:
            pygame.quit()