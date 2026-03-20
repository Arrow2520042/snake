"""Snake game controller (MVC: Controller layer).

This module orchestrates user interactions, menu flow, and evaluation
visualization. It coordinates the Model (game_model.py) and the View helpers
(game_render.py, game_layout.py, game_designer.py).
"""

import argparse
import json
import os
import random
import sys

import pygame

from game_model import BLACK, MAX_EPISODE_MOVES, PANEL_BORDER, WHITE, SnakeGameAI


def run_cli(argv=None):
    """Run the interactive Snake GUI controller from command line arguments."""
    parser = argparse.ArgumentParser(description='Run SnakeGameAI example')
    parser.add_argument('--no-render', action='store_true', help='Run without opening a window')
    parser.add_argument('--eval-seed', type=int, default=None,
                        help='Base seed for deterministic visualization episodes')
    args = parser.parse_args(argv)

    g = SnakeGameAI(render=not args.no_render)
    print(f'Starting game (render={g.render})...')

    if not g.render:
        while True:
            action = random.randint(0, 2)
            _, _, done, _ = g.play_step(action)
            if done:
                print('Game over. Score:', g.score)
                break
        return

    btn_w = 360
    btn_h = 64
    center_x = g.w // 2
    btn1_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 - 120, btn_w, btn_h)
    btn2_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 - 30, btn_w, btn_h)
    btn3_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 + 60, btn_w, btn_h)
    btn4_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 + 150, btn_w, btn_h)

    session_cfg = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'session_cfg.json',
    )

    def _load_session_cfg():
        try:
            with open(session_cfg, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_session_cfg(**kwargs):
        try:
            data = _load_session_cfg()
            data.update(kwargs)
            with open(session_cfg, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception:
            pass

    def _parse_int_or_none(value):
        try:
            return int(value)
        except Exception:
            return None

    def _detect_eval_seed_from_checkpoint(ckpt_path):
        """Try to infer eval_seed from sibling info.txt next to a checkpoint."""
        if not ckpt_path or not os.path.isfile(ckpt_path):
            return None
        info_path = os.path.join(os.path.dirname(ckpt_path), 'info.txt')
        if not os.path.isfile(info_path):
            return None
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                for raw in f:
                    line = raw.strip()
                    if not line.lower().startswith('eval_seed:'):
                        continue
                    val = line.split(':', 1)[1].strip()
                    return _parse_int_or_none(val)
        except Exception:
            return None
        return None

    g.current_level_name = None
    g.current_level_path = None
    g.current_checkpoint_path = None
    sess = _load_session_cfg()
    viz_seed = args.eval_seed if args.eval_seed is not None else _parse_int_or_none(sess.get('eval_seed'))
    if sess.get('level_path') and os.path.isfile(sess['level_path']):
        g.current_level_path = sess['level_path']
        g.current_level_name = os.path.basename(sess['level_path'])
    if sess.get('checkpoint_path') and os.path.isfile(sess['checkpoint_path']):
        g.current_checkpoint_path = sess['checkpoint_path']
        if viz_seed is None:
            detected_seed = _detect_eval_seed_from_checkpoint(g.current_checkpoint_path)
            if detected_seed is not None:
                viz_seed = detected_seed
                _save_session_cfg(eval_seed=viz_seed)

    def pick_file_dialog(
            filetypes,
            fallback_exts=('.pth',),
            fallback_dirs=('.', 'logs'),
            fallback_title='Select file - Esc to cancel'):
        """Pick a file via Tk dialog, with pygame fallback when Tk is unavailable."""
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename(filetypes=filetypes)
            root.destroy()
            return path
        except Exception:
            def _pygame_file_picker(exts, search_dirs, title):
                files = []
                for directory in search_dirs:
                    if os.path.isdir(directory):
                        for root_dir, _, fns in os.walk(directory):
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
                                rect = pygame.Rect(20, 50 + i * 30, screen.get_width() - 40, 28)
                                if rect.collidepoint(mx, my):
                                    selected = fp
                                    running = False
                                    break
                    screen.fill((30, 30, 30))
                    title_surface = font_small.render(title, True, (255, 255, 255))
                    screen.blit(title_surface, (20, 10))
                    for i, fp in enumerate(files):
                        y = 50 + i * 30
                        color = (200, 200, 200) if i % 2 == 0 else (170, 170, 170)
                        pygame.draw.rect(screen, color, (18, y, screen.get_width() - 36, 28))
                        name_surface = font_small.render(os.path.basename(fp), True, (0, 0, 0))
                        screen.blit(name_surface, (25, y + 6))
                    pygame.display.flip()
                    pygame.time.wait(30)

                if created_temp:
                    pygame.display.quit()
                return selected

            return _pygame_file_picker(fallback_exts, fallback_dirs, fallback_title)

    def visualize_agent(
            env,
            max_episodes=10000,
            max_steps=MAX_EPISODE_MOVES,
            init_ckpt=None,
            seed_base=None):
        """Run evaluation-only visualization loop for a loaded checkpoint."""
        import torch as torch_mod

        try:
            import numpy as np
        except Exception:
            np = None

        agent = None
        agent_type = 'dqn'
        ckpt_board_size = None
        if init_ckpt:
            try:
                data = torch_mod.load(init_ckpt, map_location='cpu', weights_only=False)
                if isinstance(data, dict):
                    if 'board_size' in data:
                        agent_type = 'cnn'
                        ckpt_board_size = int(data['board_size'])
            except Exception:
                pass

        if ckpt_board_size and ckpt_board_size != env.board_blocks:
            env.board_blocks = ckpt_board_size
            env.layout_cfg['board_blocks'] = ckpt_board_size
            env._recompute_layout()

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
                    for i, line in enumerate(env._wrap_text(msg, env.font, env.w - 40)):
                        env.display.blit(env.font.render(line, True, WHITE), (10, 40 + i * 30))
                    pygame.display.flip()
                    info_timer -= 1
                    if env.clock:
                        env.clock.tick(30)
                return

        agent.eps = 0.0
        agent.policy_net.eval()

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
            action_mask = env.get_safe_action_mask()
            try:
                with torch_mod.no_grad():
                    st = torch_mod.as_tensor(cur_state, dtype=torch_mod.float32).unsqueeze(0).to(device)
                    q_out = agent.policy_net(st).squeeze(0).cpu().tolist()
                q_values = [float(v) for v in q_out]
                masked_q = [q if action_mask[i] else -1e9 for i, q in enumerate(q_values)]
                best = int(max(range(len(masked_q)), key=lambda i: masked_q[i]))
                return best, q_values
            except Exception:
                return agent.act(cur_state, action_mask=action_mask), None

        def _wait_for_resume(
                env,
                ep,
                score,
                total_reward,
                recent_scores,
                recent_steps,
                last_action,
                last_q_values,
                action_names,
                reason_text):
            while True:
                panel_bg, btn_pause, btn_plus, btn_minus, btn_stop = env._get_left_control_rects(panel_h=260)
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
                    n = min(200, len(recent_scores)) if recent_scores else 0
                    if n:
                        avg_s = sum(recent_scores[-200:]) / n
                        avg_t = sum(recent_steps[-200:]) / n
                        avg_txt = f'Avg200: {avg_s:.2f} / {avg_t:.1f}'
                    else:
                        avg_txt = 'Avg200: n/a'
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
                    for line in lines:
                        if y_off > max_y:
                            break
                        for wrapped_line in env._wrap_text(line, info_font, panel_bg.width - 12):
                            if y_off > max_y:
                                break
                            surf = info_font.render(wrapped_line, True, WHITE)
                            env.display.blit(surf, (panel_bg.x + 6, y_off))
                            y_off += line_h
                    pygame.draw.rect(env.display, (100, 180, 100), btn_pause)
                    pygame.draw.rect(env.display, PANEL_BORDER, btn_pause, 2)
                    env.display.blit(env.font.render('Resume', True, BLACK), (btn_pause.x + 8, btn_pause.y + 4))
                    for btn in (btn_plus, btn_minus):
                        pygame.draw.rect(env.display, (140, 140, 140), btn)
                        pygame.draw.rect(env.display, PANEL_BORDER, btn, 2)
                    env.display.blit(env.font.render('+', True, BLACK), (btn_plus.x + 10, btn_plus.y + 4))
                    env.display.blit(env.font.render('-', True, BLACK), (btn_minus.x + 12, btn_minus.y + 4))
                    pygame.draw.rect(env.display, (200, 80, 80), btn_stop)
                    pygame.draw.rect(env.display, PANEL_BORDER, btn_stop, 2)
                    env.display.blit(env.font.render('Stop', True, BLACK), (btn_stop.x + 8, btn_stop.y + 4))
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

            if seed_base is not None:
                seed_val = int(seed_base) + ep
                random.seed(seed_val)
                if np is not None:
                    np.random.seed(seed_val & 0xFFFFFFFF)
                try:
                    torch_mod.manual_seed(seed_val)
                    if torch_mod.cuda.is_available():
                        torch_mod.cuda.manual_seed_all(seed_val)
                except Exception:
                    pass

            state = env.reset()
            total_reward = 0.0
            paused = False
            panel_bg, btn_pause, btn_plus, btn_minus, btn_stop = env._get_left_control_rects(panel_h=260)

            last_action = None
            last_q_values = None
            last_abs_dir = None
            ep_steps = 0
            step_info = {}
            do_step = False

            for _ in range(max_steps):
                panel_bg, btn_pause, btn_plus, btn_minus, btn_stop = env._get_left_control_rects(panel_h=260)
                btn_step = pygame.Rect(btn_stop.x, btn_stop.bottom + 6, btn_stop.width, 32)

                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit(0)
                    if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                        running = False
                        break
                    if ev.type == pygame.VIDEORESIZE:
                        env.resize_window(ev.w, ev.h)
                        panel_bg, btn_pause, btn_plus, btn_minus, btn_stop = env._get_left_control_rects(panel_h=260)
                        btn_step = pygame.Rect(btn_stop.x, btn_stop.bottom + 6, btn_stop.width, 32)
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
                    next_state, reward, done, step_info = env.play_step(action, skip_events=True)
                    total_reward += reward
                    ep_steps += 1
                    last_abs_dir = env.direction.name
                    do_step = False
                else:
                    next_state, done = state, False

                state = next_state

                try:
                    env._draw_panel_box(panel_bg)
                    move_txt = last_abs_dir if last_abs_dir else 'n/a'
                    q_txt = format_q_values(last_q_values)
                    if recent_scores:
                        n = min(200, len(recent_scores))
                        avg_s = sum(recent_scores[-200:]) / n
                        avg_t = sum(recent_steps[-200:]) / n
                        avg_txt = f'Avg200: {avg_s:.2f} / {avg_t:.1f}'
                    else:
                        avg_txt = 'Avg200: n/a'

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
                    for line in lines:
                        if y_off > max_y:
                            break
                        for wrapped_line in env._wrap_text(line, info_font, panel_bg.width - 12):
                            if y_off > max_y:
                                break
                            surf = info_font.render(wrapped_line, True, WHITE)
                            env.display.blit(surf, (panel_bg.x + 6, y_off))
                            y_off += line_h

                    pygame.draw.rect(
                        env.display,
                        (180, 180, 100) if paused else (100, 180, 100),
                        btn_pause,
                    )
                    pygame.draw.rect(env.display, PANEL_BORDER, btn_pause, 2)
                    env.display.blit(
                        env.font.render('Pause' if not paused else 'Resume', True, BLACK),
                        (btn_pause.x + 8, btn_pause.y + 4),
                    )
                    for btn in (btn_plus, btn_minus):
                        pygame.draw.rect(env.display, (140, 140, 140), btn)
                        pygame.draw.rect(env.display, PANEL_BORDER, btn, 2)
                    env.display.blit(env.font.render('+', True, BLACK), (btn_plus.x + 10, btn_plus.y + 4))
                    env.display.blit(env.font.render('-', True, BLACK), (btn_minus.x + 12, btn_minus.y + 4))
                    pygame.draw.rect(env.display, (200, 80, 80), btn_stop)
                    pygame.draw.rect(env.display, PANEL_BORDER, btn_stop, 2)
                    env.display.blit(env.font.render('Stop', True, BLACK), (btn_stop.x + 8, btn_stop.y + 4))
                    if paused:
                        pygame.draw.rect(env.display, (80, 140, 200), btn_step)
                        pygame.draw.rect(env.display, PANEL_BORDER, btn_step, 2)
                        env.display.blit(env.font.render('Step', True, BLACK), (btn_step.x + 8, btn_step.y + 4))

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
                    _wait_for_resume(
                        env,
                        ep,
                        env.score,
                        total_reward,
                        recent_scores,
                        recent_steps,
                        last_action,
                        last_q_values,
                        action_names,
                        'Board filled!',
                    )
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
                        env,
                        ep,
                        env.score,
                        total_reward,
                        recent_scores,
                        recent_steps,
                        last_action,
                        last_q_values,
                        action_names,
                        reason_text,
                    )
                    if not resume:
                        running = False
                    break

            recent_scores.append(float(env.score))
            recent_steps.append(float(ep_steps))

        env.state_mode = 'features'
        env.reset()

    notif_msg = ''
    notif_frames = 0
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
                    btn1_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 - 120, btn_w, btn_h)
                    btn2_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 - 30, btn_w, btn_h)
                    btn3_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 + 60, btn_w, btn_h)
                    btn4_rect = pygame.Rect(center_x - btn_w // 2, g.h // 2 + 150, btn_w, btn_h)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    for tag, rect in [
                        ('1', btn1_rect),
                        ('2', btn2_rect),
                        ('3', btn3_rect),
                        ('4', btn4_rect),
                    ]:
                        if rect.collidepoint(mx, my):
                            choice = tag
                            menu_running = False
                            break

            g.display.fill(BLACK)
            title = g.font.render('Snake AI - Menu', True, WHITE)
            g.display.blit(title, (center_x - title.get_width() // 2, g.h // 2 - 160))
            for rect, label in [
                (btn1_rect, 'Level Design'),
                (btn2_rect, 'Visualization'),
                (btn3_rect, 'Load checkpoint (.pth)'),
                (btn4_rect, 'Load level (.json)'),
            ]:
                pygame.draw.rect(g.display, (50, 50, 50), rect)
                txt = g.font.render(label, True, WHITE)
                g.display.blit(txt, (rect.x + 12, rect.y + btn_h // 2 - txt.get_height() // 2))
            if g.current_level_name:
                s = g.font.render(f'Level: {g.current_level_name}', True, WHITE)
                g.display.blit(s, (10, 10))
            if g.current_checkpoint_path:
                s = g.font.render(
                    f'Checkpoint: {os.path.basename(g.current_checkpoint_path)}',
                    True,
                    WHITE,
                )
                g.display.blit(s, (10, 36))
            if notif_frames > 0:
                nf = g.small_font or g.font
                ns = nf.render(notif_msg, True, WHITE)
                nx, ny = 10, g.h - ns.get_height() - 24
                nw, nh = ns.get_width() + 20, ns.get_height() + 16
                pygame.draw.rect(g.display, (30, 30, 30), (nx, ny, nw, nh))
                pygame.draw.rect(g.display, (140, 140, 140), (nx, ny, nw, nh), 1)
                g.display.blit(ns, (nx + 10, ny + 8))
                notif_frames -= 1
            pygame.display.flip()
            if g.clock:
                g.clock.tick(30)

        if choice == '1':
            res = g.level_designer()
            if res is None:
                info_msg = 'Level edit cancelled.'
            else:
                g.current_level_name = os.path.basename(res)
                g.current_level_path = res
                info_msg = f'Saved level: {g.current_level_name}'
            notif_msg = info_msg
            notif_frames = 90

        elif choice == '3':
            path = pick_file_dialog(
                [('PyTorch', '*.pth'), ('All files', '*.*')],
                fallback_exts=('.pth',),
                fallback_dirs=('.', 'logs'),
                fallback_title='Select checkpoint (.pth)',
            )
            if path:
                g.current_checkpoint_path = path
                detected_seed = _detect_eval_seed_from_checkpoint(path)
                if args.eval_seed is None and detected_seed is not None:
                    viz_seed = detected_seed
                _save_session_cfg(checkpoint_path=path, eval_seed=viz_seed)
                if detected_seed is not None:
                    info_msg = f'Checkpoint: {os.path.basename(path)} | eval_seed={detected_seed}'
                else:
                    info_msg = f'Checkpoint: {os.path.basename(path)}'
            else:
                info_msg = 'No checkpoint selected.'
            notif_msg = info_msg
            notif_frames = 90

        elif choice == '4':
            path = pick_file_dialog(
                [('JSON level', '*.json'), ('All files', '*.*')],
                fallback_exts=('.json',),
                fallback_dirs=('levels', '.'),
                fallback_title='Select level (.json)',
            )
            if path:
                g.current_level_path = path
                g.current_level_name = os.path.basename(path)
                _save_session_cfg(level_path=path)
                info_msg = f'Level: {g.current_level_name}'
            else:
                info_msg = 'No level selected.'
            notif_msg = info_msg
            notif_frames = 90

        elif choice == '2':
            sub_w = 460
            sub_h = 350
            ckpt_board_size = None
            if g.current_checkpoint_path:
                try:
                    import torch as torch_mod
                    data = torch_mod.load(g.current_checkpoint_path, map_location='cpu', weights_only=False)
                    if isinstance(data, dict) and 'board_size' in data:
                        ckpt_board_size = int(data['board_size'])
                except Exception:
                    ckpt_board_size = None

            board_size_str = str(ckpt_board_size if ckpt_board_size else g.board_blocks)
            bs_cursor = len(board_size_str)
            use_seed = bool(viz_seed is not None)
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
                chk_seed_rect = pygame.Rect(sx + 20, sy + 128, 24, 24)
                btn_start = pygame.Rect(sx + 20, sy + 200, 200, 50)
                btn_back = pygame.Rect(sx + 240, sy + 200, 200, 50)

                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit(0)
                    if ev.type == pygame.VIDEORESIZE:
                        g.resize_window(ev.w, ev.h)
                    if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                        mx, my = ev.pos
                        if inp_bs_rect.collidepoint(mx, my):
                            active_input = 'boardsize'
                            blink_timer = 0
                        elif chk_seed_rect.collidepoint(mx, my):
                            if viz_seed is not None:
                                use_seed = not use_seed
                            active_input = None
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
                                    init_ckpt=g.current_checkpoint_path,
                                    seed_base=(viz_seed if use_seed else None),
                                )
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
                                board_size_str = board_size_str[:bs_cursor - 1] + board_size_str[bs_cursor:]
                                bs_cursor -= 1
                        elif ev.key == pygame.K_DELETE:
                            if bs_cursor < len(board_size_str):
                                board_size_str = board_size_str[:bs_cursor] + board_size_str[bs_cursor + 1:]
                        else:
                            ch = ev.unicode
                            if ch and ch.isdigit():
                                board_size_str = board_size_str[:bs_cursor] + ch + board_size_str[bs_cursor:]
                                bs_cursor += 1

                status_font = g.small_font or g.font
                g.display.fill((20, 20, 20))
                pygame.draw.rect(g.display, (60, 60, 60), sub_rect)
                title = g.font.render('Visualization', True, WHITE)
                g.display.blit(title, (sx + sub_w // 2 - title.get_width() // 2, sy + 8))

                bs_label = status_font.render('Board size:', True, WHITE)
                g.display.blit(bs_label, (sx + 20, sy + 60))
                bs_border = (255, 220, 50) if active_input == 'boardsize' else (150, 150, 150)
                pygame.draw.rect(g.display, (40, 40, 40), inp_bs_rect)
                pygame.draw.rect(g.display, bs_border, inp_bs_rect, 2)
                bs_txt = status_font.render(board_size_str, True, WHITE)
                g.display.blit(bs_txt, (inp_bs_rect.x + 6, inp_bs_rect.y + 6))
                if active_input == 'boardsize' and show_cursor:
                    cursor_x = inp_bs_rect.x + 6 + status_font.size(board_size_str[:bs_cursor])[0]
                    pygame.draw.line(g.display, WHITE, (cursor_x, inp_bs_rect.y + 4), (cursor_x, inp_bs_rect.bottom - 4))

                chk_border = (150, 150, 150)
                chk_fill = (40, 40, 40)
                if viz_seed is None:
                    chk_fill = (28, 28, 28)
                    chk_border = (90, 90, 90)
                pygame.draw.rect(g.display, chk_fill, chk_seed_rect)
                pygame.draw.rect(g.display, chk_border, chk_seed_rect, 2)
                if use_seed and viz_seed is not None:
                    pygame.draw.line(
                        g.display, WHITE,
                        (chk_seed_rect.x + 5, chk_seed_rect.y + 12),
                        (chk_seed_rect.x + 10, chk_seed_rect.y + 18), 2)
                    pygame.draw.line(
                        g.display, WHITE,
                        (chk_seed_rect.x + 10, chk_seed_rect.y + 18),
                        (chk_seed_rect.x + 19, chk_seed_rect.y + 6), 2)
                seed_label = (
                    f'Deterministic eval seed: {viz_seed}'
                    if viz_seed is not None
                    else 'Deterministic eval seed: not available'
                )
                seed_color = WHITE if viz_seed is not None else (150, 150, 150)
                seed_surf = status_font.render(seed_label, True, seed_color)
                g.display.blit(seed_surf, (chk_seed_rect.right + 8, chk_seed_rect.y + 1))

                level_status = (
                    f'Level: {g.current_level_name}'
                    if g.current_level_name
                    else 'Level: empty map (classic snake)'
                )
                ckpt_status = (
                    f'Checkpoint: {os.path.basename(g.current_checkpoint_path)}'
                    if g.current_checkpoint_path
                    else 'No checkpoint loaded'
                )
                info_y = sy + 168
                for line in [level_status, ckpt_status]:
                    clipped = g._fit_text(line, status_font, sub_w - 24)
                    surf = status_font.render(clipped, True, WHITE)
                    g.display.blit(surf, (sx + 12, info_y))
                    info_y += status_font.get_height() + 4

                pygame.draw.rect(g.display, (80, 200, 120), btn_start)
                pygame.draw.rect(g.display, (200, 80, 80), btn_back)
                g.display.blit(g.font.render('Start', True, BLACK), (btn_start.x + 62, btn_start.y + 12))
                g.display.blit(g.font.render('Back', True, BLACK), (btn_back.x + 70, btn_back.y + 14))

                pygame.display.flip()
                if g.clock:
                    g.clock.tick(30)

            if 'info_msg' in locals() and info_msg:
                notif_msg = info_msg
                notif_frames = 90


if __name__ == '__main__':
    run_cli()
