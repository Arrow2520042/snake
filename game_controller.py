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
import traceback
from typing import cast

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

    # From here onward this CLI branch is GUI-only.
    if g.display is None or g.font is None:
        raise RuntimeError('Render mode requires initialized display and fonts.')
    g.display = cast(pygame.Surface, g.display)
    g.font = cast(pygame.font.Font, g.font)
    if g.small_font is None:
        g.small_font = g.font
    else:
        g.small_font = cast(pygame.font.Font, g.small_font)

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

    def _parse_float_or_none(value):
        try:
            return float(value)
        except Exception:
            return None

    def _extract_int_key_from_metadata(meta_path, key_name):
        if not meta_path or not os.path.isfile(meta_path):
            return None
        key_low = str(key_name).lower()
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    low = line.lower()
                    if not low.startswith(key_low):
                        continue
                    if ':' in line:
                        val = line.split(':', 1)[1].strip()
                    elif '=' in line:
                        val = line.split('=', 1)[1].strip()
                    else:
                        parts = line.split()
                        val = parts[-1] if len(parts) > 1 else ''
                    return _parse_int_or_none(val)
        except Exception:
            return None
        return None

    def _extract_float_key_from_metadata(meta_path, key_name):
        if not meta_path or not os.path.isfile(meta_path):
            return None
        key_low = str(key_name).lower()
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    low = line.lower()
                    if not low.startswith(key_low):
                        continue
                    if ':' in line:
                        val = line.split(':', 1)[1].strip()
                    elif '=' in line:
                        val = line.split('=', 1)[1].strip()
                    else:
                        parts = line.split()
                        val = parts[-1] if len(parts) > 1 else ''
                    return _parse_float_or_none(val)
        except Exception:
            return None
        return None

    def _extract_eval_metadata_from_checkpoint(ckpt_path):
        """Read eval metadata directly from checkpoint payload when available."""
        if not ckpt_path or not os.path.isfile(ckpt_path):
            return None
        try:
            import torch as torch_mod

            data = torch_mod.load(ckpt_path, map_location='cpu', weights_only=False)
            if not isinstance(data, dict):
                return None
            meta = data.get('metadata')
            if not isinstance(meta, dict):
                return None
            out = {}
            if 'eval_seed' in meta:
                out['eval_seed'] = _parse_int_or_none(meta.get('eval_seed'))
            if 'best_eval_single_seed' in meta:
                out['best_eval_single_seed'] = _parse_int_or_none(meta.get('best_eval_single_seed'))
            if 'best_eval_single_score' in meta:
                out['best_eval_single_score'] = _parse_float_or_none(meta.get('best_eval_single_score'))
            return out if out else None
        except Exception:
            return None

    def _extract_eval_seed_from_metadata(meta_path):
        return _extract_int_key_from_metadata(meta_path, 'eval_seed')

    def _detect_eval_seed_from_checkpoint(ckpt_path):
        """Infer eval_seed + best single eval seed from checkpoint or nearby metadata."""
        if not ckpt_path or not os.path.isfile(ckpt_path):
            return None, None, None, None

        ckpt_meta = _extract_eval_metadata_from_checkpoint(ckpt_path)
        if ckpt_meta:
            eval_seed_val = _parse_int_or_none(ckpt_meta.get('eval_seed'))
            single_seed_val = _parse_int_or_none(ckpt_meta.get('best_eval_single_seed'))
            single_score_val = _parse_float_or_none(ckpt_meta.get('best_eval_single_score'))
            if eval_seed_val is not None or single_seed_val is not None:
                return eval_seed_val, single_seed_val, single_score_val, 'checkpoint metadata'

        candidate_dirs = [os.path.dirname(ckpt_path)]
        ws_root = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(ws_root, 'logs')
        ckpt_name = os.path.basename(ckpt_path)
        if os.path.isdir(logs_dir):
            try:
                for root_dir, _, files in os.walk(logs_dir):
                    if ckpt_name in files:
                        candidate_dirs.append(root_dir)
            except Exception:
                pass

        seen = set()
        for cand_dir in candidate_dirs:
            norm = os.path.normcase(os.path.abspath(cand_dir))
            if norm in seen:
                continue
            seen.add(norm)
            for meta_name in ('info.txt', 'run_info.txt'):
                meta_path = os.path.join(cand_dir, meta_name)
                seed = _extract_eval_seed_from_metadata(meta_path)
                single_seed = _extract_int_key_from_metadata(meta_path, 'best_eval_single_seed')
                single_score = _extract_float_key_from_metadata(meta_path, 'best_eval_single_score')
                if seed is not None or single_seed is not None:
                    return seed, single_seed, single_score, meta_path
        return None, None, None, None

    g.current_level_name = None
    g.current_level_path = None
    g.current_checkpoint_path = None
    sess = _load_session_cfg()
    viz_seed = args.eval_seed if args.eval_seed is not None else _parse_int_or_none(sess.get('eval_seed'))
    best_eval_single_seed = _parse_int_or_none(sess.get('best_eval_single_seed'))
    if sess.get('level_path') and os.path.isfile(sess['level_path']):
        g.current_level_path = sess['level_path']
        g.current_level_name = os.path.basename(sess['level_path'])
    if sess.get('checkpoint_path') and os.path.isfile(sess['checkpoint_path']):
        g.current_checkpoint_path = sess['checkpoint_path']
        if viz_seed is None or best_eval_single_seed is None:
            detected_seed, detected_single_seed, _, _ = _detect_eval_seed_from_checkpoint(g.current_checkpoint_path)
            if detected_seed is not None:
                viz_seed = detected_seed
            if detected_single_seed is not None:
                best_eval_single_seed = detected_single_seed
                if viz_seed is None:
                    viz_seed = detected_single_seed
            _save_session_cfg(eval_seed=viz_seed, best_eval_single_seed=best_eval_single_seed)

    def pick_file_dialog(
            filetypes,
            fallback_exts=('.pth',),
            fallback_dirs=None,
            fallback_title='Select file - Esc to cancel'):
        """Pick a file via Tk dialog, with pygame fallback when Tk is unavailable."""
        project_root = os.path.dirname(os.path.abspath(__file__))

        if fallback_dirs is None:
            fallback_dirs = (project_root,)
        else:
            normalized_dirs = []
            for directory in fallback_dirs:
                if os.path.isabs(directory):
                    normalized_dirs.append(directory)
                else:
                    normalized_dirs.append(os.path.abspath(os.path.join(project_root, directory)))
            if project_root not in normalized_dirs:
                normalized_dirs.insert(0, project_root)
            fallback_dirs = tuple(normalized_dirs)

        start_dir = next((d for d in fallback_dirs if os.path.isdir(d)), project_root)

        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename(filetypes=filetypes, initialdir=start_dir)
            root.destroy()
            return path
        except Exception:
            def _pygame_file_picker(exts, initial_dir, title):
                ext_lows = [e.lower() for e in exts]

                def _list_entries(directory):
                    entries = []
                    try:
                        names = sorted(os.listdir(directory), key=lambda n: n.lower())
                    except Exception:
                        names = []

                    parent = os.path.dirname(directory)
                    if parent and parent != directory:
                        entries.append(('up', '..', parent))

                    for name in names:
                        full = os.path.join(directory, name)
                        if os.path.isdir(full):
                            entries.append(('dir', name, full))
                    for name in names:
                        full = os.path.join(directory, name)
                        if os.path.isfile(full) and any(name.lower().endswith(e) for e in ext_lows):
                            entries.append(('file', name, full))
                    return entries

                screen = pygame.display.get_surface()
                created_temp = False
                if screen is None:
                    pygame.display.init()
                    screen = pygame.display.set_mode((640, 400))
                    created_temp = True

                font_small = pygame.font.SysFont(None, 20)
                running = True
                selected = ''
                list_top = 50
                row_h = 30
                scroll = 0
                current_dir = initial_dir
                entries = _list_entries(current_dir)

                def _max_visible_rows():
                    return max(1, (screen.get_height() - list_top - 44) // row_h)

                def _max_scroll():
                    return max(0, len(entries) - _max_visible_rows())

                def _clamp_scroll():
                    nonlocal scroll
                    scroll = max(0, min(scroll, _max_scroll()))

                def _refresh_directory(new_dir):
                    nonlocal current_dir, entries, scroll
                    current_dir = new_dir
                    entries = _list_entries(current_dir)
                    scroll = 0

                def _go_parent():
                    parent = os.path.dirname(current_dir)
                    if parent and parent != current_dir:
                        _refresh_directory(parent)

                while running:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            running = False
                        if ev.type == pygame.KEYDOWN:
                            if ev.key == pygame.K_ESCAPE:
                                running = False
                            elif ev.key == pygame.K_DOWN:
                                scroll += 1
                                _clamp_scroll()
                            elif ev.key == pygame.K_UP:
                                scroll -= 1
                                _clamp_scroll()
                            elif ev.key == pygame.K_PAGEDOWN:
                                scroll += max(1, _max_visible_rows() - 1)
                                _clamp_scroll()
                            elif ev.key == pygame.K_PAGEUP:
                                scroll -= max(1, _max_visible_rows() - 1)
                                _clamp_scroll()
                            elif ev.key == pygame.K_HOME:
                                scroll = 0
                            elif ev.key == pygame.K_END:
                                scroll = _max_scroll()
                            elif ev.key == pygame.K_BACKSPACE:
                                _go_parent()
                        if ev.type == pygame.MOUSEWHEEL:
                            scroll -= int(ev.y)
                            _clamp_scroll()
                        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                            mx, my = ev.pos
                            visible_rows = _max_visible_rows()
                            for i in range(visible_rows):
                                idx = scroll + i
                                if idx >= len(entries):
                                    break
                                etype, _, target = entries[idx]
                                rect = pygame.Rect(20, list_top + i * row_h, screen.get_width() - 52, 28)
                                if rect.collidepoint(mx, my):
                                    if etype in ('dir', 'up'):
                                        _refresh_directory(target)
                                    elif etype == 'file':
                                        selected = target
                                        running = False
                                    break
                        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (4, 5):
                            scroll += -3 if ev.button == 4 else 3
                            _clamp_scroll()

                    _clamp_scroll()
                    screen.fill((30, 30, 30))
                    title_surface = font_small.render(title, True, (255, 255, 255))
                    screen.blit(title_surface, (20, 10))

                    path_surface = font_small.render(current_dir, True, (200, 200, 200))
                    screen.blit(path_surface, (20, 28))

                    hint_surface = font_small.render('Enter folder: click | Back: Backspace | Scroll: wheel/arrows/PgUp-PgDn', True, (185, 185, 185))
                    screen.blit(hint_surface, (20, screen.get_height() - 18))

                    visible_rows = _max_visible_rows()
                    for i in range(visible_rows):
                        idx = scroll + i
                        if idx >= len(entries):
                            break
                        etype, name, _ = entries[idx]
                        y = list_top + i * row_h
                        color = (200, 200, 200) if i % 2 == 0 else (170, 170, 170)
                        pygame.draw.rect(screen, color, (18, y, screen.get_width() - 52, 28))
                        if etype == 'up':
                            label = '[..] Parent directory'
                        elif etype == 'dir':
                            label = f'[DIR] {name}'
                        else:
                            label = name
                        name_surface = font_small.render(label, True, (0, 0, 0))
                        screen.blit(name_surface, (25, y + 6))

                    if len(entries) > visible_rows:
                        bar_x = screen.get_width() - 24
                        bar_y = list_top
                        bar_h = visible_rows * row_h
                        pygame.draw.rect(screen, (80, 80, 80), (bar_x, bar_y, 8, bar_h))
                        max_scroll = _max_scroll()
                        thumb_h = max(20, int(bar_h * (visible_rows / float(len(entries)))))
                        if max_scroll > 0:
                            thumb_y = bar_y + int((bar_h - thumb_h) * (scroll / float(max_scroll)))
                        else:
                            thumb_y = bar_y
                        pygame.draw.rect(screen, (180, 180, 180), (bar_x, thumb_y, 8, thumb_h))

                    pygame.display.flip()
                    pygame.time.wait(30)

                if created_temp:
                    pygame.display.quit()
                return selected

            return _pygame_file_picker(fallback_exts, start_dir, fallback_title)

    def visualize_agent(
            env,
            max_episodes=10000,
            max_steps=MAX_EPISODE_MOVES,
            init_ckpt=None,
            seed_base=None,
            single_episode_only=False):
        """Run evaluation-only visualization loop for a loaded checkpoint."""
        import torch as torch_mod

        if not env.render or env.display is None or env.font is None:
            raise RuntimeError('Visualization requires render-enabled environment.')
        env.display = cast(pygame.Surface, env.display)
        env.font = cast(pygame.font.Font, env.font)
        if env.small_font is None:
            env.small_font = env.font
        else:
            env.small_font = cast(pygame.font.Font, env.small_font)

        try:
            import numpy as np
        except Exception:
            np = None

        agent = None
        agent_type = 'cnn'
        ckpt_board_size = None
        if init_ckpt:
            try:
                data = torch_mod.load(init_ckpt, map_location='cpu', weights_only=False)
                if isinstance(data, dict) and 'board_size' in data:
                    ckpt_board_size = int(data['board_size'])
            except Exception:
                pass

        if ckpt_board_size and ckpt_board_size != env.board_blocks:
            env.board_blocks = ckpt_board_size
            env.layout_cfg['board_blocks'] = ckpt_board_size
            env._recompute_layout()

        from cnn_agent import CNNAgent
        agent = CNNAgent(board_size=env.board_blocks)
        env.state_mode = 'grid'

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
        else:
            # Avoid stale walls from previous level sessions when replaying checkpoints.
            env.walls = set()

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
            # Use the same action path as training eval to preserve RNG consumption
            # and deterministic food placement sequence for seeded replays.
            action = int(agent.act(cur_state, action_mask=action_mask))
            try:
                with torch_mod.no_grad():
                    st = torch_mod.as_tensor(cur_state, dtype=torch_mod.float32).unsqueeze(0).to(device)
                    q_out = agent.policy_net(st).squeeze(0).cpu().tolist()
                q_values = [float(v) for v in q_out]
                return action, q_values
            except Exception:
                return action, None

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
                if single_episode_only:
                    seed_val = int(seed_base)
                else:
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

        env.state_mode = 'grid'
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
                detected_seed, detected_single_seed, detected_single_score, seed_source = _detect_eval_seed_from_checkpoint(path)
                if args.eval_seed is None and detected_seed is not None:
                    viz_seed = detected_seed
                if detected_single_seed is not None:
                    best_eval_single_seed = detected_single_seed
                    viz_seed = detected_single_seed
                else:
                    best_eval_single_seed = None
                _save_session_cfg(
                    checkpoint_path=path,
                    eval_seed=viz_seed,
                    best_eval_single_seed=best_eval_single_seed,
                )
                if detected_seed is not None or detected_single_seed is not None:
                    src_name = os.path.basename(seed_source) if isinstance(seed_source, str) and os.path.isfile(seed_source) else str(seed_source)
                    if detected_single_seed is not None:
                        score_txt = f'{detected_single_score:.2f}' if detected_single_score is not None else 'n/a'
                        info_msg = (
                            f'Checkpoint: {os.path.basename(path)} | '
                            f'single_seed={detected_single_seed} score={score_txt} ({src_name})'
                        )
                    else:
                        info_msg = (
                            f'Checkpoint: {os.path.basename(path)} | '
                            f'eval_seed={detected_seed} ({src_name})'
                        )
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
            sub_h = 430
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
            default_seed = best_eval_single_seed if best_eval_single_seed is not None else viz_seed
            seed_str = '' if default_seed is None else str(default_seed)
            seed_cursor = len(seed_str)
            use_seed = bool(default_seed is not None)
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
                inp_seed_rect = pygame.Rect(sx + 20, sy + 172, 420, 32)
                chk_seed_rect = pygame.Rect(sx + 20, sy + 220, 24, 24)
                btn_start = pygame.Rect(sx + 20, sy + 338, 200, 50)
                btn_back = pygame.Rect(sx + 240, sy + 338, 200, 50)
                seed_value = _parse_int_or_none(seed_str)
                if seed_value is None:
                    use_seed = False

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
                        elif inp_seed_rect.collidepoint(mx, my):
                            active_input = 'seed'
                            blink_timer = 0
                        elif chk_seed_rect.collidepoint(mx, my):
                            if seed_value is not None:
                                use_seed = not use_seed
                            active_input = None
                        elif btn_start.collidepoint(mx, my):
                            active_input = None
                            bsize = max(5, int(board_size_str)) if board_size_str.isdigit() else g.board_blocks
                            if bsize != g.board_blocks:
                                g.board_blocks = bsize
                                g.layout_cfg['board_blocks'] = bsize
                                g._recompute_layout()
                            chosen_seed = _parse_int_or_none(seed_str)
                            if chosen_seed is not None:
                                viz_seed = chosen_seed
                            if chosen_seed is None and not use_seed:
                                viz_seed = None
                            run_single_episode = (
                                bool(use_seed)
                                and chosen_seed is not None
                                and best_eval_single_seed is not None
                                and int(chosen_seed) == int(best_eval_single_seed)
                            )
                            _save_session_cfg(eval_seed=viz_seed, best_eval_single_seed=best_eval_single_seed)
                            try:
                                visualize_agent(
                                    g,
                                    max_episodes=(1 if run_single_episode else 10000),
                                    max_steps=MAX_EPISODE_MOVES,
                                    init_ckpt=g.current_checkpoint_path,
                                    seed_base=(chosen_seed if use_seed and chosen_seed is not None else None),
                                    single_episode_only=run_single_episode,
                                )
                                if run_single_episode:
                                    info_msg = f'Visualization finished (replayed best_eval_single seed={chosen_seed}).'
                                elif use_seed and chosen_seed is not None:
                                    info_msg = f'Visualization finished (seed={chosen_seed}).'
                                else:
                                    info_msg = 'Visualization finished (seed disabled).'
                            except Exception as e:
                                traceback.print_exc()
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
                        elif active_input == 'boardsize':
                            if ev.key == pygame.K_LEFT:
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
                        elif active_input == 'seed':
                            if ev.key == pygame.K_LEFT:
                                seed_cursor = max(0, seed_cursor - 1)
                            elif ev.key == pygame.K_RIGHT:
                                seed_cursor = min(len(seed_str), seed_cursor + 1)
                            elif ev.key == pygame.K_HOME:
                                seed_cursor = 0
                            elif ev.key == pygame.K_END:
                                seed_cursor = len(seed_str)
                            elif ev.key == pygame.K_BACKSPACE:
                                if seed_cursor > 0:
                                    seed_str = seed_str[:seed_cursor - 1] + seed_str[seed_cursor:]
                                    seed_cursor -= 1
                            elif ev.key == pygame.K_DELETE:
                                if seed_cursor < len(seed_str):
                                    seed_str = seed_str[:seed_cursor] + seed_str[seed_cursor + 1:]
                            else:
                                ch = ev.unicode
                                if ch and ch.isdigit():
                                    seed_str = seed_str[:seed_cursor] + ch + seed_str[seed_cursor:]
                                    seed_cursor += 1
                        else:
                            active_input = None

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

                seed_label_top = status_font.render('Deterministic eval seed (optional):', True, WHITE)
                g.display.blit(seed_label_top, (sx + 20, sy + 150))
                seed_border = (255, 220, 50) if active_input == 'seed' else (150, 150, 150)
                pygame.draw.rect(g.display, (40, 40, 40), inp_seed_rect)
                pygame.draw.rect(g.display, seed_border, inp_seed_rect, 2)
                seed_txt = status_font.render(seed_str, True, WHITE)
                g.display.blit(seed_txt, (inp_seed_rect.x + 6, inp_seed_rect.y + 6))
                if active_input == 'seed' and show_cursor:
                    seed_px = inp_seed_rect.x + 6 + status_font.size(seed_str[:seed_cursor])[0]
                    pygame.draw.line(g.display, WHITE, (seed_px, inp_seed_rect.y + 4), (seed_px, inp_seed_rect.bottom - 4))

                chk_border = (150, 150, 150)
                chk_fill = (40, 40, 40)
                if seed_value is None:
                    chk_fill = (28, 28, 28)
                    chk_border = (90, 90, 90)
                pygame.draw.rect(g.display, chk_fill, chk_seed_rect)
                pygame.draw.rect(g.display, chk_border, chk_seed_rect, 2)
                if use_seed and seed_value is not None:
                    pygame.draw.line(
                        g.display, WHITE,
                        (chk_seed_rect.x + 5, chk_seed_rect.y + 12),
                        (chk_seed_rect.x + 10, chk_seed_rect.y + 18), 2)
                    pygame.draw.line(
                        g.display, WHITE,
                        (chk_seed_rect.x + 10, chk_seed_rect.y + 18),
                        (chk_seed_rect.x + 19, chk_seed_rect.y + 6), 2)
                seed_label = (
                    f'Use seed in visualization ({seed_value})'
                    if seed_value is not None
                    else 'Use seed in visualization (enter numeric seed)'
                )
                seed_color = WHITE if seed_value is not None else (150, 150, 150)
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
                single_seed_status = (
                    f'Best eval single seed: {best_eval_single_seed} (auto 1-episode replay)'
                    if best_eval_single_seed is not None
                    else 'Best eval single seed: n/a'
                )
                info_y = sy + 260
                for line in [level_status, ckpt_status, single_seed_status]:
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
