import datetime
import json
import os

import pygame


def _make_point(self, x, y):
    factory = getattr(self, 'point_factory', None)
    if factory is not None:
        return factory(int(x), int(y))
    return (int(x), int(y))


def level_designer(self):
    """Prosty edytor poziomow: kliknij, aby dodac/usunac blok. Nacisnij Enter, aby zakonczyc."""
    if not self.render:
        print('Level designer wymaga trybu render (GUI).')
        return set()
    if not hasattr(self, 'walls'):
        self.walls = set()

    theme = getattr(self, 'theme', {})
    white = theme.get('white', (255, 255, 255))
    panel_border = theme.get('panel_border', (115, 115, 115))

    designing = True
    status_msg = ''
    status_timer = 0
    back_rect = pygame.Rect(self.left_panel_rect.x + 10, self.left_panel_rect.y + 10, 90, 34)

    while designing:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            if event.type == pygame.VIDEORESIZE:
                self.resize_window(event.w, event.h, preserve_state=True)
                back_rect = pygame.Rect(self.left_panel_rect.x + 10, self.left_panel_rect.y + 10, 90, 34)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    name = ''
                    entering = True
                    input_w = max(280, min(self.left_panel_rect.width - 40, 520))
                    input_rect = pygame.Rect(
                        self.left_panel_rect.x + 20,
                        self.left_panel_rect.centery - 20,
                        input_w,
                        40,
                    )
                    while entering:
                        for ie in pygame.event.get():
                            if ie.type == pygame.QUIT:
                                pygame.quit()
                                return None
                            if ie.type == pygame.VIDEORESIZE:
                                self.resize_window(ie.w, ie.h, preserve_state=True)
                                back_rect = pygame.Rect(self.left_panel_rect.x + 10, self.left_panel_rect.y + 10, 90, 34)
                                input_w = max(280, min(self.left_panel_rect.width - 40, 520))
                                input_rect = pygame.Rect(
                                    self.left_panel_rect.x + 20,
                                    self.left_panel_rect.centery - 20,
                                    input_w,
                                    40,
                                )
                            if ie.type == pygame.KEYDOWN:
                                if ie.key == pygame.K_RETURN:
                                    entering = False
                                    break
                                if ie.key == pygame.K_ESCAPE:
                                    entering = False
                                    name = ''
                                    break
                                if ie.key == pygame.K_BACKSPACE:
                                    name = name[:-1]
                                else:
                                    ch = ie.unicode
                                    if ch and len(name) < 64:
                                        name += ch

                        try:
                            self._update_ui()
                            self._draw_panel_box(input_rect)
                            prompt_raw = 'Level name (Enter save, Esc cancel): ' + name
                            prompt_line = self._fit_text(prompt_raw, self.small_font or self.font, input_rect.width - 12)
                            prompt = (self.small_font or self.font).render(prompt_line, True, white)
                            self.display.blit(prompt, (input_rect.x + 6, input_rect.y + 10))
                            self._draw_footer_block([
                                'Type level name and press Enter to save',
                                'Esc cancels naming and exits designer without save',
                                'Window resize keeps current wall layout aligned'
                            ])
                            pygame.display.flip()
                        except Exception:
                            pass
                        if self.clock:
                            self.clock.tick(30)

                    if not name:
                        return None

                    levels_dir = os.path.join('levels')
                    os.makedirs(levels_dir, exist_ok=True)
                    safe = ''.join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                    level_name = f'{safe}_{ts}.json'
                    level_path = os.path.join(levels_dir, level_name)
                    try:
                        data = [[p.x - self.board_x, p.y - self.board_y] for p in sorted(self.walls, key=lambda x: (x.x, x.y))]
                        data_cells = [[int(x // self.bs), int(y // self.bs)] for x, y in data]
                        with open(level_path, 'w', encoding='utf-8') as f:
                            json.dump(data_cells, f)
                        status_msg = f'Saved {level_name}'
                    except Exception as e:
                        status_msg = f'Save error: {e}'
                    status_timer = 60
                    return level_path

                if event.key == pygame.K_s:
                    try:
                        data_cells = []
                        for p in sorted(self.walls, key=lambda x: (x.x, x.y)):
                            cx = int((p.x - self.board_x) // self.bs)
                            cy = int((p.y - self.board_y) // self.bs)
                            data_cells.append([cx, cy])
                        with open('level.json', 'w', encoding='utf-8') as f:
                            json.dump(data_cells, f)
                        status_msg = 'Saved level.json'
                    except Exception as e:
                        status_msg = f'Save error: {e}'
                    status_timer = 30

                if event.key == pygame.K_l:
                    try:
                        with open('level.json', 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        conv = set()
                        for item in data:
                            try:
                                cx, cy = int(item[0]), int(item[1])
                                px = self.board_x + cx * self.bs
                                py = self.board_y + cy * self.bs
                                conv.add(_make_point(self, px, py))
                            except Exception:
                                pass
                        self.walls = conv
                        status_msg = 'Loaded level.json'
                    except Exception as e:
                        status_msg = f'Load error: {e}'
                    status_timer = 30

                if event.key == pygame.K_ESCAPE or event.key == pygame.K_BACKSPACE:
                    return None

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if back_rect.collidepoint(mx, my):
                    return None
                if mx >= self.board_x and mx < self.board_x + self.board_w and my >= self.board_y and my < self.board_y + self.board_h:
                    cell_x = (mx - self.board_x) // self.bs
                    cell_y = (my - self.board_y) // self.bs
                    gx = self.board_x + cell_x * self.bs
                    gy = self.board_y + cell_y * self.bs
                    p = _make_point(self, gx, gy)
                    if p in self.walls:
                        self.walls.remove(p)
                    else:
                        self.walls.add(p)

        self._update_ui()
        back_rect = pygame.Rect(self.left_panel_rect.x + 10, self.left_panel_rect.y + 10, 90, 34)
        pygame.draw.rect(self.display, (80, 80, 80), back_rect)
        pygame.draw.rect(self.display, panel_border, back_rect, 2)
        back_font = self.small_font or self.font
        back_s = back_font.render('Back', True, white)
        self.display.blit(back_s, (back_rect.x + 10, back_rect.y + 8))

        footer_lines = [
            'Designer: LMB toggle wall | Enter save',
            'Esc/Back: cancel | S: temp save | L: temp load',
            f'Status: {status_msg if status_timer > 0 and status_msg else "-"}'
        ]
        self._draw_footer_block(footer_lines)

        if status_timer > 0:
            status_timer -= 1
        pygame.display.flip()
        if self.clock:
            self.clock.tick(30)

    return self.walls
