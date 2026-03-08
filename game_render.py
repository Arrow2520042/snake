import pygame


def update_ui(self):
    if not self.render:
        return

    theme = getattr(self, 'theme', {})
    white = theme.get('white', (255, 255, 255))
    red = theme.get('red', (200, 0, 0))
    blue1 = theme.get('blue1', (0, 0, 255))
    blue2 = theme.get('blue2', (0, 100, 255))
    black = theme.get('black', (0, 0, 0))
    panel_bg = theme.get('panel_bg', (24, 24, 24))
    panel_border = theme.get('panel_border', (115, 115, 115))
    board_border = theme.get('board_border', (185, 185, 185))
    footer_bg = theme.get('footer_bg', (28, 28, 28))
    footer_border = theme.get('footer_border', (105, 105, 105))

    disp = self.display
    bs = self.bs
    disp.fill(black)

    for pt in self.snake:
        if pt.x < self.board_x or pt.x >= self.board_x + self.board_w:
            continue
        r_outer = pygame.Rect(pt.x, pt.y, bs, bs)
        r_inner = pygame.Rect(pt.x + 4, pt.y + 4, bs - 8, bs - 8)
        pygame.draw.rect(disp, blue1, r_outer)
        pygame.draw.rect(disp, blue2, r_inner)

    pygame.draw.rect(disp, red, pygame.Rect(self.food.x, self.food.y, bs, bs))

    walls = getattr(self, 'walls', None)
    if walls:
        for w in walls:
            pygame.draw.rect(disp, (100, 100, 100), pygame.Rect(w.x, w.y, bs, bs))

    board_rect = pygame.Rect(self.board_x, self.board_y, self.board_w, self.board_h)
    pygame.draw.rect(disp, board_border, board_rect, 2)

    try:
        disp.fill(panel_bg, rect=self.left_panel_rect)
        pygame.draw.rect(disp, panel_border, self.left_panel_rect, 2)

        disp.fill(footer_bg, rect=self.footer_rect)
        pygame.draw.rect(disp, footer_border, self.footer_rect, 2)
    except Exception:
        pass

    if hasattr(self, 'generation'):
        gen_text = self.font.render(f'Gen: {self.generation}', True, white)
        gr = pygame.Rect(self.w - gen_text.get_width() - 12, 4, gen_text.get_width() + 8, gen_text.get_height() + 4)
        disp.fill(black, rect=gr)
        disp.blit(gen_text, (gr.x + 4, gr.y))


def draw_panel_box(self, rect):
    if not self.render:
        return
    theme = getattr(self, 'theme', {})
    panel_bg = theme.get('panel_bg', (24, 24, 24))
    panel_border = theme.get('panel_border', (115, 115, 115))
    self.display.fill(panel_bg, rect=rect)
    pygame.draw.rect(self.display, panel_border, rect, 2)


def fit_text(self, text, font, max_width):
    t = str(text)
    if font is None:
        return t
    if font.size(t)[0] <= max_width:
        return t
    ellipsis = '...'
    if font.size(ellipsis)[0] > max_width:
        return ''
    while t and font.size(t + ellipsis)[0] > max_width:
        t = t[:-1]
    return t + ellipsis


def wrap_text(self, text, font, max_width):
    if font is None:
        return [str(text)]
    src = str(text)
    words = src.split()
    if not words:
        return ['']
    lines = []
    current = ''
    for word in words:
        candidate = word if not current else f'{current} {word}'
        if font.size(candidate)[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
                current = word
            else:
                lines.append(self._fit_text(word, font, max_width))
                current = ''
    if current:
        lines.append(current)
    return lines


def draw_footer_block(self, lines):
    if not self.render:
        return None

    theme = getattr(self, 'theme', {})
    white = theme.get('white', (255, 255, 255))
    footer_bg = theme.get('footer_bg', (28, 28, 28))
    footer_border = theme.get('footer_border', (105, 105, 105))

    footer_rect = self.footer_rect
    self.display.fill(footer_bg, rect=footer_rect)
    pygame.draw.rect(self.display, footer_border, footer_rect, 2)
    footer_font = self.small_font or self.font
    line_h = (footer_font.get_height() + 4) if footer_font else 20
    max_lines = max(1, (footer_rect.height - 12) // line_h)

    wrapped = []
    for line in lines:
        wrapped.extend(self._wrap_text(line, footer_font, footer_rect.width - 16))

    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        wrapped[-1] = self._fit_text(wrapped[-1], footer_font, footer_rect.width - 16)

    for i, line in enumerate(wrapped):
        clipped = self._fit_text(line, footer_font, footer_rect.width - 16)
        txt = footer_font.render(clipped, True, white)
        self.display.blit(txt, (footer_rect.x + 8, footer_rect.y + 6 + i * line_h))
    return footer_rect
