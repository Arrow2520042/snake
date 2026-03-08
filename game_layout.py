import pygame


def recompute_layout(self):
    """Recompute pixel sizes and positions from window dimensions.
    Game state uses cell coordinates so no entity remapping is needed.
    """
    cfg = getattr(self, 'layout_cfg', {})
    min_window_w = int(cfg.get('min_window_w', 1000))
    min_window_h = int(cfg.get('min_window_h', 700))
    panel_min_w = int(cfg.get('panel_min_w', 280))
    panel_max_w = int(cfg.get('panel_max_w', 460))
    min_block_pixels = int(cfg.get('min_block_pixels', 10))
    ui_margin = int(cfg.get('ui_margin', 10))

    self.w = max(self.w, min_window_w)
    self.h = max(self.h, min_window_h)

    target_panel = int(self.w * 0.32)
    max_panel = max(panel_min_w, self.w - (self.board_blocks * min_block_pixels) - (ui_margin * 3))
    self.left_panel_width = max(panel_min_w, min(panel_max_w, target_panel, max_panel))

    self.footer_height = min(max(96, self.h // 6), 140)
    content_top = ui_margin
    content_h = max(1, self.h - self.footer_height - (ui_margin * 3))

    avail_w = max(1, self.w - self.left_panel_width - (ui_margin * 3))
    avail_h = max(1, content_h)
    self.bs = max(min_block_pixels, min(avail_w // self.board_blocks, avail_h // self.board_blocks))

    self.board_w = self.bs * self.board_blocks
    self.board_h = self.bs * self.board_blocks
    self.board_x = self.w - ui_margin - self.board_w
    self.board_y = content_top + max(0, (avail_h - self.board_h) // 2)

    self.left_panel_rect = pygame.Rect(ui_margin, content_top, self.left_panel_width, content_h)
    self.footer_rect = pygame.Rect(
        ui_margin,
        self.h - self.footer_height - ui_margin,
        self.w - (ui_margin * 2),
        self.footer_height,
    )


def resize_window(self, w, h):
    cfg = getattr(self, 'layout_cfg', {})
    min_window_w = int(cfg.get('min_window_w', 1000))
    min_window_h = int(cfg.get('min_window_h', 700))

    self.w = max(int(w), min_window_w)
    self.h = max(int(h), min_window_h)
    if self.render:
        self.display = pygame.display.set_mode((self.w, self.h), pygame.RESIZABLE)
    self._recompute_layout()


def get_left_control_rects(self, panel_h=260):
    cfg = getattr(self, 'layout_cfg', {})
    ui_margin = int(cfg.get('ui_margin', 10))

    panel_x = self.left_panel_rect.x + ui_margin
    panel_y = self.left_panel_rect.y + ui_margin
    panel_w = max(180, self.left_panel_rect.width - ui_margin * 2)
    max_panel_h = max(120, self.left_panel_rect.height - 70)
    panel_h = min(panel_h, max_panel_h)
    panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)

    btn_y = panel_rect.bottom + 10
    btn_gap = 10
    small_w = 40
    pause_w = max(92, panel_w - (small_w * 2) - (btn_gap * 2))
    pause_w = min(170, pause_w)
    btn_pause = pygame.Rect(panel_x, btn_y, pause_w, 34)
    btn_plus = pygame.Rect(btn_pause.right + btn_gap, btn_y, small_w, 32)
    btn_minus = pygame.Rect(btn_plus.right + btn_gap, btn_y, small_w, 32)
    return panel_rect, btn_pause, btn_plus, btn_minus
