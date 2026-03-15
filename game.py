"""Compatibility facade for the Snake MVC refactor.

This module keeps the original import path stable (from game import SnakeGameAI)
while delegating implementation to dedicated MVC modules.
"""

from game_model import (
    BLACK,
    BLUE1,
    BLUE2,
    BOARD_BLOCKS,
    BOARD_BORDER,
    BLOCK_SIZE,
    DEFAULT_SPEED,
    DIR_VECTORS,
    FOOTER_BG,
    FOOTER_BORDER,
    FOOTER_H,
    LEFT_PANEL_W,
    MAX_EPISODE_MOVES,
    MIN_BLOCK_PIXELS,
    MIN_WINDOW_H,
    MIN_WINDOW_W,
    PANEL_BG,
    PANEL_BORDER,
    PANEL_MAX_W,
    PANEL_MIN_W,
    RED,
    UI_MARGIN,
    WHITE,
    Direction,
    SnakeGameAI,
)


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


if __name__ == '__main__':
    from game_controller import run_cli

    run_cli()
