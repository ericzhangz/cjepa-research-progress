"""Camera projection of the T shape in the stock 96-pixel PushT observation."""

import math

import numpy as np


_CANVAS_SIZE = 512
_IMAGE_SIZE = 96
_T_FILL_CHROMA = 41
_T_RECTANGLES = (
    ((-60, 0), (60, 0), (60, 30), (-60, 30)),
    ((-15, 30), (15, 30), (15, 120), (-15, 120)),
)


def _chroma(rgb):
    return rgb[..., 2].astype(np.int16) - rgb[..., 0]


def render_t_chroma(poses: np.ndarray) -> np.ndarray:
    """Render B-R/41 from geometric-origin (x, y, angle) poses, with y down."""
    import cv2
    import pygame
    from gym_pusht.envs.pymunk_override import DrawOptions
    from pymunk.space_debug_draw_options import SpaceDebugColor

    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim == 0 or poses.shape[-1] != 3 or not np.isfinite(poses).all():
        raise ValueError("Expected finite PushT poses with final dimension (x, y, angle)")

    surface = pygame.Surface((_CANVAS_SIZE, _CANVAS_SIZE))
    draw = DrawOptions(surface)
    gray = SpaceDebugColor(119, 136, 153, 255)
    output = np.empty((poses.size // 3, _IMAGE_SIZE, _IMAGE_SIZE), dtype=np.float32)
    for image, (x, y, angle) in zip(output, poses.reshape(-1, 3)):
        surface.fill((255, 255, 255))
        sin, cos = math.sin(angle), math.cos(angle)
        for rectangle in _T_RECTANGLES:
            vertices = [(cos * u - sin * v + x, sin * u + cos * v + y) for u, v in rectangle]
            draw.draw_polygon(vertices, 0, gray, gray)

        rgb = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
        resized = cv2.resize(rgb, (_IMAGE_SIZE, _IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        image[:] = _chroma(resized) / np.float32(_T_FILL_CHROMA)
    return output.reshape(*poses.shape[:-1], _IMAGE_SIZE, _IMAGE_SIZE)


def visible_t_chroma(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read signed T chroma and a conservative valid mask from legal RGB alone."""
    rgb = np.asarray(rgb)
    if rgb.shape[-3:] != (_IMAGE_SIZE, _IMAGE_SIZE, 3) or rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 PushT RGB with final dimensions (96, 96, 3)")

    chroma = _chroma(rgb)
    # White, walls and the goal have R=B. The T has 0 <= B-R <= 41.
    # Blue or red pixels outside this range also censor their mixed neighbors.
    nuisance = (chroma < 0) | (chroma > _T_FILL_CHROMA) | (rgb[..., 0] > rgb[..., 1])
    horizontal = nuisance.copy()
    horizontal[..., :, 1:] |= nuisance[..., :, :-1]
    horizontal[..., :, :-1] |= nuisance[..., :, 1:]
    excluded = horizontal.copy()
    excluded[..., 1:, :] |= horizontal[..., :-1, :]
    excluded[..., :-1, :] |= horizontal[..., 1:, :]
    # The fixed walls can cover the T at the first/last two resize samples.
    excluded[..., :2, :] = True
    excluded[..., -2:, :] = True
    excluded[..., :, :2] = True
    excluded[..., :, -2:] = True
    signal = chroma.astype(np.float32) / _T_FILL_CHROMA
    return signal, ~excluded
