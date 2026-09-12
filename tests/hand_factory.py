"""Synthetic 21-point hands for the rule-based classifier."""

from __future__ import annotations

from gesture_recognition.classifier import (
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    NUM_LANDMARKS,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_CMC,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
    Landmark,
)

# Default palm facing the camera, wrist at the bottom, thumb on the left.
_BASE = {
    WRIST: (0.50, 0.88),
    THUMB_CMC: (0.40, 0.80),
    THUMB_MCP: (0.32, 0.72),
    INDEX_MCP: (0.42, 0.60),
    MIDDLE_MCP: (0.50, 0.58),
    RING_MCP: (0.58, 0.60),
    PINKY_MCP: (0.64, 0.64),
}


def _lm(x: float, y: float, z: float = 0.0) -> Landmark:
    return Landmark(x=x, y=y, z=z)


def _finger_points(mcp: tuple[float, float], *, extended: bool) -> tuple[tuple[float, float], tuple[float, float]]:
    mx, my = mcp
    if extended:
        pip = (mx, my - 0.16)
        tip = (mx, my - 0.36)
    else:
        pip = (mx, my + 0.02)
        tip = (mx + 0.01, my + 0.08)
    return pip, tip


def _thumb_points(*, extended: bool, pointing_up: bool) -> dict[int, tuple[float, float]]:
    if pointing_up and extended:
        return {
            THUMB_MCP: (0.34, 0.70),
            THUMB_IP: (0.32, 0.52),
            THUMB_TIP: (0.31, 0.30),
        }
    if extended:
        # Out to the side, not "up".
        return {
            THUMB_MCP: (0.32, 0.72),
            THUMB_IP: (0.20, 0.70),
            THUMB_TIP: (0.08, 0.68),
        }
    # Tucked across the palm toward the index MCP.
    return {
        THUMB_MCP: (0.36, 0.74),
        THUMB_IP: (0.40, 0.68),
        THUMB_TIP: (0.44, 0.62),
    }


def make_hand(
    *,
    index: bool = False,
    middle: bool = False,
    ring: bool = False,
    pinky: bool = False,
    thumb: bool = False,
    thumb_up: bool = False,
) -> list[Landmark]:
    """Build 21 landmarks. `True` means that finger is extended."""
    coords: dict[int, tuple[float, float]] = dict(_BASE)
    coords.update(_thumb_points(extended=thumb or thumb_up, pointing_up=thumb_up))

    finger_specs = (
        (INDEX_MCP, INDEX_PIP, INDEX_TIP, index),
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP, middle),
        (RING_MCP, RING_PIP, RING_TIP, ring),
        (PINKY_MCP, PINKY_PIP, PINKY_TIP, pinky),
    )
    for mcp_id, pip_id, tip_id, extended in finger_specs:
        pip, tip = _finger_points(coords[mcp_id], extended=extended)
        coords[pip_id] = pip
        coords[tip_id] = tip
        # DIP sits between PIP and tip.
        coords[pip_id + 1] = ((pip[0] + tip[0]) / 2, (pip[1] + tip[1]) / 2)

    points = [_lm(0.0, 0.0) for _ in range(NUM_LANDMARKS)]
    for idx, (x, y) in coords.items():
        points[idx] = _lm(x, y)
    return points
