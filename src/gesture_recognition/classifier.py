"""Rule-based gesture labels from MediaPipe's 21 hand landmarks.

Finger "extended" vs "folded" is inferred from 2D landmark geometry, not a
trained classifier. See README.md for the rule summary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

# MediaPipe Hands landmark indices.
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

NUM_LANDMARKS = 21


class Gesture(str, Enum):
    OPEN_PALM = "Open Palm"
    FIST = "Fist"
    THUMBS_UP = "Thumbs Up"
    VICTORY = "Victory"
    POINTING = "Pointing"
    UNKNOWN = "Unknown"
    NO_HAND = "No Hand"


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class FingerState:
    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool
    thumb_up: bool


def _dist(a: Landmark, b: Landmark) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _palm_width(landmarks: Sequence[Landmark]) -> float:
    return max(_dist(landmarks[INDEX_MCP], landmarks[PINKY_MCP]), 1e-6)


def _non_thumb_extended(
    landmarks: Sequence[Landmark],
    tip: int,
    pip: int,
    mcp: int,
) -> bool:
    """A finger is extended when the tip sits well past the PIP from the MCP."""
    palm = _palm_width(landmarks)
    mcp_to_tip = _dist(landmarks[tip], landmarks[mcp])
    mcp_to_pip = _dist(landmarks[pip], landmarks[mcp])
    stretched = mcp_to_tip > mcp_to_pip * 1.25
    long_enough = mcp_to_tip > palm * 0.55
    return stretched and long_enough


def _thumb_extended(landmarks: Sequence[Landmark]) -> bool:
    """Thumb is extended when the tip is away from the index MCP (not tucked)."""
    palm = _palm_width(landmarks)
    tip = landmarks[THUMB_TIP]
    ip = landmarks[THUMB_IP]
    index_mcp = landmarks[INDEX_MCP]
    wrist = landmarks[WRIST]
    away_from_index = _dist(tip, index_mcp) > palm * 0.65
    farther_than_ip = _dist(tip, wrist) > _dist(ip, wrist)
    return away_from_index and farther_than_ip


def _thumb_points_up(landmarks: Sequence[Landmark]) -> bool:
    """True when the thumb is mostly vertical (image y grows downward)."""
    tip = landmarks[THUMB_TIP]
    mcp = landmarks[THUMB_MCP]
    dx = abs(tip.x - mcp.x)
    dy = mcp.y - tip.y
    return dy > 0 and dy > dx * 1.1


def finger_state(landmarks: Sequence[Landmark]) -> FingerState:
    if len(landmarks) < NUM_LANDMARKS:
        raise ValueError(f"Expected {NUM_LANDMARKS} landmarks, got {len(landmarks)}")
    thumb = _thumb_extended(landmarks)
    return FingerState(
        thumb=thumb,
        index=_non_thumb_extended(landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP),
        middle=_non_thumb_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
        ring=_non_thumb_extended(landmarks, RING_TIP, RING_PIP, RING_MCP),
        pinky=_non_thumb_extended(landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP),
        thumb_up=thumb and _thumb_points_up(landmarks),
    )


def classify_gesture(landmarks: Sequence[Landmark] | None) -> Gesture:
    """Return a gesture label for one hand, or NO_HAND when landmarks are missing."""
    if landmarks is None:
        return Gesture.NO_HAND
    if len(landmarks) < NUM_LANDMARKS:
        return Gesture.UNKNOWN

    state = finger_state(landmarks)
    others = (state.index, state.middle, state.ring, state.pinky)

    # More specific two-/one-finger poses first so they are not swallowed by palm.
    if state.index and state.middle and not state.ring and not state.pinky:
        return Gesture.VICTORY
    if state.index and not state.middle and not state.ring and not state.pinky:
        return Gesture.POINTING
    if state.thumb_up and not any(others):
        return Gesture.THUMBS_UP
    if all(others):
        return Gesture.OPEN_PALM
    if not any(others) and not state.thumb_up:
        return Gesture.FIST
    return Gesture.UNKNOWN
