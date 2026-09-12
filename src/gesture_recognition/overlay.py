"""Draw hand skeletons, gesture labels, and FPS onto a BGR frame."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from gesture_recognition.classifier import Gesture, Landmark
from gesture_recognition.landmarks import HandObservation

# Landmark pairs that form the MediaPipe hand skeleton.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)

_DOT_COLOR = (0, 220, 80)
_LINE_COLOR = (240, 240, 240)
_LABEL_BG = (32, 32, 32)
_LABEL_FG = (255, 255, 255)


def _px(landmark: Landmark, width: int, height: int) -> tuple[int, int]:
    x = int(np.clip(landmark.x, 0.0, 1.0) * (width - 1))
    y = int(np.clip(landmark.y, 0.0, 1.0) * (height - 1))
    return x, y


def draw_hand(frame: NDArray[np.uint8], landmarks: list[Landmark]) -> None:
    height, width = frame.shape[:2]
    for start, end in HAND_CONNECTIONS:
        if start < len(landmarks) and end < len(landmarks):
            cv2.line(
                frame,
                _px(landmarks[start], width, height),
                _px(landmarks[end], width, height),
                _LINE_COLOR,
                2,
                cv2.LINE_AA,
            )
    for landmark in landmarks:
        cv2.circle(frame, _px(landmark, width, height), 4, _DOT_COLOR, -1, cv2.LINE_AA)


def _banner(frame: NDArray[np.uint8], lines: list[str]) -> None:
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    padding = 8
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    line_h = max(size[1] for size in sizes) + 10
    box_w = max(size[0] for size in sizes) + padding * 2
    box_h = line_h * len(lines) + padding
    cv2.rectangle(frame, (0, 0), (box_w, box_h), _LABEL_BG, -1)
    for i, line in enumerate(lines):
        y = padding + (i + 1) * line_h - 8
        cv2.putText(
            frame,
            line,
            (padding, y),
            font,
            scale,
            _LABEL_FG,
            thickness,
            cv2.LINE_AA,
        )


def annotate_frame(
    frame: NDArray[np.uint8],
    hands: list[tuple[HandObservation, Gesture]],
    fps: float | None = None,
    *,
    show_quit: bool = True,
) -> NDArray[np.uint8]:
    """Draw landmarks plus a top-left HUD. Mutates and returns `frame`."""
    if not hands:
        lines = ["No Hand"]
        if fps is not None:
            lines.append(f"FPS: {fps:.1f}")
        if show_quit:
            lines.append("Press q to quit")
        _banner(frame, lines)
        return frame

    for observation, _gesture in hands:
        draw_hand(frame, observation.landmarks)

    primary_obs, primary_gesture = hands[0]
    lines = [primary_gesture.value, f"{primary_obs.handedness} hand"]
    if fps is not None:
        lines.append(f"FPS: {fps:.1f}")
    if len(hands) > 1:
        extra = ", ".join(g.value for _, g in hands[1:])
        lines.append(f"Also: {extra}")
    if show_quit:
        lines.append("Press q to quit")
    _banner(frame, lines)
    return frame
