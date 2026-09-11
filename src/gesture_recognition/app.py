"""Webcam / still-image loops: capture, detect, classify, overlay, display."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

from gesture_recognition.camera import CameraError, open_capture, read_frame, release_capture
from gesture_recognition.classifier import classify_gesture
from gesture_recognition.config import Settings
from gesture_recognition.landmarks import HandTracker, LandmarkError
from gesture_recognition.model import ModelError, ensure_model
from gesture_recognition.overlay import annotate_frame

_MAX_CONSECUTIVE_MISSES = 30


class AppError(RuntimeError):
    """User-facing failure that should exit the process with a message."""


def _classify_hands(tracker: HandTracker, frame) -> list:
    observations = tracker.detect(frame)
    return [(obs, classify_gesture(obs.landmarks)) for obs in observations]


def _maybe_write(frame, output: Path | None) -> None:
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame):
        raise AppError(f"Failed to write annotated image to {output}")


def _show_or_skip(window_name: str, frame, show: bool) -> bool:
    """Display a frame. Returns False if the user pressed q (or ESC)."""
    if not show:
        return True
    try:
        cv2.imshow(window_name, frame)
    except cv2.error as exc:
        raise AppError(
            "OpenCV could not open a preview window (no display?). "
            "Re-run with --no-window, and optionally --output path.jpg."
        ) from exc
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), ord("Q"), 27)


def run_webcam(settings: Settings) -> int:
    try:
        model_path = ensure_model(settings.model_path)
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    cap = None
    try:
        cap = open_capture(settings)
        with HandTracker(settings, model_path, video=True) as tracker:
            last_time = time.perf_counter()
            fps = 0.0
            misses = 0
            while True:
                frame = read_frame(cap)
                if frame is None:
                    misses += 1
                    if misses >= _MAX_CONSECUTIVE_MISSES:
                        raise CameraError(
                            "Camera stopped delivering frames. "
                            "It may have been disconnected or claimed by another app."
                        )
                    continue
                misses = 0
                if settings.mirror:
                    frame = cv2.flip(frame, 1)

                hands = _classify_hands(tracker, frame)
                now = time.perf_counter()
                dt = now - last_time
                last_time = now
                if dt > 0:
                    instant = 1.0 / dt
                    fps = instant if fps == 0.0 else (0.85 * fps + 0.15 * instant)

                annotate_frame(frame, hands, fps=fps, show_quit=settings.show_window)
                if not _show_or_skip(settings.window_name, frame, settings.show_window):
                    break
                if not settings.show_window:
                    # Headless webcam does not make sense as an infinite loop.
                    _maybe_write(frame, settings.output)
                    break
    except (CameraError, AppError, ModelError, LandmarkError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        release_capture(cap)
        if settings.show_window:
            cv2.destroyAllWindows()
    return 0


def run_image(settings: Settings) -> int:
    assert settings.image is not None
    image_path = settings.image
    if not image_path.is_file():
        print(f"error: image not found: {image_path}", file=sys.stderr)
        return 1

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(
            f"error: could not read image (unsupported or corrupt): {image_path}",
            file=sys.stderr,
        )
        return 1

    try:
        model_path = ensure_model(settings.model_path)
        with HandTracker(settings, model_path, video=False) as tracker:
            if settings.mirror:
                frame = cv2.flip(frame, 1)
            hands = _classify_hands(tracker, frame)
            annotate_frame(frame, hands, fps=None, show_quit=settings.show_window)
            _maybe_write(frame, settings.output)
            if settings.show_window:
                try:
                    cv2.imshow(settings.window_name, frame)
                    print("Press q in the preview window to close.", file=sys.stderr)
                    while True:
                        key = cv2.waitKey(50) & 0xFF
                        if key in (ord("q"), ord("Q"), 27):
                            break
                except cv2.error as exc:
                    raise AppError(
                        "OpenCV could not open a preview window (no display?). "
                        "Re-run with --no-window and --output path.jpg."
                    ) from exc
    except (AppError, ModelError, LandmarkError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if settings.show_window:
            cv2.destroyAllWindows()
    return 0


def run(settings: Settings) -> int:
    if settings.image is not None:
        return run_image(settings)
    return run_webcam(settings)
