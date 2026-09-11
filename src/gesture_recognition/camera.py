"""Open and read a webcam (or other OpenCV VideoCapture source) with clear errors."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from gesture_recognition.config import Settings


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or a frame cannot be read."""


def open_capture(settings: Settings) -> cv2.VideoCapture:
    """Open `settings.camera` and fail with a readable message if that does not work."""
    cap = cv2.VideoCapture(settings.camera)
    if settings.frame_width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.frame_width)
    if settings.frame_height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.frame_height)

    if not cap.isOpened():
        cap.release()
        raise CameraError(
            f"Could not open camera index {settings.camera}. "
            "Check that a webcam is connected and not already in use, "
            "or pass --camera with a different index."
        )

    # Some backends report opened=True but never deliver frames.
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise CameraError(
            f"Opened camera index {settings.camera} but failed to read a frame. "
            "Try another --camera index or close other apps using the webcam."
        )
    return cap


def read_frame(cap: cv2.VideoCapture) -> Optional[NDArray[np.uint8]]:
    """Return the next BGR frame, or None on a transient read miss."""
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def release_capture(cap: cv2.VideoCapture | None) -> None:
    if cap is not None:
        cap.release()
