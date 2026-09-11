"""MediaPipe Hand Landmarker wrapper: BGR frames in, 21 landmarks out."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray

from gesture_recognition.classifier import NUM_LANDMARKS, Landmark
from gesture_recognition.config import Settings


class LandmarkError(RuntimeError):
    """Raised when the Hand Landmarker native library cannot be loaded."""


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


@dataclass(frozen=True)
class HandObservation:
    landmarks: list[Landmark]
    handedness: str
    score: float


def _to_landmarks(mp_landmarks) -> list[Landmark]:
    points = [
        Landmark(x=float(lm.x), y=float(lm.y), z=float(getattr(lm, "z", 0.0) or 0.0))
        for lm in mp_landmarks
    ]
    if len(points) < NUM_LANDMARKS:
        raise ValueError(f"Hand Landmarker returned {len(points)} points")
    return points[:NUM_LANDMARKS]


class HandTracker:
    """Detects up to `settings.max_hands` hands per frame."""

    def __init__(self, settings: Settings, model_path: Path, *, video: bool) -> None:
        self._settings = settings
        self._timestamp_ms = 0
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO if video else RunningMode.IMAGE,
            num_hands=settings.max_hands,
            min_hand_detection_confidence=settings.min_detection_confidence,
            min_hand_presence_confidence=settings.min_presence_confidence,
            min_tracking_confidence=settings.min_tracking_confidence,
        )
        try:
            self._landmarker = HandLandmarker.create_from_options(options)
        except OSError as exc:
            raise LandmarkError(
                "MediaPipe failed to load its native library. On Debian/Ubuntu, "
                "install EGL with: sudo apt-get install libegl1. "
                f"Original error: {exc}"
            ) from exc
        self._video = video

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def detect(self, bgr_frame: NDArray[np.uint8]) -> list[HandObservation]:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._video:
            self._timestamp_ms += 33  # ~30 FPS; must be strictly increasing
            result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)

        observations: list[HandObservation] = []
        hands = result.hand_landmarks or []
        handedness_lists = result.handedness or []
        for index, mp_landmarks in enumerate(hands):
            label = "Unknown"
            score = 0.0
            if index < len(handedness_lists) and handedness_lists[index]:
                category = handedness_lists[index][0]
                label = getattr(category, "category_name", None) or "Unknown"
                score = float(getattr(category, "score", 0.0) or 0.0)
            observations.append(
                HandObservation(
                    landmarks=_to_landmarks(mp_landmarks),
                    handedness=label,
                    score=score,
                )
            )
        return observations
