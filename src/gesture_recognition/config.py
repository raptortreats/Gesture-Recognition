"""Default runtime settings and MediaPipe model location."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Official float16 Hand Landmarker bundle (v1). Downloaded on first run;
# not committed to the repo.
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_FILENAME = "hand_landmarker.task"


def default_model_path() -> Path:
    """User-cache path for the Hand Landmarker .task file."""
    return Path.home() / ".cache" / "gesture-recognition" / HAND_LANDMARKER_FILENAME


@dataclass(frozen=True)
class Settings:
    camera: int = 0
    max_hands: int = 2
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    mirror: bool = True
    model_path: Path | None = None
    image: Path | None = None
    output: Path | None = None
    show_window: bool = True
    frame_width: int | None = None
    frame_height: int | None = None
    window_name: str = "Gesture Recognition"
