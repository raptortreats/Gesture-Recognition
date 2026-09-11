"""Fetch the official MediaPipe Hand Landmarker bundle on first run."""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

from gesture_recognition.config import HAND_LANDMARKER_URL, default_model_path


class ModelError(RuntimeError):
    """Raised when the Hand Landmarker model cannot be found or downloaded."""


def ensure_model(model_path: Path | None = None) -> Path:
    """Return a local path to `hand_landmarker.task`, downloading it if needed.

    The bundle is cached under ``~/.cache/gesture-recognition/`` by default so
    it is not committed to git.
    """
    path = Path(model_path) if model_path is not None else default_model_path()
    if path.is_file() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Hand Landmarker model to {path} ...", file=sys.stderr)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(HAND_LANDMARKER_URL, tmp_path)
    except urllib.error.URLError as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise ModelError(
            "Could not download the MediaPipe Hand Landmarker model from "
            f"{HAND_LANDMARKER_URL}. Check your network, or pass --model-path "
            "to an already-downloaded hand_landmarker.task file. "
            f"Original error: {exc}"
        ) from exc

    if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
        if tmp_path.exists():
            tmp_path.unlink()
        raise ModelError(f"Downloaded model at {tmp_path} is empty.")

    tmp_path.replace(path)
    return path
