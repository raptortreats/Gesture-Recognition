#!/usr/bin/env python3
"""Run the webcam gesture-recognition app without installing the package."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from gesture_recognition.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
