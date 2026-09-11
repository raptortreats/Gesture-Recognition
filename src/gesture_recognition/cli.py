"""Command-line entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from gesture_recognition.app import run
from gesture_recognition.config import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gesture-recognition",
        description=(
            "Real-time webcam gesture recognition using MediaPipe Hands landmarks "
            "and a small rule-based classifier (open palm, fist, thumbs up, "
            "victory, pointing)."
        ),
    )
    parser.add_argument(
        "--camera",
        "-c",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0).",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=2,
        help="Maximum number of hands to detect (default: 2).",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        metavar="CONF",
        help="Palm-detection confidence threshold, 0–1 (default: 0.5).",
    )
    parser.add_argument(
        "--min-presence-confidence",
        type=float,
        default=0.5,
        metavar="CONF",
        help="Hand-presence confidence threshold, 0–1 (default: 0.5).",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        metavar="CONF",
        help="Landmark-tracking confidence threshold, 0–1 (default: 0.5).",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable the default selfie-style horizontal flip.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional capture width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional capture height in pixels.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to hand_landmarker.task (downloaded on first run if omitted).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Classify a still image instead of opening the webcam.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write the annotated frame to this image path.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Skip the OpenCV preview window (useful on headless machines).",
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    return Settings(
        camera=args.camera,
        max_hands=max(1, args.max_hands),
        min_detection_confidence=_clamp_conf(args.min_detection_confidence),
        min_presence_confidence=_clamp_conf(args.min_presence_confidence),
        min_tracking_confidence=_clamp_conf(args.min_tracking_confidence),
        mirror=not args.no_mirror,
        model_path=args.model_path,
        image=args.image,
        output=args.output,
        show_window=not args.no_window,
        frame_width=args.width,
        frame_height=args.height,
    )


def _clamp_conf(value: float) -> float:
    return min(1.0, max(0.0, value))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(settings_from_args(args))
