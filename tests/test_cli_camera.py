import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gesture_recognition.camera import CameraError, open_capture
from gesture_recognition.cli import build_parser, settings_from_args
from gesture_recognition.config import Settings


class CliTests(unittest.TestCase):
    def test_defaults(self) -> None:
        settings = settings_from_args(build_parser().parse_args([]))
        self.assertEqual(settings.camera, 0)
        self.assertTrue(settings.mirror)
        self.assertTrue(settings.show_window)
        self.assertEqual(settings.min_detection_confidence, 0.5)

    def test_flags(self) -> None:
        args = build_parser().parse_args(
            [
                "--camera",
                "2",
                "--no-mirror",
                "--no-window",
                "--max-hands",
                "1",
                "--min-detection-confidence",
                "0.8",
            ]
        )
        settings = settings_from_args(args)
        self.assertEqual(settings.camera, 2)
        self.assertFalse(settings.mirror)
        self.assertFalse(settings.show_window)
        self.assertEqual(settings.max_hands, 1)
        self.assertEqual(settings.min_detection_confidence, 0.8)

    def test_confidence_is_clamped(self) -> None:
        args = build_parser().parse_args(["--min-tracking-confidence", "2.5"])
        settings = settings_from_args(args)
        self.assertEqual(settings.min_tracking_confidence, 1.0)


class CameraFailureTests(unittest.TestCase):
    def test_open_capture_raises_when_not_opened(self) -> None:
        fake_cap = mock.Mock()
        fake_cap.isOpened.return_value = False
        with mock.patch("gesture_recognition.camera.cv2.VideoCapture", return_value=fake_cap):
            with self.assertRaises(CameraError) as ctx:
                open_capture(Settings(camera=99))
        self.assertIn("Could not open camera index 99", str(ctx.exception))
        fake_cap.release.assert_called()

    def test_open_capture_raises_when_first_frame_fails(self) -> None:
        fake_cap = mock.Mock()
        fake_cap.isOpened.return_value = True
        fake_cap.read.return_value = (False, None)
        with mock.patch("gesture_recognition.camera.cv2.VideoCapture", return_value=fake_cap):
            with self.assertRaises(CameraError) as ctx:
                open_capture(Settings(camera=0))
        self.assertIn("failed to read a frame", str(ctx.exception))
        fake_cap.release.assert_called()


class OverlayTests(unittest.TestCase):
    def test_annotate_empty_frame(self) -> None:
        import numpy as np

        from gesture_recognition.overlay import annotate_frame

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        out = annotate_frame(frame, [], fps=12.3)
        self.assertEqual(out.shape, (240, 320, 3))
        # HUD should paint some non-black pixels.
        self.assertGreater(int(out.sum()), 0)


if __name__ == "__main__":
    unittest.main()
