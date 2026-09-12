import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gesture_recognition.classifier import Gesture, classify_gesture, finger_state
from hand_factory import make_hand


class ClassifierTests(unittest.TestCase):
    def test_no_hand(self) -> None:
        self.assertEqual(classify_gesture(None), Gesture.NO_HAND)

    def test_too_few_landmarks(self) -> None:
        self.assertEqual(classify_gesture([]), Gesture.UNKNOWN)

    def test_open_palm(self) -> None:
        hand = make_hand(index=True, middle=True, ring=True, pinky=True, thumb=True)
        self.assertEqual(classify_gesture(hand), Gesture.OPEN_PALM)

    def test_open_palm_without_thumb(self) -> None:
        hand = make_hand(index=True, middle=True, ring=True, pinky=True)
        self.assertEqual(classify_gesture(hand), Gesture.OPEN_PALM)

    def test_fist(self) -> None:
        hand = make_hand()
        self.assertEqual(classify_gesture(hand), Gesture.FIST)

    def test_thumbs_up(self) -> None:
        hand = make_hand(thumb_up=True)
        self.assertEqual(classify_gesture(hand), Gesture.THUMBS_UP)

    def test_sideways_thumb_is_not_thumbs_up(self) -> None:
        hand = make_hand(thumb=True)
        self.assertEqual(classify_gesture(hand), Gesture.FIST)

    def test_victory(self) -> None:
        hand = make_hand(index=True, middle=True)
        self.assertEqual(classify_gesture(hand), Gesture.VICTORY)

    def test_pointing(self) -> None:
        hand = make_hand(index=True)
        self.assertEqual(classify_gesture(hand), Gesture.POINTING)

    def test_pointing_with_thumb_out(self) -> None:
        hand = make_hand(index=True, thumb=True)
        self.assertEqual(classify_gesture(hand), Gesture.POINTING)

    def test_three_fingers_is_unknown(self) -> None:
        hand = make_hand(index=True, middle=True, ring=True)
        self.assertEqual(classify_gesture(hand), Gesture.UNKNOWN)

    def test_finger_state_victory(self) -> None:
        state = finger_state(make_hand(index=True, middle=True))
        self.assertTrue(state.index)
        self.assertTrue(state.middle)
        self.assertFalse(state.ring)
        self.assertFalse(state.pinky)


if __name__ == "__main__":
    unittest.main()
