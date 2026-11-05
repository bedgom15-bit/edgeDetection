import cv2
import numpy as np

from hand_detection import hand_mask, hand_region, matching_hand, missing_models


def hand_points(x: float, y: float) -> np.ndarray:
    """Puntos de una mano de 60x80 píxeles con la esquina superior izquierda en (x, y)."""
    grid = [(dx, dy) for dx in (0, 20, 40, 60) for dy in (0, 40, 80)]
    return np.array([(x + dx, y + dy) for dx, dy in grid], np.float32)


def skin_with_hand(x: int, y: int) -> np.ndarray:
    skin = np.zeros((240, 320), np.uint8)
    skin[y : y + 80, x : x + 60] = 255
    return skin


def test_region_covers_the_hand_and_a_margin_around_it():
    region = hand_region(hand_points(100, 80), (240, 320))

    assert region[120, 130] == 255  # dentro de la mano
    assert region[75, 130] == 255  # justo por encima: el borde de los dedos
    assert region[20, 130] == 0  # lejos


def test_region_of_a_hand_partly_outside_the_image():
    region = hand_region(hand_points(-30, 200), (240, 320))

    assert region.shape == (240, 320)
    assert region[230, 10] == 255
    assert region[100, 200] == 0


def test_region_matches_growing_the_hand_with_a_circle():
    points = hand_points(100, 80)
    expected = np.zeros((240, 320), np.uint8)
    cv2.fillConvexPoly(expected, cv2.convexHull(points.astype(np.int32)), 255)
    grow = int(0.25 * 80)
    circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    expected = cv2.dilate(expected, circle)

    different = cv2.countNonZero(cv2.absdiff(hand_region(points, (240, 320)), expected))

    # Solo pueden diferir píxeles sueltos del borde redondeado.
    assert different < 0.01 * cv2.countNonZero(expected)


def test_mask_keeps_only_skin_with_a_hand_on_it():
    skin = skin_with_hand(100, 80)
    skin[200:240, 0:80] = 255  # piel sin mano encima, como el cuello o alguien al fondo
    points = hand_points(100, 80)

    mask = hand_mask(skin, [points], [points])

    assert cv2.countNonZero(mask[80:160, 100:160]) == 60 * 80
    assert cv2.countNonZero(mask[200:240, 0:80]) == 0


def test_mask_follows_a_hand_that_moved_since_segmenting():
    skin = skin_with_hand(100, 80)  # segmentada cuando la mano estaba en (100, 80)

    mask = hand_mask(skin, [hand_points(140, 100)], [hand_points(100, 80)])

    assert cv2.countNonZero(mask[100:180, 140:200]) == 60 * 80
    assert cv2.countNonZero(mask) == 60 * 80


def test_hand_that_appeared_after_segmenting_has_no_match():
    assert matching_hand(hand_points(200, 100), [hand_points(0, 0)]) is None
    assert matching_hand(hand_points(200, 100), []) is None


def test_missing_models_lists_what_is_not_downloaded(tmp_path):
    assert len(missing_models(tmp_path)) == 2
    (tmp_path / "hand_landmarker.task").write_bytes(b"x")

    assert missing_models(tmp_path) == ["selfie_multiclass_256x256.tflite"]
