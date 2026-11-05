import cv2
import numpy as np
import pytest

from depth_detection import DEPTH_MODEL, INPUT_SIZE, MODELS_DIR, estimate_depth, near_mask

SHAPE = (480, 640)


def region(nearness: np.ndarray, top: float, bottom: float, left: float, right: float) -> tuple[slice, slice]:
    """Zona de la imagen en fracciones de su alto y ancho, sea cual sea el tamaño del modelo."""
    height, width = nearness.shape
    return slice(round(top * height), round(bottom * height)), slice(round(left * width), round(right * width))


def scene_nearness() -> np.ndarray:
    """Cercanía al tamaño del modelo: fondo lejano, una cara y una mano más cerca que la cara."""
    width, height = INPUT_SIZE
    nearness = np.full((height, width), 0.2, np.float32)
    nearness[region(nearness, 0.2, 0.8, 0.6, 0.85)] = 2.5  # cara
    nearness[region(nearness, 0.3, 0.7, 0.15, 0.45)] = 3.8  # mano
    return nearness


def test_only_the_nearest_object_is_marked():
    mask = near_mask(scene_nearness(), SHAPE, 85)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    assert mask.shape == SHAPE
    assert len(contours) == 1
    x, _, width, _ = cv2.boundingRect(contours[0])
    # La mano va del 15 % al 45 % del ancho: en 640 píxeles, de 96 a 288.
    assert abs(x - 96) < 12 and abs(width - 192) < 20


def test_lower_threshold_also_marks_what_is_a_bit_farther():
    mask = near_mask(scene_nearness(), SHAPE, 60)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    assert len(contours) == 2


def test_a_single_extreme_pixel_does_not_hide_the_nearest_object():
    nearness = scene_nearness()
    nearness[0, 0] = 100

    mask = near_mask(nearness, SHAPE, 85)

    assert cv2.countNonZero(mask) > 0


def test_empty_nearness_gives_an_empty_mask():
    width, height = INPUT_SIZE

    assert cv2.countNonZero(near_mask(np.zeros((height, width), np.float32), SHAPE, 85)) == 0


@pytest.mark.skipif(not (MODELS_DIR / DEPTH_MODEL).is_file(), reason="falta el modelo de distancia")
def test_model_gives_one_nearness_value_per_pixel_of_its_input():
    net = cv2.dnn.readNetFromONNX(str(MODELS_DIR / DEPTH_MODEL))
    frame = np.random.default_rng(0).integers(0, 255, (*SHAPE, 3), dtype=np.uint8)

    nearness = estimate_depth(net, frame)

    assert nearness.shape == (INPUT_SIZE[1], INPUT_SIZE[0])
    assert np.isfinite(nearness).all()
