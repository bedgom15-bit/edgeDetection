import cv2
import numpy as np

from edge_detection import ForegroundDetector, Params, detect_edges, draw_contours, find_contours


def noisy_scene(rng: np.random.Generator) -> np.ndarray:
    return np.clip(90 + rng.normal(0, 3, (240, 320, 3)), 0, 255).astype(np.uint8)


def learned_detector(rng: np.random.Generator) -> ForegroundDetector:
    detector = ForegroundDetector()
    while detector.learning:
        detector.apply(noisy_scene(rng), Params())
    return detector


def test_empty_scene_gives_no_objects():
    rng = np.random.default_rng(0)
    detector = learned_detector(rng)

    mask = detector.apply(noisy_scene(rng), Params())

    assert find_contours(mask, Params()) == []


def test_each_new_object_gets_a_contour():
    rng = np.random.default_rng(0)
    detector = learned_detector(rng)
    frame = noisy_scene(rng)
    add_rectangle(frame, 30, 40, 80, 80)
    add_rectangle(frame, 190, 60, 90, 70)

    contours = find_contours(detector.apply(frame, Params()), Params())

    assert len(contours) == 2


def test_still_object_is_not_absorbed_into_the_background():
    rng = np.random.default_rng(0)
    detector = learned_detector(rng)
    for _ in range(200):
        frame = noisy_scene(rng)
        add_rectangle(frame, 100, 80, 100, 60)
        mask = detector.apply(frame, Params())

    contours = find_contours(mask, Params())

    assert len(contours) == 1
    assert abs(cv2.contourArea(contours[0]) - 100 * 60) < 0.2 * 100 * 60


def test_reset_learns_the_background_again():
    rng = np.random.default_rng(0)
    detector = learned_detector(rng)

    detector.reset()

    assert detector.learning


def black_image() -> np.ndarray:
    return np.zeros((240, 320, 3), np.uint8)


def add_rectangle(image: np.ndarray, x: int, y: int, width: int, height: int) -> None:
    cv2.rectangle(image, (x, y), (x + width - 1, y + height - 1), (255, 255, 255), -1)


def contours_of(image: np.ndarray, params: Params) -> list[np.ndarray]:
    return find_contours(detect_edges(image, params), params)


def test_rectangle_gives_one_contour_with_expected_area():
    image = black_image()
    add_rectangle(image, 100, 80, 100, 60)

    contours = contours_of(image, Params())

    assert len(contours) == 1
    assert abs(cv2.contourArea(contours[0]) - 100 * 60) < 0.1 * 100 * 60


def test_uniform_image_gives_no_contours():
    assert contours_of(black_image(), Params()) == []


def test_small_contours_are_filtered_by_min_area():
    image = black_image()
    add_rectangle(image, 20, 20, 10, 10)
    add_rectangle(image, 100, 80, 100, 60)

    assert len(contours_of(image, Params(min_area=500))) == 1
    assert len(contours_of(image, Params(min_area=10))) == 2


def test_draw_contours_does_not_modify_the_input():
    image = black_image()
    add_rectangle(image, 100, 80, 100, 60)
    original = image.copy()

    output = draw_contours(image, contours_of(image, Params()))

    assert np.array_equal(image, original)
    assert not np.array_equal(output, original)
