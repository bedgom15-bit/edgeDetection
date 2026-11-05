import cv2
import numpy as np

from edge_detection import (
    ForegroundDetector,
    MaskStabilizer,
    Params,
    detect_edges,
    draw_contours,
    find_contours,
    smooth_contour,
)


def blob_mask() -> np.ndarray:
    mask = np.zeros((240, 320), np.uint8)
    mask[80:160, 100:200] = 255
    return mask


def test_stabilizer_ignores_a_one_frame_flicker():
    stabilizer = MaskStabilizer()
    empty = np.zeros((240, 320), np.uint8)
    for _ in range(10):
        stabilizer.apply(empty, Params())

    assert cv2.countNonZero(stabilizer.apply(blob_mask(), Params())) == 0


def test_stabilizer_shows_a_persistent_object_after_a_few_frames():
    stabilizer = MaskStabilizer()
    stabilizer.apply(np.zeros((240, 320), np.uint8), Params())

    visible = [cv2.countNonZero(stabilizer.apply(blob_mask(), Params())) > 0 for _ in range(5)]

    assert not visible[0]
    assert visible[-1]


def test_stabilizer_with_zero_stability_passes_the_mask_through():
    stabilizer = MaskStabilizer()
    stabilizer.apply(np.zeros((240, 320), np.uint8), Params(stability=0))

    assert np.array_equal(stabilizer.apply(blob_mask(), Params(stability=0)), blob_mask())


def noisy_circle(rng: np.random.Generator) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    radii = 50 + rng.uniform(-3, 3, angles.size)
    points = np.column_stack((160 + radii * np.cos(angles), 120 + radii * np.sin(angles)))
    return np.round(points).astype(np.int32).reshape(-1, 1, 2)


def test_smoothing_removes_spikes_and_keeps_the_area():
    contour = noisy_circle(np.random.default_rng(0))

    smoothed = smooth_contour(contour, 5)

    assert cv2.arcLength(smoothed, True) < 0.9 * cv2.arcLength(contour, True)
    assert abs(cv2.contourArea(smoothed) - cv2.contourArea(contour)) < 0.05 * cv2.contourArea(contour)


def test_smoothing_radius_zero_leaves_the_contour_unchanged():
    contour = noisy_circle(np.random.default_rng(0))

    assert smooth_contour(contour, 0) is contour


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


def detector_learned_with_object(rng: np.random.Generator) -> ForegroundDetector:
    detector = ForegroundDetector()
    while detector.learning:
        frame = noisy_scene(rng)
        add_rectangle(frame, 30, 40, 80, 80)
        detector.apply(frame, Params())
    return detector


def test_object_present_at_start_leaves_no_ghost_when_removed():
    rng = np.random.default_rng(0)
    detector = detector_learned_with_object(rng)

    masks = [detector.apply(noisy_scene(rng), Params()) for _ in range(3)]

    assert all(find_contours(mask, Params()) == [] for mask in masks)


def test_object_present_at_start_is_tracked_after_moving():
    rng = np.random.default_rng(0)
    detector = detector_learned_with_object(rng)
    for _ in range(3):
        frame = noisy_scene(rng)
        add_rectangle(frame, 190, 60, 90, 70)
        mask = detector.apply(frame, Params())

    contours = find_contours(mask, Params())

    assert len(contours) == 1
    x, y, _, _ = cv2.boundingRect(contours[0])
    assert abs(x - 190) < 10 and abs(y - 60) < 10


def test_full_resolution_frames_give_a_full_size_mask_without_ghosts():
    rng = np.random.default_rng(0)
    detector = ForegroundDetector()
    while detector.learning:
        frame = cv2.resize(noisy_scene(rng), (640, 480))
        add_rectangle(frame, 60, 80, 160, 160)
        detector.apply(frame, Params())
    frame = cv2.resize(noisy_scene(rng), (640, 480))
    add_rectangle(frame, 380, 120, 180, 140)

    mask = detector.apply(frame, Params())
    contours = find_contours(mask, Params())

    assert mask.shape == (480, 640)
    assert len(contours) == 1
    assert abs(cv2.contourArea(contours[0]) - 180 * 140) < 0.1 * 180 * 140


def test_exposure_change_does_not_hide_the_object():
    for exposure in (1.4, 0.7):
        rng = np.random.default_rng(0)
        detector = learned_detector(rng)
        frame = noisy_scene(rng).astype(np.float32) * exposure
        frame[80:160, 100:200] = 230
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        contours = find_contours(detector.apply(frame, Params()), Params())

        assert len(contours) == 1, exposure
        assert abs(cv2.contourArea(contours[0]) - 100 * 80) < 0.15 * 100 * 80, exposure


def textured_scene(rng: np.random.Generator) -> np.ndarray:
    pattern = cv2.resize(np.random.default_rng(42).uniform(40, 160, (24, 32)), (320, 240))
    noise = rng.normal(0, 3, (240, 320))
    return np.clip(pattern + noise, 0, 255).astype(np.uint8)[:, :, None].repeat(3, axis=2)


def learned_textured_detector(rng: np.random.Generator) -> ForegroundDetector:
    detector = ForegroundDetector()
    while detector.learning:
        detector.apply(textured_scene(rng), Params())
    return detector


def test_uneven_lighting_change_is_not_an_object():
    rng = np.random.default_rng(0)
    detector = learned_textured_detector(rng)
    frame = textured_scene(rng).astype(np.float32)
    # Una luz que ilumina más la mitad izquierda que la derecha.
    frame[:, :160] = np.clip(frame[:, :160] * 1.5 + 20, 0, 255)

    contours = find_contours(detector.apply(frame.astype(np.uint8), Params()), Params())

    assert contours == []


def test_object_on_textured_background_is_detected():
    rng = np.random.default_rng(0)
    detector = learned_textured_detector(rng)
    frame = textured_scene(rng)
    frame[80:160, 100:200] = (60, 120, 200)

    contours = find_contours(detector.apply(frame, Params()), Params())

    assert len(contours) == 1
    assert abs(cv2.contourArea(contours[0]) - 100 * 80) < 0.15 * 100 * 80


def test_hand_with_the_same_tone_as_the_wall_is_not_a_shadow():
    # Pared beige medida en una captura real, y la palma medida en esa captura un 25 % más
    # clara, como sus zonas más iluminadas: relación con la pared de 0,6 a 0,7 en cada canal.
    rng = np.random.default_rng(0)
    wall = np.array([161, 171, 170], np.float32)
    palm = np.array([78, 81, 94], np.float32) * 1.25

    def scene() -> np.ndarray:
        return np.clip(wall + rng.normal(0, 3, (240, 320, 3)), 0, 255).astype(np.uint8)

    detector = ForegroundDetector()
    while detector.learning:
        detector.apply(scene(), Params())
    frame = scene()
    frame[60:200, 100:200] = np.clip(palm + rng.normal(0, 3, (140, 100, 3)), 0, 255)

    contours = find_contours(detector.apply(frame, Params()), Params())

    assert len(contours) == 1
    assert abs(cv2.contourArea(contours[0]) - 100 * 140) < 0.15 * 100 * 140


def square_contour() -> np.ndarray:
    contour, _ = cv2.findContours(blob_mask(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contour[0]


def test_blurry_moving_object_is_not_a_ghost():
    # Valores medidos con la webcam: una mano en movimiento tenía bordes 42 frente a 47 del fondo.
    current = np.full((240, 320), 42, np.float32)
    background = np.full((240, 320), 47, np.float32)

    assert not ForegroundDetector.looks_like_ghost(square_contour(), current, background)


def test_edgeless_hole_is_a_ghost():
    current = np.full((240, 320), 5, np.float32)
    background = np.full((240, 320), 50, np.float32)

    assert ForegroundDetector.looks_like_ghost(square_contour(), current, background)


def test_shadow_is_not_an_object():
    rng = np.random.default_rng(0)
    detector = learned_detector(rng)
    frame = noisy_scene(rng)
    frame[80:160, 100:200] = (frame[80:160, 100:200] * 0.7).astype(np.uint8)

    assert find_contours(detector.apply(frame, Params()), Params()) == []


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
