"""Silueta de las manos con MediaPipe: los puntos de cada mano y la piel segmentada."""

import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODELS_DIR = Path(__file__).parent / "models"
HAND_MODEL = "hand_landmarker.task"
SEGMENTER_MODEL = "selfie_multiclass_256x256.tflite"
# Categoría del segmentador multiclase: piel del cuerpo, separada de la piel de la cara.
BODY_SKIN = 2
# Segundos mínimos entre segmentaciones. Segmentar sin pausa le quita procesador al
# detector de manos (medido: de 27 a 40 ms por frame), y la silueta se mueve con la mano
# entre una segmentación y otra.
SEGMENT_INTERVAL = 0.2


def missing_models(models_dir: Path = MODELS_DIR) -> list[str]:
    """Devuelve los nombres de los modelos que no están descargados."""
    return [name for name in (HAND_MODEL, SEGMENTER_MODEL) if not (models_dir / name).is_file()]


def hand_region(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Zona donde puede estar la mano: el polígono de sus puntos, ampliado.

    Los puntos llegan al centro de los dedos y de la palma, no a su borde: sin ampliar,
    la silueta quedaría recortada.
    """
    polygon = cv2.convexHull(points.astype(np.int32))
    x, y, width, height = cv2.boundingRect(polygon)
    grow = max(int(0.25 * max(width, height)), 5)
    # Se amplía con la distancia a la mano en un recuadro a su alrededor: dilatar con un
    # círculo de ese tamaño tardaba ~20 ms por mano.
    left, top = x - grow - 1, y - grow - 1
    outside = np.full((height + 2 * grow + 2, width + 2 * grow + 2), 255, np.uint8)
    cv2.fillConvexPoly(outside, polygon - [left, top], 0)
    distance = cv2.distanceTransform(outside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    box = np.where(distance <= grow, 255, 0).astype(np.uint8)

    region = np.zeros(shape, np.uint8)
    # Recorte del recuadro a los bordes de la imagen (la mano puede salirse en parte).
    top_clip, left_clip = max(top, 0), max(left, 0)
    bottom, right = min(top + box.shape[0], shape[0]), min(left + box.shape[1], shape[1])
    if bottom > top_clip and right > left_clip:
        region[top_clip:bottom, left_clip:right] = box[
            top_clip - top : bottom - top, left_clip - left : right - left
        ]
    return region


def follow_hand(skin: np.ndarray, then: np.ndarray, now: np.ndarray) -> np.ndarray:
    """Mueve la piel segmentada en un frame anterior a donde está ahora la mano."""
    matrix, _ = cv2.estimateAffinePartial2D(then, now)
    if matrix is None:
        return skin
    height, width = skin.shape
    return cv2.warpAffine(skin, matrix, (width, height), flags=cv2.INTER_NEAREST)


def matching_hand(points: np.ndarray, candidates: list[np.ndarray]) -> np.ndarray | None:
    """La mano de un frame anterior que corresponde a esta: la más cercana, si no está lejos."""
    if not candidates:
        return None
    center = points.mean(axis=0)
    size = np.ptp(points, axis=0).max()
    best = min(candidates, key=lambda hand: np.linalg.norm(hand.mean(axis=0) - center))
    return best if np.linalg.norm(best.mean(axis=0) - center) < size else None


def hand_mask(skin: np.ndarray, hands_now: list[np.ndarray], hands_then: list[np.ndarray]) -> np.ndarray:
    """Une la silueta de cada mano: la piel segmentada que cae dentro de su zona.

    `skin` se segmentó cuando las manos estaban en `hands_then`; cada mano la mueve a su
    posición actual. Así se descartan la piel del cuello, de los brazos lejos de la mano o
    de otras personas, que no tienen una mano detectada encima.
    """
    mask = np.zeros_like(skin)
    for points in hands_now:
        then = matching_hand(points, hands_then)
        moved = skin if then is None else follow_hand(skin, then, points)
        mask |= cv2.bitwise_and(moved, hand_region(points, skin.shape))
    return mask


class _SkinSegmenter(threading.Thread):
    """Segmenta la piel en segundo plano: tarda ~110 ms, demasiado para hacerlo en cada frame."""

    def __init__(self, segmenter: vision.ImageSegmenter) -> None:
        super().__init__(daemon=True)
        self._segmenter = segmenter
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._pending: tuple[np.ndarray, list[np.ndarray]] | None = None
        self._result: tuple[np.ndarray, list[np.ndarray]] | None = None
        self._running = True

    def submit(self, rgb: np.ndarray, hands: list[np.ndarray]) -> None:
        """Pide segmentar este frame; si había otro esperando, se descarta por viejo."""
        with self._lock:
            self._pending = (rgb, hands)
        self._wake.set()

    @property
    def latest(self) -> tuple[np.ndarray, list[np.ndarray]] | None:
        """La última piel segmentada y las manos que había en ese frame."""
        with self._lock:
            return self._result

    def run(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            if not self._running:
                return
            with self._lock:
                job, self._pending = self._pending, None
            if job is None:
                continue
            rgb, hands = job
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            categories = self._segmenter.segment(image).category_mask.numpy_view()
            skin = np.where(categories == BODY_SKIN, 255, 0).astype(np.uint8)
            if skin.shape != rgb.shape[:2]:
                skin = cv2.resize(skin, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            with self._lock:
                self._result = (skin, hands)

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        self.join(timeout=2)
        self._segmenter.close()


class HandSilhouetteDetector:
    """Detecta la silueta de hasta dos manos, aunque estén delante de la cara."""

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self._landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(models_dir / HAND_MODEL)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
            )
        )
        segmenter = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=str(models_dir / SEGMENTER_MODEL)),
                running_mode=vision.RunningMode.IMAGE,
                output_category_mask=True,
            )
        )
        self._skin = _SkinSegmenter(segmenter)
        self._skin.start()
        self._start = time.perf_counter()
        self._last_timestamp = -1
        self._last_submit = -SEGMENT_INTERVAL
        self.hands: list[np.ndarray] = []

    def apply(self, frame: np.ndarray, params: object = None) -> np.ndarray:
        """Devuelve una máscara binaria con la silueta de las manos del frame BGR."""
        height, width = frame.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # El modo VIDEO exige marcas de tiempo estrictamente crecientes.
        timestamp = max(int((time.perf_counter() - self._start) * 1000), self._last_timestamp + 1)
        self._last_timestamp = timestamp
        result = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp
        )
        self.hands = [
            np.array([[point.x * width, point.y * height] for point in hand], np.float32)
            for hand in result.hand_landmarks
        ]
        if not self.hands:
            return np.zeros((height, width), np.uint8)
        now = time.perf_counter()
        if now - self._last_submit >= SEGMENT_INTERVAL:
            self._last_submit = now
            self._skin.submit(rgb, self.hands)
        latest = self._skin.latest
        if latest is None or latest[0].shape != (height, width):
            return np.zeros((height, width), np.uint8)
        skin, hands_then = latest
        return hand_mask(skin, self.hands, hands_then)

    def close(self) -> None:
        self._skin.stop()
        self._landmarker.close()
