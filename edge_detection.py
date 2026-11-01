"""Detección de bordes y contornos, sin dependencias de interfaz."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Params:
    canny_low: int = 50
    canny_high: int = 150
    blur_kernel: int = 5  # debe ser impar
    close_kernel: int = 3
    min_area: int = 500
    background_threshold: int = 25  # más bajo = más sensible a los cambios
    mask_kernel: int = 7


class ForegroundDetector:
    """Aprende cómo se ve la escena y detecta lo que aparece delante de ella."""

    SKIP_FRAMES = 15  # la autoexposición de la cámara tarda en estabilizarse
    LEARN_FRAMES = 30

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Olvida el fondo y vuelve a aprenderlo con los próximos frames."""
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.LEARN_FRAMES, detectShadows=True
        )
        self._frames_seen = 0
        self.foreground_ratio = 0.0

    @property
    def learning(self) -> bool:
        return self._frames_seen < self.SKIP_FRAMES + self.LEARN_FRAMES

    def apply(self, frame: np.ndarray, params: Params) -> np.ndarray:
        """Devuelve una máscara binaria de lo que no pertenece al fondo."""
        blurred = cv2.GaussianBlur(frame, (params.blur_kernel, params.blur_kernel), 0)
        if self.learning:
            self._frames_seen += 1
            if self._frames_seen > self.SKIP_FRAMES:
                self._subtractor.apply(blurred, learningRate=-1)
            return np.zeros(frame.shape[:2], np.uint8)

        self._subtractor.setVarThreshold(params.background_threshold)
        # Con tasa 0 el fondo no cambia: un objeto quieto sigue detectándose.
        raw = self._subtractor.apply(blurred, learningRate=0)
        # Las sombras se marcan con 127; el umbral las descarta.
        _, mask = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((params.mask_kernel, params.mask_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        self.foreground_ratio = cv2.countNonZero(mask) / mask.size
        return mask


def detect_edges(frame: np.ndarray, params: Params) -> np.ndarray:
    """Devuelve una imagen binaria con los bordes de un frame BGR."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (params.blur_kernel, params.blur_kernel), 0)
    edges = cv2.Canny(blurred, params.canny_low, params.canny_high)
    # El cierre une bordes cortados para que los contornos queden completos.
    kernel = np.ones((params.close_kernel, params.close_kernel), np.uint8)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def find_contours(edges: np.ndarray, params: Params) -> list[np.ndarray]:
    """Devuelve los contornos externos con área de al menos `min_area`."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= params.min_area]


def draw_contours(frame: np.ndarray, contours: list[np.ndarray]) -> np.ndarray:
    """Devuelve una copia del frame con los contornos dibujados."""
    output = frame.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    return output
