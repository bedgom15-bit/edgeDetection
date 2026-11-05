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
    background_threshold: int = 25  # diferencia de color (0-255) con el fondo; más bajo = más sensible
    mask_kernel: int = 7
    stability: int = 80  # % de la detección anterior que se conserva (0 = sin promediar)
    smoothing: int = 5  # puntos a cada lado que se promedian en el contorno (0 = sin suavizar)
    nearness: int = 85  # modo cerca: % de lo más cercano a partir del cual algo cuenta como cerca


class MaskStabilizer:
    """Promedia la máscara entre frames para que los contornos no parpadeen."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._average: np.ndarray | None = None

    def apply(self, mask: np.ndarray, params: Params) -> np.ndarray:
        if params.stability == 0 or self._average is None or self._average.shape != mask.shape:
            self._average = mask.astype(np.float32)
            return mask
        cv2.accumulateWeighted(mask, self._average, 1 - params.stability / 100)
        _, stable = cv2.threshold(self._average, 127.5, 255, cv2.THRESH_BINARY)
        return stable.astype(np.uint8)


def smooth_contour(contour: np.ndarray, radius: int) -> np.ndarray:
    """Promedia cada punto con sus vecinos para quitar los picos del contorno."""
    points = contour.reshape(-1, 2).astype(np.float32)
    size = 2 * radius + 1
    if radius == 0 or len(points) < size:
        return contour
    # El contorno es cerrado: se extiende por ambos lados con sus propios puntos.
    padded = np.concatenate((points[-radius:], points, points[:radius]))
    kernel = np.ones(size, np.float32) / size
    smoothed = np.column_stack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in (0, 1)]
    )
    return np.round(smoothed).astype(np.int32).reshape(-1, 1, 2)


def _edge_strength(image: np.ndarray) -> np.ndarray:
    """Intensidad de borde de cada píxel, extendida unos píxeles a su alrededor."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    magnitude = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    # El contorno de la máscara puede quedar a unos píxeles del borde real.
    return cv2.dilate(magnitude, np.ones((5, 5), np.uint8))


class ForegroundDetector:
    """Aprende cómo se ve la escena y detecta lo que aparece delante de ella.

    Si algo que estaba al aprender el fondo se mueve, el hueco que deja (un "fantasma")
    se reconoce y se incorpora al fondo.
    """

    SKIP_FRAMES = 15  # la autoexposición de la cámara tarda en estabilizarse
    LEARN_FRAMES = 30
    # Lo que no es objeto se va mezclando con el fondo, para seguir cambios lentos de luz.
    ADAPT_RATE = 0.02
    # Se detecta a esta anchura como máximo: a resolución completa no llega a 30 FPS.
    PROCESS_WIDTH = 320
    # Las manchas de ruido más pequeñas no se revisan como posibles fantasmas.
    GHOST_MIN_AREA = 100
    # Un fantasma tiene en su contorno menos de esta fracción de los bordes del fondo.
    # Medido: fantasmas reales 0,01-0,12; una mano en movimiento (borrosa) 0,69-0,89.
    GHOST_EDGE_RATIO = 0.5
    # Una zona cuya textura coincide con la del fondo así de bien es un cambio de luz.
    LIGHTING_CORRELATION = 0.8
    # Por debajo de esta variación (una pared lisa) no hay textura que comparar.
    LIGHTING_MIN_TEXTURE = 5.0

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Olvida el fondo y vuelve a aprenderlo con los próximos frames."""
        self._background: np.ndarray | None = None
        self._frames_seen = 0
        self.foreground_ratio = 0.0

    @property
    def learning(self) -> bool:
        return self._frames_seen < self.SKIP_FRAMES + self.LEARN_FRAMES

    def apply(self, frame: np.ndarray, params: Params) -> np.ndarray:
        """Devuelve una máscara binaria de lo que no pertenece al fondo."""
        height, width = frame.shape[:2]
        small = frame
        if width > self.PROCESS_WIDTH:
            size = (self.PROCESS_WIDTH, round(height * self.PROCESS_WIDTH / width))
            small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        mask = self._detect(small, params)
        if small is not frame:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        self.foreground_ratio = cv2.countNonZero(mask) / mask.size
        return mask

    def _detect(self, frame: np.ndarray, params: Params) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, (params.blur_kernel, params.blur_kernel), 0).astype(np.float32)
        if self.learning:
            self._frames_seen += 1
            if self._frames_seen > self.SKIP_FRAMES:
                if self._background is None:
                    self._background = np.zeros_like(blurred)
                self._background += blurred / self.LEARN_FRAMES
            return np.zeros(frame.shape[:2], np.uint8)

        # La exposición automática de la cámara cambia el brillo de toda la imagen cuando
        # entra algo cerca: se ajusta el fondo a ese brillo antes de comparar.
        gain = self._exposure_gain(blurred)
        reference = self._background * gain
        blue, green, red = cv2.split(cv2.absdiff(blurred, reference))
        difference = cv2.max(cv2.max(blue, green), red)
        _, mask = cv2.threshold(difference, params.background_threshold, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        mask[self._shadows(blurred, reference)] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        kernel = np.ones((params.mask_kernel, params.mask_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        self._remove_ghosts(mask, blurred, reference, gain)

        cv2.accumulateWeighted(blurred, self._background, self.ADAPT_RATE, mask=cv2.bitwise_not(mask))
        return mask

    def _exposure_gain(self, blurred: np.ndarray) -> np.ndarray:
        """Cuánto más claro (>1) u oscuro (<1) está cada canal respecto al fondo."""
        # La mediana ignora los objetos mientras ocupen menos de la mitad de la imagen.
        ratio = (blurred[::4, ::4] + 1) / (self._background[::4, ::4] + 1)
        return np.median(ratio.reshape(-1, 3), axis=0).astype(np.float32)

    @staticmethod
    def _shadows(blurred: np.ndarray, background: np.ndarray) -> np.ndarray:
        # Una sombra oscurece el fondo casi por igual en los tres canales, sin cambiar su color.
        # Los límites son estrictos: con más margen, una mano sobre una pared de tono parecido
        # (piel más oscura que la pared, del mismo color) se confunde con una sombra.
        blue, green, red = cv2.split(cv2.divide(cv2.add(blurred, 1.0), cv2.add(background, 1.0)))
        darkest = cv2.min(cv2.min(blue, green), red)
        brightest = cv2.max(cv2.max(blue, green), red)
        return (darkest > 0.6) & (brightest < 0.95) & (brightest - darkest < 0.05)

    def _remove_ghosts(
        self, mask: np.ndarray, blurred: np.ndarray, reference: np.ndarray, gain: np.ndarray
    ) -> None:
        """Quita de la máscara los fantasmas y los cambios de luz, y los incorpora al fondo."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [c for c in contours if cv2.contourArea(c) >= self.GHOST_MIN_AREA]
        if not contours:
            return
        current_edges = _edge_strength(blurred)
        background_edges = _edge_strength(reference)
        current_gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        ghosts = [
            contour
            for contour in contours
            if self.looks_like_ghost(contour, current_edges, background_edges)
            or self._same_texture(contour, current_gray, reference_gray)
        ]
        if not ghosts:
            return
        region = np.zeros_like(mask)
        cv2.drawContours(region, ghosts, -1, 255, cv2.FILLED)
        region = cv2.dilate(region, np.ones((5, 5), np.uint8))
        # Se guarda con el brillo del fondo, no con el de este frame.
        cv2.copyTo(blurred / gain, region, self._background)
        mask[region > 0] = 0

    @classmethod
    def looks_like_ghost(
        cls, contour: np.ndarray, current_edges: np.ndarray, background_edges: np.ndarray
    ) -> bool:
        """Indica si el contorno es el hueco que dejó algo que estaba al aprender el fondo."""
        # Un objeto real tiene su borde en la imagen actual; un fantasma solo en el fondo.
        # Hace falta que sea claramente más débil: un objeto que se mueve rápido sale borroso.
        xs, ys = contour[:, 0, 0], contour[:, 0, 1]
        current = current_edges[ys, xs].mean()
        return current < cls.GHOST_EDGE_RATIO * background_edges[ys, xs].mean()

    def _same_texture(
        self, contour: np.ndarray, current_gray: np.ndarray, reference_gray: np.ndarray
    ) -> bool:
        """Indica si dentro del contorno se ve el mismo fondo, solo con otra iluminación."""
        x, y, w, h = cv2.boundingRect(contour)
        inside = np.zeros((h, w), np.uint8)
        cv2.drawContours(inside, [contour], -1, 255, cv2.FILLED, offset=(-x, -y))
        inside = inside > 0
        current = current_gray[y : y + h, x : x + w][inside]
        reference = reference_gray[y : y + h, x : x + w][inside]
        if reference.std() < self.LIGHTING_MIN_TEXTURE or current.std() < 1e-3:
            return False
        return np.corrcoef(current, reference)[0, 1] > self.LIGHTING_CORRELATION


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
    # CHAIN_APPROX_NONE conserva todos los puntos, necesarios para suavizar bien.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c for c in contours if cv2.contourArea(c) >= params.min_area]


def draw_contours(frame: np.ndarray, contours: list[np.ndarray]) -> np.ndarray:
    """Devuelve una copia del frame con los contornos dibujados."""
    output = frame.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    return output
