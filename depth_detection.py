"""Silueta de lo más cercano a la cámara, con un modelo que estima la distancia (Depth Anything V2)."""

import threading
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).parent / "models"
DEPTH_MODEL = "depth_anything_v2_small.onnx"
# Tamaño de entrada del modelo: múltiplo de 14 (sus parches) y de proporción parecida a la
# webcam. Medido en un i5-1145G7 con la webcam: 137 ms por cálculo, y la silueta sale de un
# frame de hace ~184 ms. A 196x140 era más rápido (86 ms), pero el modelo ya no distinguía a
# una persona del fondo: solo veía una rampa de abajo (cerca) a arriba (lejos).
INPUT_SIZE = (252, 196)
MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


def estimate_depth(net: cv2.dnn.Net, frame: np.ndarray) -> np.ndarray:
    """Devuelve la cercanía relativa de cada píxel (más alto = más cerca), al tamaño del modelo."""
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, INPUT_SIZE, mean=MEAN, swapRB=True)
    net.setInput(blob / STD)
    return net.forward()[0]


def near_mask(nearness: np.ndarray, shape: tuple[int, int], threshold: int) -> np.ndarray:
    """Máscara de lo que está cerca: lo que llega al `threshold` % de lo más cercano.

    El modelo no da distancias absolutas: en cada imagen, lo más cercano recibe el valor
    más alto, sea una mano o la cara. Por eso se compara con lo más cercano de la imagen.
    """
    # Un percentil alto en lugar del máximo: un solo píxel extremo no mueve el límite.
    nearest = np.percentile(nearness, 99.5)
    if nearest <= 0:
        return np.zeros(shape, np.uint8)
    resized = cv2.resize(nearness, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    mask = np.where(resized >= nearest * threshold / 100, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


class _DepthWorker(threading.Thread):
    """Estima la distancia en segundo plano: tarda demasiado para hacerlo en cada frame."""

    def __init__(self, net: cv2.dnn.Net) -> None:
        super().__init__(daemon=True)
        self._net = net
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._pending: np.ndarray | None = None
        self._result: np.ndarray | None = None
        self._running = True

    def submit(self, frame: np.ndarray) -> None:
        """Pide estimar este frame; si había otro esperando, se descarta por viejo."""
        with self._lock:
            self._pending = frame
        self._wake.set()

    @property
    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._result

    def run(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            if not self._running:
                return
            with self._lock:
                frame, self._pending = self._pending, None
            if frame is None:
                continue
            nearness = estimate_depth(self._net, frame)
            with self._lock:
                self._result = nearness

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        self.join(timeout=2)


class NearObjectDetector:
    """Detecta la silueta de lo más cercano a la cámara, ignorando lo que hay detrás.

    La silueta sale del último cálculo de distancia, que es de un frame de hace ~184 ms
    (medido). Moverla con el flujo óptico hasta el frame actual no la hizo coincidir mejor
    con lo que se ve (medido con la webcam) y costaba ~20 ms por frame, así que no se hace.
    """

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self._worker = _DepthWorker(cv2.dnn.readNetFromONNX(str(models_dir / DEPTH_MODEL)))
        self._worker.start()

    @property
    def ready(self) -> bool:
        """Indica si ya hay una primera estimación de distancia."""
        return self._worker.latest is not None

    def apply(self, frame: np.ndarray, params) -> np.ndarray:
        """Devuelve una máscara binaria de lo más cercano en el frame BGR."""
        self._worker.submit(frame)
        nearness = self._worker.latest
        if nearness is None:
            return np.zeros(frame.shape[:2], np.uint8)
        return near_mask(nearness, frame.shape[:2], params.nearness)

    def close(self) -> None:
        self._worker.stop()
