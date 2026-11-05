"""Detector de bordes y contornos para la webcam o una imagen."""

import argparse
import sys
import time
from pathlib import Path
from typing import Protocol

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

WINDOW = "Contornos"
TRACKBAR_LOW = "Canny bajo"
TRACKBAR_HIGH = "Canny alto"
TRACKBAR_AREA = "Area minima"
TRACKBAR_SMOOTHING = "Suavizado"
TRACKBAR_BACKGROUND = "Umbral fondo"
TRACKBAR_STABILITY = "Estabilidad"
TRACKBAR_NEARNESS = "Cercania"
CANNY_MAX = 500
AREA_MAX = 20000
SMOOTHING_MAX = 20
BACKGROUND_MAX = 100
# Por encima de 95 la detección tarda demasiado en seguir a un objeto que se mueve.
STABILITY_MAX = 95
NEARNESS_MAX = 99
# Si casi toda la imagen parece nueva, lo más probable es que haya cambiado la luz.
RELEARN_HINT_RATIO = 0.7

VIEWS = ("contornos", "bordes", "lado a lado")
# Modos de la webcam, en el orden en que los recorre la tecla e.
MODES = ("cerca", "manos", "objetos", "bordes")
MAX_CAMERAS = 4
MAX_READ_FAILURES = 30
ESC = 27

# Media Foundation: en las pruebas, DirectShow dejó la cámara en 10 FPS en lugar de 30.
CAMERA_BACKEND = cv2.CAP_MSMF if sys.platform == "win32" else cv2.CAP_ANY


def fail(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return 1


def load_image(path: Path) -> np.ndarray | None:
    # imdecode en lugar de imread: imread no abre rutas con acentos en Windows.
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def save_image(path: Path, image: np.ndarray) -> bool:
    try:
        ok, data = cv2.imencode(path.suffix, image)
        if ok:
            data.tofile(path)
        return ok
    except (cv2.error, OSError):
        return False


def open_camera(index: int) -> cv2.VideoCapture | None:
    """Abre una cámara y confirma que entrega frames."""
    capture = cv2.VideoCapture(index, CAMERA_BACKEND)
    if capture.isOpened() and capture.read()[0]:
        return capture
    capture.release()
    return None


def open_next_camera(current: int) -> tuple[cv2.VideoCapture, int] | None:
    """Busca la siguiente cámara que funcione; la actual es el último intento."""
    for offset in range(1, MAX_CAMERAS + 1):
        index = (current + offset) % MAX_CAMERAS
        capture = open_camera(index)
        if capture is not None:
            return capture, index
    return None


def create_window(params: Params, camera_sliders: bool = False) -> None:
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar(TRACKBAR_LOW, WINDOW, params.canny_low, CANNY_MAX, lambda _: None)
    cv2.createTrackbar(TRACKBAR_HIGH, WINDOW, params.canny_high, CANNY_MAX, lambda _: None)
    cv2.createTrackbar(TRACKBAR_AREA, WINDOW, params.min_area, AREA_MAX, lambda _: None)
    cv2.createTrackbar(TRACKBAR_SMOOTHING, WINDOW, params.smoothing, SMOOTHING_MAX, lambda _: None)
    if camera_sliders:
        cv2.createTrackbar(
            TRACKBAR_BACKGROUND, WINDOW, params.background_threshold, BACKGROUND_MAX, lambda _: None
        )
        cv2.createTrackbar(TRACKBAR_STABILITY, WINDOW, params.stability, STABILITY_MAX, lambda _: None)
        cv2.createTrackbar(TRACKBAR_NEARNESS, WINDOW, params.nearness, NEARNESS_MAX, lambda _: None)


def read_trackbars(params: Params, camera_sliders: bool = False) -> None:
    params.canny_low = cv2.getTrackbarPos(TRACKBAR_LOW, WINDOW)
    params.canny_high = cv2.getTrackbarPos(TRACKBAR_HIGH, WINDOW)
    params.min_area = cv2.getTrackbarPos(TRACKBAR_AREA, WINDOW)
    params.smoothing = cv2.getTrackbarPos(TRACKBAR_SMOOTHING, WINDOW)
    if camera_sliders:
        # Con un umbral de 0, cualquier ruido de la cámara contaría como objeto.
        params.background_threshold = max(cv2.getTrackbarPos(TRACKBAR_BACKGROUND, WINDOW), 1)
        params.stability = cv2.getTrackbarPos(TRACKBAR_STABILITY, WINDOW)
        # Con 0, todo lo que el modelo vea contaría como cerca, incluido el fondo.
        params.nearness = max(cv2.getTrackbarPos(TRACKBAR_NEARNESS, WINDOW), 1)


class MaskDetector(Protocol):
    def apply(self, frame: np.ndarray, params: Params) -> np.ndarray: ...

    def close(self) -> None: ...


def load_near_detector() -> MaskDetector | None:
    """Carga el detector de lo más cercano, o explica qué falta y devuelve None."""
    from depth_detection import DEPTH_MODEL, MODELS_DIR, NearObjectDetector

    if not (MODELS_DIR / DEPTH_MODEL).is_file():
        print(
            f"Error: falta el modelo {DEPTH_MODEL}. Descargalo con: python descargar_modelos.py",
            file=sys.stderr,
        )
        return None
    print("Cargando el modelo de distancia...")
    return NearObjectDetector()


# Modos con un modelo que se carga al usarlo por primera vez: cargarlo tarda.
LOADERS = {"cerca": lambda: load_near_detector(), "manos": lambda: load_hand_detector()}


def load_hand_detector() -> MaskDetector | None:
    """Carga el detector de manos, o explica qué falta y devuelve None."""
    # Se importa aquí: MediaPipe tarda en cargar y los otros modos no lo necesitan.
    try:
        from hand_detection import HandSilhouetteDetector, missing_models
    except ImportError:
        print("Error: falta MediaPipe. Instalalo con: pip install -r requirements.txt", file=sys.stderr)
        return None
    missing = missing_models()
    if missing:
        print(
            f"Error: faltan los modelos {', '.join(missing)}. Descargalos con: python descargar_modelos.py",
            file=sys.stderr,
        )
        return None
    print("Cargando los modelos de manos...")
    return HandSilhouetteDetector()


def render(
    frame: np.ndarray,
    params: Params,
    view: str,
    detector: MaskDetector | None = None,
    stabilizer: MaskStabilizer | None = None,
) -> tuple[np.ndarray, int]:
    """Devuelve la imagen de la vista elegida y el número de contornos.

    Con `detector` la máscara sale de él (manos u objetos nuevos); sin él, bordes por contraste.
    Con `stabilizer` promedia la detección entre frames.
    """
    mask = detect_edges(frame, params) if detector is None else detector.apply(frame, params)
    if stabilizer is not None:
        mask = stabilizer.apply(mask, params)
    contours = [smooth_contour(c, params.smoothing) for c in find_contours(mask, params)]
    annotated = draw_contours(frame, contours)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if view == "contornos":
        return annotated, len(contours)
    if view == "bordes":
        return mask_bgr, len(contours)
    return np.hstack((annotated, mask_bgr)), len(contours)


def camera_status(
    index: int, fps: float, count: int, mode: str, detector: ForegroundDetector, near_ready: bool = True
) -> str:
    if mode == "bordes":
        return f"Cam {index} | {fps:.0f} FPS | Bordes: {count}"
    if mode == "manos":
        return f"Cam {index} | {fps:.0f} FPS | Manos: {count}"
    if mode == "cerca":
        if not near_ready:
            return "Calculando la distancia..."
        return f"Cam {index} | {fps:.0f} FPS | Cerca: {count}"
    if detector.learning:
        return "Aprendiendo el fondo: deja la escena sin objetos"
    status = f"Cam {index} | {fps:.0f} FPS | Objetos: {count}"
    if detector.foreground_ratio > RELEARN_HINT_RATIO:
        status += " | Pulsa b para reaprender"
    return status


def draw_status(image: np.ndarray, text: str) -> None:
    # Texto con contorno negro para que se lea sobre cualquier fondo.
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def save_snapshot(image: np.ndarray) -> None:
    path = Path(f"captura_{time.strftime('%Y%m%d_%H%M%S')}.png")
    if save_image(path, image):
        print(f"Captura guardada: {path}")
    else:
        print(f"No se pudo guardar la captura: {path}", file=sys.stderr)


def window_closed() -> bool:
    return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1


def activate_mode(mode: str, models: dict[str, MaskDetector]) -> str:
    """Carga el modelo del modo si hace falta; si no se puede, pasa al siguiente modo que funcione."""
    while mode in LOADERS and mode not in models:
        detector = LOADERS[mode]()
        if detector is not None:
            models[mode] = detector
        else:
            mode = MODES[(MODES.index(mode) + 1) % len(MODES)]
    return mode


def next_mode(mode: str, models: dict[str, MaskDetector]) -> str:
    """Pasa al modo siguiente, saltando los que no se pueden cargar."""
    return activate_mode(MODES[(MODES.index(mode) + 1) % len(MODES)], models)


def run_camera(index: int, params: Params, mode: str = "cerca") -> int:
    capture = open_camera(index)
    if capture is None:
        return fail(
            f"la camara {index} no esta disponible o no entrega imagen. "
            "Puede estar ocupada por otro programa, o ser la camara IR: "
            "prueba con otro indice, por ejemplo --camera 1."
        )

    create_window(params, camera_sliders=True)
    detector = ForegroundDetector()
    stabilizer = MaskStabilizer()
    # Los modelos de los modos cerca y manos se cargan al usarlos por primera vez.
    models: dict[str, MaskDetector] = {}
    mode = activate_mode(mode, models)
    view = 0
    failures = 0
    average_time = 0.0
    last_time = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                failures += 1
                if failures >= MAX_READ_FAILURES:
                    return fail(f"la camara {index} dejo de entregar imagen.")
                continue
            failures = 0

            now = time.perf_counter()
            # Se promedia el tiempo, no los FPS: promediar tasas infla el resultado si hay altibajos.
            frame_time = now - last_time
            average_time = frame_time if average_time == 0 else 0.9 * average_time + 0.1 * frame_time
            fps = 1 / max(average_time, 1e-6)
            last_time = now

            read_trackbars(params, camera_sliders=True)
            # No se estabiliza el modo bordes: promediar las líneas finas de Canny las borra.
            if mode == "cerca":
                # Sin estabilizador: la silueta ya cambia solo con cada cálculo de distancia, y
                # promediarla añadía ~130 ms de retraso (medido).
                image, count = render(frame, params, VIEWS[view], models[mode])
            elif mode in models:
                image, count = render(frame, params, VIEWS[view], models[mode], stabilizer)
            elif mode == "objetos":
                image, count = render(frame, params, VIEWS[view], detector, stabilizer)
            else:
                image, count = render(frame, params, VIEWS[view])
            near_ready = mode != "cerca" or models["cerca"].ready
            draw_status(image, camera_status(index, fps, count, mode, detector, near_ready))
            cv2.imshow(WINDOW, image)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ESC) or window_closed():
                return 0
            if key == ord("m"):
                view = (view + 1) % len(VIEWS)
            elif key == ord("s"):
                save_snapshot(image)
            elif key == ord("b"):
                detector.reset()
                stabilizer.reset()
            elif key == ord("e"):
                mode = next_mode(mode, models)
                detector.reset()
                stabilizer.reset()
            elif key == ord("c"):
                # Se libera antes de abrir otra: algunos controladores no permiten dos a la vez.
                capture.release()
                opened = open_next_camera(index)
                if opened is None:
                    return fail("no se encontro ninguna camara disponible.")
                capture, index = opened
                print(f"Usando la camara {index}")
                detector.reset()
                stabilizer.reset()
                last_time = time.perf_counter()
                average_time = 0.0
    finally:
        capture.release()
        for model in models.values():
            model.close()
        cv2.destroyAllWindows()


def run_image(image_path: Path, output: Path | None, params: Params) -> int:
    frame = load_image(image_path)
    if frame is None:
        return fail(f"no se pudo leer la imagen '{image_path}'.")

    if output is not None:
        image, count = render(frame, params, "contornos")
        if not save_image(output, image):
            return fail(f"no se pudo guardar el resultado en '{output}'.")
        print(f"Contornos: {count}. Resultado guardado en {output}")
        return 0

    create_window(params)
    view = 0
    try:
        while True:
            read_trackbars(params)
            image, count = render(frame, params, VIEWS[view])
            draw_status(image, f"Contornos: {count}")
            cv2.imshow(WINDOW, image)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), ESC) or window_closed():
                return 0
            if key == ord("m"):
                view = (view + 1) % len(VIEWS)
            elif key == ord("s"):
                save_snapshot(image)
    finally:
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    defaults = Params()
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="indice de la camara (por defecto: 0)")
    source.add_argument("--image", type=Path, help="procesar una imagen en lugar de la webcam")
    parser.add_argument("--output", type=Path, help="con --image: guardar el resultado sin abrir ventana")
    parser.add_argument(
        "--mode", choices=MODES, default="cerca", help="modo inicial de la webcam (por defecto: cerca)"
    )
    parser.add_argument("--low", type=int, default=defaults.canny_low, help="umbral bajo de Canny")
    parser.add_argument("--high", type=int, default=defaults.canny_high, help="umbral alto de Canny")
    parser.add_argument("--min-area", type=int, default=defaults.min_area, help="area minima del contorno en pixeles")
    args = parser.parse_args()
    if args.output is not None and args.image is None:
        parser.error("--output requiere --image")
    return args


def main() -> int:
    args = parse_args()
    params = Params(canny_low=args.low, canny_high=args.high, min_area=args.min_area)
    try:
        if args.image is not None:
            return run_image(args.image, args.output, params)
        return run_camera(args.camera, params, args.mode)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
