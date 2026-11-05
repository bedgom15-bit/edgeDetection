"""Descarga los modelos de los modos cerca y manos (unos 118 MB en total)."""

import hashlib
import sys
import urllib.request

from depth_detection import DEPTH_MODEL
from hand_detection import HAND_MODEL, MODELS_DIR, SEGMENTER_MODEL

BASE_URL = "https://storage.googleapis.com/mediapipe-models"
# Versión fija del modelo de distancia, para que su huella SHA-256 no cambie.
DEPTH_REVISION = "4472b7362082ad9968fee890ca0f1e5aca36b93d"
URLS = {
    DEPTH_MODEL: f"https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/{DEPTH_REVISION}/onnx/model.onnx",
    HAND_MODEL: f"{BASE_URL}/hand_landmarker/hand_landmarker/float16/latest/{HAND_MODEL}",
    SEGMENTER_MODEL: f"{BASE_URL}/image_segmenter/selfie_multiclass_256x256/float32/latest/{SEGMENTER_MODEL}",
}
SHA256 = {DEPTH_MODEL: "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c"}


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    missing = [name for name in URLS if not (MODELS_DIR / name).is_file()]
    if not missing:
        print(f"Los modelos ya estan en {MODELS_DIR}")
        return 0
    MODELS_DIR.mkdir(exist_ok=True)
    for name in missing:
        target = MODELS_DIR / name
        print(f"Descargando {name}...")
        # Se descarga a un archivo temporal para no dejar un modelo a medias si falla.
        partial = target.with_suffix(target.suffix + ".part")
        try:
            urllib.request.urlretrieve(URLS[name], partial)
        except OSError as error:
            print(f"Error: no se pudo descargar {name}: {error}", file=sys.stderr)
            return 1
        if name in SHA256 and sha256(partial) != SHA256[name]:
            partial.unlink()
            print(f"Error: {name} no coincide con el archivo esperado; no se guardo.", file=sys.stderr)
            return 1
        partial.replace(target)
        print(f"  {target.stat().st_size / 1_000_000:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
