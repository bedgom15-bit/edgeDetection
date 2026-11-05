# Detector de bordes y contornos

Detecta en tiempo real el contorno de lo que está más cerca de la webcam, de las manos y de los objetos que se ponen delante, con OpenCV, MediaPipe y Depth Anything V2.

## Cómo funciona

La webcam tiene cuatro modos, que se alternan con la tecla `e`:

- **Cerca (por defecto):** marca lo más cercano a la cámara, sea una mano, una taza o un teléfono, e ignora lo que hay detrás. Un modelo de IA (Depth Anything V2 small) estima a qué distancia está cada parte de la imagen. No hace falta aprender el fondo, así que da igual lo que haya delante al iniciar.
- **Manos:** dibuja la silueta de hasta dos manos, incluso delante de la cara o con gente detrás. Usa dos modelos de MediaPipe: uno encuentra cada mano y 21 puntos de ella, y otro separa la piel del cuerpo de la piel de la cara. La silueta es la piel que cae alrededor de cada mano encontrada. El segundo modelo es lento (unos 110 ms), así que funciona en segundo plano unas 5 veces por segundo, y entre medias la silueta se mueve con los puntos de la mano.
- **Objetos:** al iniciar aprende cómo se ve la escena durante unos 1,5 segundos. Después marca lo que cambió respecto a ella (sustracción de fondo). Si había algo delante al iniciar y luego lo mueves, el programa reconoce el hueco que deja y lo incorpora al fondo.
- **Bordes:** detecta bordes por contraste (Canny). Es el único modo para imágenes sueltas (`--image`), que no tienen un fondo con el que comparar.

Limitaciones del modo cerca:

- La distancia es relativa: el modelo sabe qué está más cerca que qué, pero no mide centímetros. Siempre se marca lo más cercano de la imagen: si no acercas nada, lo más cercano eres tú, y se marca tu silueta.
- El modelo tarda unos 140 ms, así que la silueta se actualiza unas 7 veces por segundo y va unos 185 ms por detrás del objeto si lo mueves rápido. La imagen sigue a 30 FPS.
- Los bordes de la silueta son aproximados: el modelo trabaja a 252×196 píxeles.

Limitaciones del modo manos:

- Solo detecta piel: con guantes no hay silueta.
- Detecta también las manos de otras personas, como mucho dos en total.
- Si la mano se mueve muy rápido, la silueta puede quedarse un poco atrás.

Limitaciones del modo objetos:

- La cámara debe estar quieta. Si se mueve, pulsa `b` para reaprender el fondo.
- Si tú estás en la imagen al aprender el fondo, cuando te muevas se marcarán partes de tu cuerpo, y tu mano no se distinguirá bien delante de tu cara o tu ropa. Para manos, usa el modo manos.
- Los cambios de luz y de exposición automática de la cámara se compensan: si una zona tiene la misma textura que el fondo, solo con otro brillo, no cuenta como objeto. Sobre superficies lisas, como una pared blanca, no hay textura que comparar: si queda una zona marcada tras un cambio de luz, pulsa `b`.
- Todo lo que se mueva se detecta como objeto, incluidas las personas.
- El hueco que deja un objeto se reconoce porque el fondo que queda a la vista tiene muchos menos bordes que el objeto. Con fondos con mucha textura puede fallar y dejar una silueta; en ese caso, pulsa `b`.
- No mide la distancia: un objeto cercano solo se distingue porque ocupa más imagen. Sube `Area minima` para ignorar los pequeños o lejanos.
- No sabe qué es cada objeto, y dos objetos que se tocan pueden salir como un solo contorno.

## Instalación

Necesitas Python 3.12 o superior y una webcam. Solo se ha probado con Python 3.15 en Windows 11; en Mac y Linux debería funcionar, pero no está probado.

Descarga el proyecto, ya sea con **Code > Download ZIP** en esta página (y descomprímelo) o con `git clone`. Luego abre una terminal dentro de la carpeta del proyecto y crea un entorno virtual:

```bash
python -m venv .venv
```

Instala las dependencias:

```bash
.venv\Scripts\python -m pip install -r requirements.txt
```

Descarga los modelos de los modos cerca y manos (unos 118 MB): Depth Anything V2 small desde Hugging Face, y los de MediaPipe desde los servidores de Google:

```bash
.venv\Scripts\python descargar_modelos.py
```

Sin los modelos, el programa funciona igual en los modos objetos y bordes.

Depth Anything V2 small tiene licencia Apache 2.0. `descargar_modelos.py` comprueba su huella SHA-256 y no lo guarda si no coincide.

En Mac y Linux, cambia `.venv\Scripts\python` por `.venv/bin/python` en todos los comandos de esta página.

## Uso

Webcam:

```bash
.venv\Scripts\python main.py
```

Empezar en otro modo (`cerca`, `manos`, `objetos` o `bordes`):

```bash
.venv\Scripts\python main.py --mode objetos
```

Otra cámara (si la imagen sale oscura o da error, quizá sea la cámara IR):

```bash
.venv\Scripts\python main.py --camera 1
```

Imagen en una ventana:

```bash
.venv\Scripts\python main.py --image foto.jpg
```

Imagen guardando el resultado, sin abrir ventana:

```bash
.venv\Scripts\python main.py --image foto.jpg --output salida.png
```

Los valores iniciales se cambian con `--low`, `--high` y `--min-area`.

## Controles

En la ventana hay deslizadores para:

- `Canny bajo` y `Canny alto`: umbrales del modo bordes.
- `Area minima`: tamaño mínimo de un contorno.
- `Suavizado`: cuánto se suavizan los contornos para quitarles picos (0 = nada).
- `Umbral fondo` (solo webcam, modo objetos): sensibilidad a los cambios respecto al fondo.
- `Estabilidad` (solo webcam, modos manos y objetos; el modo cerca no la usa, porque retrasaría la silueta): cuánto se promedia la detección entre frames para que las líneas no parpadeen (0 = nada).
- `Cercania` (solo webcam, modo cerca): qué tan cerca de lo más cercano tiene que estar algo para marcarse, en % (más alto = solo lo más cercano).

| Tecla | Acción |
| --- | --- |
| `e` | Cambia de modo: cerca, manos, objetos, bordes (solo webcam) |
| `b` | Reaprende el fondo del modo objetos (solo webcam) |
| `m` | Cambia la vista: contornos, máscara o bordes, o lado a lado |
| `c` | Cambia de cámara (solo webcam) |
| `s` | Guarda una captura de la vista actual en la carpeta donde se ejecuta |
| `q` o `Esc` | Sale |

## Ajustes

- **Modo cerca, se marca también lo que está detrás del objeto (por ejemplo, tu cara):** sube `Cercania`.
- **Modo cerca, el objeto sale incompleto:** baja `Cercania`.
- **Modo objetos, aparecen manchas sin que haya nada:** sube `Umbral fondo` o `Area minima`.
- **Modo objetos, el objeto sale incompleto:** baja `Umbral fondo`.
- **Las líneas tiemblan o parpadean:** sube `Estabilidad` y `Suavizado`. Con mucha estabilidad, el contorno tarda un poco más en seguir a lo que se mueve.
- **Modo bordes, demasiados contornos:** sube los umbrales de Canny o el área mínima.
- **Modo bordes, contornos incompletos:** baja los umbrales de Canny.

## Pruebas

```bash
.venv\Scripts\python -m pytest
```
