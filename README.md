# Detector de bordes y contornos

Detecta en tiempo real el contorno de los objetos que se ponen delante de la webcam, con OpenCV.

## Cómo funciona

- **Webcam, modo objetos (por defecto):** al iniciar aprende cómo se ve la escena durante unos 1,5 segundos. Después marca lo que cambió respecto a ella (sustracción de fondo con MOG2). Deja la escena sin objetos mientras aprende.
- **Webcam, modo bordes (tecla `e`), e imágenes:** detecta bordes por contraste (Canny). Una sola imagen no tiene un fondo con el que comparar.

Limitaciones del modo objetos:

- La cámara debe estar quieta. Si se mueve o cambia la luz, pulsa `b` para reaprender el fondo.
- Todo lo que se mueva se detecta como objeto, incluida una persona que estaba en la escena al aprender el fondo.
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

En Mac y Linux, cambia `.venv\Scripts\python` por `.venv/bin/python` en todos los comandos de esta página.

## Uso

Webcam:

```bash
.venv\Scripts\python main.py
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

En la ventana hay deslizadores para el área mínima del contorno, los dos umbrales de Canny (modo bordes) y, en la webcam, `Umbral fondo` (modo objetos).

| Tecla | Acción |
| --- | --- |
| `b` | Reaprende el fondo (solo webcam) |
| `e` | Alterna entre modo objetos y modo bordes (solo webcam) |
| `m` | Cambia la vista: contornos, máscara o bordes, o lado a lado |
| `c` | Cambia de cámara (solo webcam) |
| `s` | Guarda una captura de la vista actual en la carpeta donde se ejecuta |
| `q` o `Esc` | Sale |

## Ajustes

- **Modo objetos, aparecen manchas sin que haya nada:** sube `Umbral fondo` o `Area minima`.
- **Modo objetos, el objeto sale incompleto:** baja `Umbral fondo`.
- **Modo bordes, demasiados contornos:** sube los umbrales de Canny o el área mínima.
- **Modo bordes, contornos incompletos:** baja los umbrales de Canny.

## Pruebas

```bash
.venv\Scripts\python -m pytest
```
