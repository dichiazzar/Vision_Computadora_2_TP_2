"""Genera un video de prueba a partir de una foto, simulando una cámara en mano sobre la mesa.

Sirve para probar el seguimiento y medir la estabilidad sin cámara ni cartas. Sobre la foto se aplica:
  - movimiento suave y continuo: rotación ±6°, zoom 0.92-1.08, traslación ±4% (suma de senoides);
  - desenfoque de movimiento cuando la "cámara" se mueve rápido;
  - cambios de brillo y contraste, ruido de sensor y compresión JPEG.
El movimiento es chico, así que cada carta sigue del mismo lado (Casa arriba / Jugador abajo): la mano real
es la misma en todo el video y se puede pasar con --esperado al identificador.

Uso (desde la carpeta del repo):
    python Experimentos/generar_video_prueba.py --foto prueba.jpeg --salida videos/prueba.mp4
    python Identificador_Cartas/identificador_cartas.py --video videos/prueba.mp4 --esperado "Q|7 K" --sin-ventana
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def senoide(t, rng, amplitud, n=3):
    """Suma de senoides con frecuencias y fases al azar: un movimiento suave, no periódico."""
    f = rng.uniform(0.05, 0.5, n)
    fase = rng.uniform(0, 2 * np.pi, n)
    a = rng.uniform(0.3, 1.0, n)
    return amplitud * np.sum(a[:, None] * np.sin(2 * np.pi * f[:, None] * t[None] + fase[:, None]), 0) / a.sum()


def desenfoque_movimiento(img, dx, dy):
    largo = int(min(25, np.hypot(dx, dy)))
    if largo < 3:
        return img
    k = np.zeros((largo, largo), np.float32)
    ang = np.arctan2(dy, dx)
    c = (largo - 1) / 2
    for i in range(largo):
        x = int(round(c + (i - c) * np.cos(ang)))
        y = int(round(c + (i - c) * np.sin(ang)))
        k[y, x] = 1
    return cv2.filter2D(img, -1, k / k.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--foto", default="prueba.jpeg")
    ap.add_argument("--salida", default=str(REPO / "videos" / "prueba.mp4"))
    ap.add_argument("--segundos", type=float, default=10)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--ancho", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    foto = Path(args.foto) if Path(args.foto).exists() else REPO / args.foto
    img = cv2.imread(str(foto))
    if img is None:
        raise FileNotFoundError(foto)
    alto = int(round(args.ancho * img.shape[0] / img.shape[1]))
    img = cv2.resize(img, (args.ancho, alto), interpolation=cv2.INTER_AREA)

    rng = np.random.default_rng(args.seed)
    n = int(args.segundos * args.fps)
    t = np.arange(n) / args.fps
    ang = senoide(t, rng, 6.0)
    esc = 1.0 + senoide(t, rng, 0.08)
    tx = senoide(t, rng, 0.04 * args.ancho)
    ty = senoide(t, rng, 0.04 * alto)
    brillo = senoide(t, rng, 25)
    contraste = 1.0 + senoide(t, rng, 0.15)

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(salida), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.ancho, alto))
    centro = (args.ancho / 2, alto / 2)
    for i in range(n):
        M = cv2.getRotationMatrix2D(centro, ang[i], esc[i])
        M[:, 2] += (tx[i], ty[i])
        cuadro = cv2.warpAffine(img, M, (args.ancho, alto), borderMode=cv2.BORDER_REFLECT)
        if i > 0:      # desenfoque proporcional a la velocidad de la "cámara"
            cuadro = desenfoque_movimiento(cuadro, (tx[i] - tx[i - 1]) * 2.5, (ty[i] - ty[i - 1]) * 2.5)
        cuadro = cv2.convertScaleAbs(cuadro, alpha=contraste[i], beta=brillo[i])
        ruido = rng.normal(0, 4, cuadro.shape).astype(np.int16)
        cuadro = np.clip(cuadro.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
        _, jpg = cv2.imencode(".jpg", cuadro, [cv2.IMWRITE_JPEG_QUALITY, 70])
        writer.write(cv2.imdecode(jpg, cv2.IMREAD_COLOR))
    writer.release()
    print(f"{n} cuadros ({args.segundos:.0f} s a {args.fps} FPS, {args.ancho}x{alto}) -> {salida}")


if __name__ == "__main__":
    main()
