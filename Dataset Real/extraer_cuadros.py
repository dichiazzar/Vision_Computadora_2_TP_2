"""Extrae cuadros de videos CRUDOS (grabados con --guardar-crudo) para usarlos como fotos de entrenamiento.

Toma un cuadro cada `--cada` segundos y descarta:
  - los casi iguales al último guardado (la mesa no cambió: no aportan nada nuevo);
  - los movidos o desenfocados (nitidez muy por debajo de la mediana del video).
Los guarda en Dataset Real/fotos/<split>/, donde los toma prelabel.py.

IMPORTANTE: separar por sesión, no al azar. Todos los cuadros de un mismo video van al mismo split; el test tiene
que ser de otra sesión (otra luz, otro momento). Si no, cuadros casi iguales quedan en train y en test.

Uso (desde la carpeta del repo):
    python "Dataset Real/extraer_cuadros.py" videos/crudo_sesion1.mp4 videos/crudo_sesion2.mp4 --split train
    python "Dataset Real/extraer_cuadros.py" videos/crudo_sesion3.mp4 --split valid
    python "Dataset Real/extraer_cuadros.py" videos/crudo_sesion4.mp4 --split test
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def miniatura(frame):
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)


def nitidez(frame):
    return cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def candidatos(video, cada):
    """Cuadros cada `cada` segundos: [(indice, cuadro)]."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    paso = max(1, int(round(cada * fps)))
    salida, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % paso == 0:
            salida.append((i, f))
        i += 1
    cap.release()
    return salida, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--split", required=True, choices=["train", "valid", "test"])
    ap.add_argument("--cada", type=float, default=1.5, help="segundos entre cuadros candidatos")
    ap.add_argument("--diferencia", type=float, default=6.0,
                    help="diferencia media mínima (0-255) con el último cuadro guardado")
    ap.add_argument("--nitidez", type=float, default=0.5,
                    help="se descartan los cuadros con nitidez < este factor x la mediana del video")
    ap.add_argument("--salida", default=str(HERE / "fotos"))
    args = ap.parse_args()

    destino = Path(args.salida) / args.split
    destino.mkdir(parents=True, exist_ok=True)
    total = 0
    for video in args.videos:
        cuadros, fps = candidatos(video, args.cada)
        if not cuadros:
            print(f"{video}: no se pudo leer")
            continue
        nit = [nitidez(f) for _, f in cuadros]
        minimo = args.nitidez * float(np.median(nit))
        ultimo, guardados, borrosos, repetidos = None, 0, 0, 0
        for (i, f), n in zip(cuadros, nit):
            if n < minimo:
                borrosos += 1
                continue
            m = miniatura(f)
            if ultimo is not None and float(np.abs(m - ultimo).mean()) < args.diferencia:
                repetidos += 1
                continue
            cv2.imwrite(str(destino / f"{Path(video).stem}_{i:06d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
            ultimo, guardados = m, guardados + 1
        total += guardados
        print(f"{Path(video).name}: {len(cuadros)} candidatos ({args.cada:g} s) -> {guardados} guardados "
              f"| {repetidos} repetidos | {borrosos} borrosos")
    print(f"\n{total} cuadros en {destino}")
    print('Siguiente paso: python "Dataset Real/prelabel.py" (pre-etiqueta para corregir en Roboflow)')


if __name__ == "__main__":
    main()
