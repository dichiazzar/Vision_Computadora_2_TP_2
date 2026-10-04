"""Evaluación de los gestos EN VIDEO con clips sintéticos hechos con manos reales de HaGRID.

Cada clip (20 cuadros, ~1.3 s) pega una mano real recortada de HaGRID sobre la mesa (zona del Jugador), rotada al
azar (con cámara cenital la mano aparece en cualquier orientación), y la deja QUIETA o la hace BARRER la mesa.
Se pasa cuadro a cuadro por el ReconocedorDeGestos (MediaPipe en modo video + reglas + lógica temporal) y se
cuenta qué gesto se disparó. Lo esperado:

    one   quieta  -> PEDIR          palm/stop/four barriendo -> PLANTARSE
    palm  quieta  -> PLANTARSE (palma sostenida ~1.2 s; con plantarse="barrido" no dispararía)
    fist, like, dislike, peace, call, three (quietas o barriendo) -> nada

Uso (desde la carpeta del repo; usa las imágenes que baja evaluar_gestos.py):
    python Gestos/simular_gestos.py
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parent
sys.path.insert(0, str(AQUI))
from gestos import DetectorDeManos, ReconocedorDeGestos   # noqa: E402

ESPERADO = {("one", "quieta"): "PEDIR", ("palm", "barrido"): "PLANTARSE", ("stop", "barrido"): "PLANTARSE",
            ("four", "barrido"): "PLANTARSE", ("palm", "quieta"): "PLANTARSE"}   # palma sostenida (modo "ambos")
CASOS = [("one", "quieta"), ("palm", "barrido"), ("stop", "barrido"), ("four", "barrido"), ("palm", "quieta"),
         ("fist", "quieta"), ("like", "quieta"), ("dislike", "quieta"), ("peace", "quieta"), ("call", "quieta"),
         ("three", "quieta"), ("fist", "barrido"), ("peace", "barrido")]


def recortar_mano(img, detector):
    puntos = detector.detectar(img)
    if not puntos:
        return None
    p = puntos[0]
    x1, y1 = p.min(0)
    x2, y2 = p.max(0)
    m = 0.8 * max(x2 - x1, y2 - y1)          # contexto: el detector de palmas de MediaPipe lo necesita
    x1, y1 = int(max(0, x1 - m)), int(max(0, y1 - m))
    x2, y2 = int(min(img.shape[1], x2 + m)), int(min(img.shape[0], y2 + m))
    return img[y1:y2, x1:x2]


def clip(mesa, mano, movimiento, angulo, rng, n=20, alto_mano=120):
    """Genera los cuadros: la mano (escalada a ~alto_mano px y rotada) quieta o cruzando la zona del Jugador."""
    esc = min(alto_mano / max(mano.shape[:2]), 1.5)   # no agrandar de más los recortes chicos (borrosos)
    mano = cv2.resize(mano, None, fx=esc, fy=esc)
    h, w = mano.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angulo, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    W, H = int(h * sin + w * cos), int(h * cos + w * sin)
    M[:, 2] += (W / 2 - w / 2, H / 2 - h / 2)
    mano = cv2.warpAffine(mano, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    alto, ancho = mesa.shape[:2]
    y = int(alto * 0.5 + rng.uniform(0, max(1.0, alto * 0.5 - H - 5)))
    x0 = int(rng.uniform(10, max(11, ancho - 2.6 * W)))
    cuadros = []
    for k in range(n):
        x = x0 + (int(k * 1.6 * W / n) if movimiento == "barrido" else int(rng.normal(0, 2)))
        x = int(np.clip(x, 0, ancho - W))
        f = mesa.copy()
        f[y:y + H, x:x + W] = mano
        cuadros.append(f)
    return cuadros


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default=str(AQUI / "datos_hagrid"))
    ap.add_argument("--n", type=int, default=15, help="manos por caso")
    ap.add_argument("--salida", default=str(REPO / "Experimentos" / "resultados" / "gestos_video.csv"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    mesa = cv2.imread(str(REPO / "prueba.jpeg"))
    # cuadro de 640x360 con la mano de ~1/3 del alto: la proporción de una cámara a 40-60 cm de la mesa
    # (MediaPipe busca palmas en una versión reducida del cuadro: lo que importa es el tamaño RELATIVO)
    mesa = cv2.resize(mesa, (640, 360))
    recortador = DetectorDeManos(num_manos=1, modo="imagen")

    filas = []
    for clase, movimiento in CASOS:
        disparos = Counter()
        archivos = sorted((Path(args.datos) / clase).glob("*.jpg"))
        usadas = 0
        for f in archivos:
            if usadas >= args.n:
                break
            mano = recortar_mano(cv2.imread(str(f)), recortador)
            if mano is None or min(mano.shape[:2]) < 30:
                continue
            usadas += 1
            angulo = float(rng.uniform(0, 360))
            rec = ReconocedorDeGestos(DetectorDeManos(num_manos=2, modo="video"))
            gestos = [g for g in (rec.procesar(c) for c in clip(mesa, mano, movimiento, angulo, rng)) if g]
            disparos[gestos[0] if gestos else "nada"] += 1
        esperado = ESPERADO.get((clase, movimiento), "nada")
        filas.append(dict(clase=clase, movimiento=movimiento, esperado=esperado, clips=usadas,
                          acierto=disparos[esperado] / max(usadas, 1),
                          PEDIR=disparos["PEDIR"], PLANTARSE=disparos["PLANTARSE"], nada=disparos["nada"]))
        print(f"  {clase:8s} {movimiento:8s} esperado {esperado:10s} -> {dict(disparos)}", flush=True)

    import pandas as pd
    df = pd.DataFrame(filas)
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.salida, index=False, float_format="%.3f")
    pos = df[df.esperado != "nada"]
    neg = df[df.esperado == "nada"]
    print(f"\nGestos detectados (recall): {(pos.acierto * pos.clips).sum() / pos.clips.sum():.1%} "
          f"| falsos disparos en clips sin gesto: {((1 - neg.acierto) * neg.clips).sum() / neg.clips.sum():.1%}")
    print("Resultados en", args.salida)


if __name__ == "__main__":
    main()
