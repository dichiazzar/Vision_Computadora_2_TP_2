"""Experimento de robustez ante el FONDO, con escenas compuestas de cartas reales.

Reemplazar el fondo de una foto con una máscara de color no es confiable (ver robustez.py). Acá se hace al revés:
se recortan cartas REALES del mazo de la demo, limpias y con su máscara exacta, del último cuadro de las partidas
reales del registro (cartas quietas sobre la mesa de madera, sin manos), y se pegan sobre distintos fondos. Las
esquinas de cada carta (cajas reales) se conocen y se transforman junto con la carta.

Todas las escenas de un fondo usan EXACTAMENTE las mismas cartas, posiciones y rotaciones que las de los otros
fondos (misma semilla por escena): lo único que cambia es el fondo. Así la diferencia de desempeño entre fondos se
debe sólo al fondo.

Fondos: verde (paño), oscuro, blanco (casi del color de la carta), textura, desorden (rectángulos de colores),
ruido, mármol (de prueba.jpeg) y madera (la mesa de las partidas: es el fondo "conocido").

Uso (desde la carpeta del repo):
    python Experimentos/robustez_fondo.py                 # 30 escenas por fondo, configuraciones A, B, A_propio, B_propio
Resultados: Experimentos/resultados/robustez_fondo.csv, .png y robustez_fondo_ejemplos.jpg
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Experimentos"))
sys.path.insert(0, str(REPO / "Identificador_Cartas"))
sys.path.insert(0, str(REPO / "Dataset Real"))

import detectores as D                         # noqa: E402
import evaluar_real as E                       # noqa: E402
from extraer_cuadros import zona_util          # noqa: E402
from robustez import configuraciones, hacer_fondo   # noqa: E402

TRABAJO = REPO / "Experimentos" / "robustez_fondo_data"
FONDOS = ["madera", "verde", "oscuro", "blanco", "textura", "desorden", "ruido", "marmol"]
ALTO, ANCHO = 720, 508          # mismo formato que la cámara de la demo (celular vertical, sin franjas)


# ============================================================
# 1. RECORTE DE CARTAS REALES
# ============================================================

def mascara_blanca(img):
    """Cartas (papel blanco: poca saturación, mucho brillo) sobre la mesa de madera."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] < 60) & (hsv[..., 2] > 140)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    return m


def extraer_cartas(videos, detector):
    """[(recorte BGR, máscara, [(valor, x1, y1, x2, y2) en coords del recorte])] de cartas aisladas y completas."""
    cartas = []
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        ultimo = None
        while True:
            ok, f = cap.read()
            if not ok:
                break
            ultimo = f
        x1, x2, y1, y2 = zona_util(ultimo)
        img = ultimo[y1:y2, x1:x2]
        esquinas = detector.detectar(img)
        m = mascara_blanca(img)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = img.shape[:2]
        for c in cnts:
            area = cv2.contourArea(c)
            (_, _), (rw, rh), _ = cv2.minAreaRect(c)
            if area < 0.02 * h * w or area > 0.2 * h * w:
                continue
            if area / max(rw * rh, 1) < 0.85:            # no es un rectángulo limpio (cartas pegadas o tapadas)
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if bx <= 2 or by <= 2 or bx + bw >= w - 2 or by + bh >= h - 2:     # cortada por el borde
                continue
            dentro = [e for e in esquinas if cv2.pointPolygonTest(c, tuple(map(float, e.centroid)), False) > 0]
            if len(dentro) != 2 or dentro[0].rank != dentro[1].rank or min(e.conf for e in dentro) < 0.6:
                continue                                 # sólo cartas con sus dos esquinas claras y del mismo valor
            mask = np.zeros((h, w), np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
            gts = [(e.rank, e.sup_izq[0] - bx, e.sup_izq[1] - by, e.inf_der[0] - bx, e.inf_der[1] - by) for e in dentro]
            cartas.append((img[by:by + bh, bx:bx + bw].copy(), mask[by:by + bh, bx:bx + bw].copy(), gts,
                           f"{Path(video).stem}"))
    return cartas


# ============================================================
# 2. FONDOS Y ESCENAS
# ============================================================

def mosaico(parche, h, w):
    """Repite un parche en espejo hasta cubrir h x w (textura de mesa real)."""
    fila = np.hstack([parche, cv2.flip(parche, 1)])
    bloque = np.vstack([fila, cv2.flip(fila, 0)])
    reps = (h // bloque.shape[0] + 1, w // bloque.shape[1] + 1, 1)
    return np.tile(bloque, reps)[:h, :w].copy()


def fondo_real(nombre, h, w):
    if nombre == "marmol":
        return mosaico(cv2.imread(str(REPO / "prueba.jpeg"))[320:880, 1000:1340], h, w)
    if nombre == "madera":    # mesa de las partidas reales: un cuadro con la mesa vacía, antes del reparto
        cap = cv2.VideoCapture(str(REPO / "Experimentos" / "partidas_reales" / "crudo_9.mp4"))
        for _ in range(40):   # los primeros cuadros son negros (la cámara recién arranca)
            ok, g = cap.read()
            if not ok:
                break
            x1, x2, y1, y2 = zona_util(g)
            g = g[y1:y2, x1:x2]
            if g.mean() > 60:
                return mosaico(g[int(g.shape[0] * 0.4):int(g.shape[0] * 0.6), :], h, w)
        raise RuntimeError("no se encontró un cuadro con la mesa vacía")
    raise ValueError(nombre)


def rotar_carta(img, mask, gts, angulo, escala):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angulo, escala)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    W, H = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[:, 2] += (W / 2 - w / 2, H / 2 - h / 2)
    img_r = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR)
    mask_r = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST)
    nuevos = []
    for r, x1, y1, x2, y2 in gts:
        pts = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], np.float32) @ M.T
        nuevos.append((r, pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))
    return img_r, mask_r, nuevos


def disposicion(cartas, rng, n_min=3, n_max=5, intentos=200):
    """Elige cartas, rotación, escala y posición (sin superponerse) para una escena."""
    n = int(rng.integers(n_min, n_max + 1))
    elegidas = rng.choice(len(cartas), size=min(n, len(cartas)), replace=False)
    colocadas, ocupado = [], np.zeros((ALTO, ANCHO), bool)
    for k in elegidas:
        img, mask, gts, _ = cartas[k]
        img_r, mask_r, gts_r = rotar_carta(img, mask, gts, float(rng.uniform(-25, 25)), float(rng.uniform(0.9, 1.1)))
        H, W = mask_r.shape
        for _ in range(intentos):
            x, y = int(rng.integers(0, max(ANCHO - W, 1))), int(rng.integers(0, max(ALTO - H, 1)))
            zona = ocupado[y:y + H, x:x + W]
            if zona.shape == mask_r.shape and not (zona & (mask_r > 0)).any():
                ocupado[y:y + H, x:x + W] |= mask_r > 0
                colocadas.append((img_r, mask_r, gts_r, x, y))
                break
    return colocadas


def componer(fondo, colocadas, rng_ruido):
    escena = fondo.copy()
    gts = []
    for img_r, mask_r, gts_r, x, y in colocadas:
        H, W = mask_r.shape
        zona = escena[y:y + H, x:x + W]
        zona[mask_r > 0] = img_r[mask_r > 0]
        gts += [(r, x1 + x, y1 + y, x2 + x, y2 + y) for r, x1, y1, x2, y2 in gts_r]
    ruido = rng_ruido.normal(0, 3, escena.shape)                 # ruido de sensor leve, igual en todos los fondos
    escena = np.clip(escena.astype(np.float32) + ruido, 0, 255).astype(np.uint8)
    return escena, gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--escenas", type=int, default=30, help="escenas por fondo")
    ap.add_argument("--configs", nargs="+", default=["A", "B", "A_propio", "B_propio"])
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--salida", default=str(REPO / "Experimentos" / "resultados"))
    args = ap.parse_args()

    videos = sorted((REPO / "Experimentos" / "partidas_reales").glob("*.mp4"))
    cartas = extraer_cartas(videos, D.DetectorA(conf=0.5, pesos=D.pesos_finetune("propio")[0]))
    print(f"{len(cartas)} cartas reales recortadas: {sorted(c[2][0][0] for c in cartas)}", flush=True)
    if len(cartas) < 3:
        sys.exit("muy pocas cartas recortadas")

    # escenas: misma disposición (semilla por escena) en todos los fondos
    if TRABAJO.exists():
        shutil.rmtree(TRABAJO)
    muestras = {f: [] for f in FONDOS}
    for i in range(args.escenas):
        colocadas = disposicion(cartas, np.random.default_rng([0, i]))
        for f in FONDOS:
            rng_f = np.random.default_rng([1, i, FONDOS.index(f)])
            fondo = fondo_real(f, ALTO, ANCHO) if f in ("marmol", "madera") else hacer_fondo(f, ALTO, ANCHO, rng_f)
            escena, gts = componer(fondo, colocadas, np.random.default_rng([2, i]))
            ruta = TRABAJO / f / f"escena_{i:03d}.jpg"
            ruta.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ruta), escena, [cv2.IMWRITE_JPEG_QUALITY, 95])
            muestras[f].append((ruta, gts))

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    ejemplo = [cv2.resize(cv2.imread(str(muestras[f][0][0])), (254, 360)) for f in FONDOS]
    for im, f in zip(ejemplo, FONDOS):
        cv2.putText(im, f, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(im, f, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(salida / "robustez_fondo_ejemplos.jpg"), np.vstack([np.hstack(ejemplo[:4]), np.hstack(ejemplo[4:])]))
    n_esq = sum(len(g) for _, g in muestras[FONDOS[0]])
    print(f"{args.escenas} escenas por fondo, {n_esq} esquinas por fondo", flush=True)

    import pandas as pd
    filas = []
    for nombre, crear in configuraciones(args.configs):
        det = crear()
        t0 = time.time()
        for f in FONDOS:
            r, _, _ = E.evaluar(det, muestras[f], "original", args.conf)
            filas.append(dict(config=nombre, fondo=f, **{k: r[k] for k in ("mAP50", "P", "R", "F1", "R_loc", "ms_imagen")}))
            print(f"  {nombre:9s} {f:9s} mAP50 {r['mAP50']:.3f} P {r['P']:.3f} R {r['R']:.3f} F1 {r['F1']:.3f} "
                  f"R_loc {r['R_loc']:.3f}", flush=True)
        print(f"  {nombre}: {(time.time() - t0) / 60:.1f} min", flush=True)
    df = pd.DataFrame(filas)
    df.to_csv(salida / "robustez_fondo.csv", index=False, float_format="%.4f")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tabla = df.pivot(index="fondo", columns="config", values="F1").reindex(FONDOS)
    ax = tabla.plot.bar(figsize=(8, 3.6), rot=0)
    ax.set_ylabel("F1 (valor correcto)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Robustez ante el fondo - {args.escenas} escenas por fondo, mismas cartas y posiciones")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(salida / "robustez_fondo.png", dpi=130)
    print("\nF1 por fondo:\n", tabla.round(3))


if __name__ == "__main__":
    main()
