"""Experimentos de robustez: desempeño ante rotación, oclusión, iluminación y fondo (alcance acordado con el profesor).

Cada perturbación se aplica a un conjunto fijo de imágenes a varios niveles de severidad, y TODAS las configuraciones
ven exactamente la misma imagen perturbada (semilla determinada por condición e imagen). Las cajas reales se
transforman junto con la imagen cuando la perturbación es geométrica (rotación). Diseño tomado de la notebook de
Cristhian Pettico (TP_cartas_A_vs_B.ipynb), adaptado al evaluador del proyecto (Experimentos/evaluar_real.py).

| Familia     | Perturbación               | Niveles                                    | Qué simula                                 |
|-------------|----------------------------|--------------------------------------------|--------------------------------------------|
| Rotación    | giro alrededor del centro  | 15°-180°                                   | cámara o cartas giradas                    |
| Oclusión    | rectángulo sobre el índice | 10%-60% del índice, desde un borde al azar | otra carta, una ficha o la mano tapando    |
| Iluminación | brillo global              | x0.2-x2.5                                  | sub / sobreexposición                      |
|             | contraste                  | x0.6-x0.15                                 | luz difusa, cámara lavada                  |
|             | sombra (gradiente)         | luz mínima 60%-10%                         | iluminación no uniforme                    |
| (Fondo: ver robustez_fondo.py, escenas compuestas con cartas reales)                                          |

Métricas por condición: mAP50, precisión, recall, F1 (IoU >= 0.5 y valor correcto; conf >= 0.5) y R_loc (encontrar la
esquina, sin importar el valor). Al rotar, la caja real pasa a ser la caja alineada a los ejes que contiene la caja
rotada (algo más grande que el índice), por eso se usan mAP50 y F1 y no mAP50-95.

Uso (desde la carpeta del repo):
    python Experimentos/robustez.py --dataset propio                 # test real propio (30 imágenes, 412 esquinas)
    python Experimentos/robustez.py --dataset sintetico --n 100      # muestra del test sintético de YOLO-13
Resultados: Experimentos/resultados/robustez_<dataset>.csv y .png
"""

import argparse
import math
import random
import sys
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Experimentos"))
sys.path.insert(0, str(REPO / "Identificador_Cartas"))

import detectores as D                    # noqa: E402
import evaluar_real as E                  # noqa: E402

FILL = (114, 114, 114)
SEED = 0
DATASETS = {"propio": (REPO / "Dataset Real" / "roboflow", "test"),
            "teogopk": (REPO / "Dataset Real" / "teogopk", "test"),
            "sintetico": (REPO / "YOLO - 13" / "Dataset YOLO 13", "test")}


# ============================================================
# PERTURBACIONES (gts = [(valor, x1, y1, x2, y2), ...])
# ============================================================

def a_cajas(gts):
    return np.array([g[1:] for g in gts], np.float32).reshape(-1, 4)


def rotar(img, gts, nivel, rng):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), float(nivel), 1.0)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=FILL)
    if not gts:
        return out, gts
    x1, y1, x2, y2 = a_cajas(gts).T
    pts = np.stack([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], 0).transpose(2, 0, 1)
    pts = np.concatenate([pts, np.ones((*pts.shape[:2], 1))], 2) @ M.T
    nb = np.stack([pts[..., 0].min(1), pts[..., 1].min(1), pts[..., 0].max(1), pts[..., 1].max(1)], 1)
    completo = (nb[:, 2] - nb[:, 0]) * (nb[:, 3] - nb[:, 1])
    nb[:, [0, 2]] = nb[:, [0, 2]].clip(0, w)
    nb[:, [1, 3]] = nb[:, [1, 3]].clip(0, h)
    queda = (nb[:, 2] - nb[:, 0]) * (nb[:, 3] - nb[:, 1]) >= 0.6 * completo   # índices que quedaron fuera
    return out, [(g[0], *map(float, b)) for g, b, k in zip(gts, nb, queda) if k]


def ocluir(img, gts, nivel, rng):
    out = img.copy()
    for _, x1, y1, x2, y2 in gts:
        x1, y1, x2, y2 = int(x1), int(y1), int(math.ceil(x2)), int(math.ceil(y2))
        bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        lado = rng.integers(4)
        if lado == 0:
            cv2.rectangle(out, (x1, y1), (x2, y1 + round(bh * nivel)), color, -1)
        elif lado == 1:
            cv2.rectangle(out, (x1, y2 - round(bh * nivel)), (x2, y2), color, -1)
        elif lado == 2:
            cv2.rectangle(out, (x1, y1), (x1 + round(bw * nivel), y2), color, -1)
        else:
            cv2.rectangle(out, (x2 - round(bw * nivel), y1), (x2, y2), color, -1)
    return out, gts


def brillo(img, gts, nivel, rng):
    return np.clip(img.astype(np.float32) * nivel, 0, 255).astype(np.uint8), gts


def contraste(img, gts, nivel, rng):
    return np.clip(128 + (img.astype(np.float32) - 128) * nivel, 0, 255).astype(np.uint8), gts


def sombra(img, gts, nivel, rng):
    """Gradiente lineal de luz en dirección al azar: de `nivel` (lado oscuro) a 1 (lado iluminado)."""
    h, w = img.shape[:2]
    a = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (xx - w / 2) * np.cos(a) + (yy - h / 2) * np.sin(a)
    t = (t - t.min()) / (t.max() - t.min() + 1e-9)
    ganancia = nivel + (1 - nivel) * t
    return np.clip(img.astype(np.float32) * ganancia[..., None], 0, 255).astype(np.uint8), gts


def mascara_cartas(img, gts):
    """Máscara aproximada de las cartas (papel blanco: poca saturación y mucho brillo) + los índices dilatados."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] < 70) & (hsv[..., 2] > 150)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(min(h, w) * 0.02) | 1),) * 2)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    llena = np.zeros_like(m)
    for c in cnts:
        if cv2.contourArea(c) >= 0.002 * h * w:
            cv2.drawContours(llena, [c], -1, 255, -1)
    for _, x1, y1, x2, y2 in gts:
        dx, dy = (x2 - x1) * 0.25, (y2 - y1) * 0.25
        cv2.rectangle(llena, (int(x1 - dx), int(y1 - dy)), (int(x2 + dx), int(y2 + dy)), 255, -1)
    return llena > 0


def hacer_fondo(tipo, h, w, rng):
    if tipo == "verde":
        return np.full((h, w, 3), (40, 110, 30), np.uint8)        # paño de casino
    if tipo == "oscuro":
        return np.full((h, w, 3), (30, 45, 70), np.uint8)         # madera oscura
    if tipo == "blanco":
        return np.full((h, w, 3), (245, 245, 245), np.uint8)      # mismo color que la carta
    if tipo == "ruido":
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    if tipo == "textura":
        chico = rng.integers(0, 256, (max(h // 40, 2), max(w // 40, 2), 3), dtype=np.uint8)
        return cv2.resize(chico, (w, h), interpolation=cv2.INTER_CUBIC)
    if tipo == "desorden":
        bg = np.full((h, w, 3), 90, np.uint8)
        for _ in range(60):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            bw, bh = int(rng.integers(w // 20, w // 3)), int(rng.integers(h // 20, h // 3))
            cv2.rectangle(bg, (x, y), (x + bw, y + bh), tuple(int(c) for c in rng.integers(0, 256, 3)), -1)
        return bg
    raise ValueError(tipo)


def fondo(img, gts, nivel, rng):
    h, w = img.shape[:2]
    return np.where(mascara_cartas(img, gts)[..., None], img, hacer_fondo(nivel, h, w, rng)), gts


CONDICIONES = (
    [("Original", "original", None, 0)] +
    [("Rotación", "rotación (°)", rotar, lv) for lv in [15, 30, 45, 90, 135, 180]] +
    [("Oclusión", "oclusión (fracción del índice)", ocluir, lv) for lv in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]] +
    [("Iluminación", "brillo (ganancia)", brillo, lv) for lv in [0.2, 0.4, 0.6, 1.5, 2.0, 2.5]] +
    [("Iluminación", "contraste (factor)", contraste, lv) for lv in [0.6, 0.4, 0.25, 0.15]] +
    [("Iluminación", "sombra (luz mínima)", sombra, lv) for lv in [0.6, 0.4, 0.2, 0.1]]
)
# El reemplazo de fondo con máscara de color (fondo()) NO se usa: separar carta de fondo por "papel blanco" falla
# tanto en las fotos reales (mesa clara, cartas en sombra) como en el sintético (fondos claros, figuras de color), y
# el experimento mediría los errores de la máscara. El fondo se evalúa con escenas compuestas en las que se sabe
# exactamente dónde está cada carta: Experimentos/robustez_fondo.py.


class Perturbacion:
    """Callable para evaluar_real.evaluar: misma imagen perturbada para todas las configuraciones."""

    def __init__(self, sub, fn, nivel):
        self.fn, self.nivel = fn, nivel
        self.nombre = f"{sub}={nivel}"

    def __call__(self, img, gts, ruta):
        if self.fn is None:
            return img, gts
        rng = np.random.default_rng([SEED, zlib.crc32(self.nombre.encode()), zlib.crc32(ruta.name.encode())])
        return self.fn(img, gts, self.nivel, rng)


def configuraciones(nombres):
    y13, y1, cnn = D.pesos_finetune("propio")
    todas = {"A": lambda: D.DetectorA(nombre="A"),
             "B": lambda: D.DetectorB(nombre="B"),
             "A_propio": lambda: D.DetectorA(pesos=y13, nombre="A_propio"),
             "B_propio": lambda: D.DetectorB(pesos_yolo=y1, pesos_cnn=cnn, nombre="B_propio")}
    return [(n, todas[n]) for n in nombres]


def ejemplos(muestras, salida):
    """Una imagen con cada familia de perturbación, para el informe."""
    ruta, gts = muestras[min(3, len(muestras) - 1)]
    img = cv2.imread(str(ruta))
    elegidas = [("original", None, 0), ("rotación 45°", rotar, 45), ("oclusión 40%", ocluir, 0.4),
                ("brillo x0.2", brillo, 0.2), ("contraste x0.25", contraste, 0.25), ("sombra 0.2", sombra, 0.2),
                ("sombra 0.6", sombra, 0.6), ("brillo x2.0", brillo, 2.0)]
    tiles = []
    for titulo, fn, nivel in elegidas:
        im, g = Perturbacion(titulo, fn, nivel)(img, gts, ruta)
        im = im.copy()
        for _, x1, y1, x2, y2 in g:
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        im = cv2.resize(im, (300, int(300 * im.shape[0] / im.shape[1])))
        (tw, th), _ = cv2.getTextSize(titulo, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(im, (0, 0), (tw + 12, th + 14), (0, 0, 0), -1)
        cv2.putText(im, titulo, (6, th + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    alto = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, alto - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    cv2.imwrite(str(salida), np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))


def graficar(df, salida, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subs = [s for s in df["perturbacion"].unique() if s != "original"]
    fig, axes = plt.subplots(1, len(subs), figsize=(3.6 * len(subs), 3.4), sharey=True)
    for ax, sub in zip(np.atleast_1d(axes), subs):
        d = df[df.perturbacion == sub]
        for cfg in df["config"].unique():
            dc = d[d.config == cfg]
            base = df[(df.config == cfg) & (df.perturbacion == "original")]["F1"].iloc[0]
            x = list(range(len(dc)))
            ax.plot(x, dc["F1"], marker="o", label=cfg)
            ax.axhline(base, ls=":", lw=0.8, color=ax.lines[-1].get_color())
            ax.set_xticks(x, [str(v) for v in dc["nivel"]], rotation=45, fontsize=7)
        ax.set_title(sub, fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("F1 (valor correcto)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle(titulo + "  (punteado: sin perturbar)", fontsize=10)
    fig.tight_layout()
    fig.savefig(salida, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="propio", choices=list(DATASETS))
    ap.add_argument("--n", type=int, default=0, help="usar sólo n imágenes (muestra fija); 0 = todas")
    ap.add_argument("--configs", nargs="+", default=["A", "B", "A_propio", "B_propio"])
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.5, help="IoU mínimo para el acierto (al rotar, la caja real se "
                    "agranda y con 0.5 se penalizan detecciones correctas: ver README)")
    ap.add_argument("--familias", nargs="+", help="sólo estas familias (p. ej. Rotación)")
    ap.add_argument("--salida", default=str(REPO / "Experimentos" / "resultados"))
    args = ap.parse_args()
    E.IOU_MIN = args.iou
    sufijo = "" if args.iou == 0.5 else f"_iou{int(round(args.iou * 100)):02d}"
    condiciones = [c for c in CONDICIONES if not args.familias or c[0] in args.familias or c[0] == "Original"]

    carpeta, split = DATASETS[args.dataset]
    muestras = E.cargar_split(carpeta, split)
    if args.n and len(muestras) > args.n:
        muestras = sorted(random.Random(SEED).sample(muestras, args.n), key=lambda m: m[0].name)
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    if not sufijo:
        ejemplos(muestras, salida / f"robustez_{args.dataset}_ejemplos.jpg")
    print(f"{len(muestras)} imágenes | {sum(len(g) for _, g in muestras)} esquinas | {len(condiciones)} condiciones "
          f"| configuraciones {args.configs}", flush=True)

    import pandas as pd
    filas = []
    for nombre, crear in configuraciones(args.configs):
        det = crear()
        t0 = time.time()
        for fam, sub, fn, nivel in condiciones:
            f, _, _ = E.evaluar(det, muestras, Perturbacion(sub, fn, nivel), args.conf)
            filas.append(dict(config=nombre, familia=fam, perturbacion=sub, nivel=nivel,
                              **{k: f[k] for k in ("mAP50", "P", "R", "F1", "R_loc", "ms_imagen")}))
            print(f"  {nombre:9s} {sub:32s} {str(nivel):9s} mAP50 {f['mAP50']:.3f} F1 {f['F1']:.3f} "
                  f"R_loc {f['R_loc']:.3f}", flush=True)
        print(f"  {nombre}: {(time.time() - t0) / 60:.1f} min", flush=True)
        df = pd.DataFrame(filas)
        df.to_csv(salida / f"robustez_{args.dataset}{sufijo}.csv", index=False, float_format="%.4f")

    df = pd.DataFrame(filas)
    graficar(df, salida / f"robustez_{args.dataset}{sufijo}.png",
             f"Robustez - test {args.dataset} ({len(muestras)} imágenes, IoU >= {args.iou})")
    resumen = df[df.perturbacion != "original"].groupby(["familia", "config"])["F1"].mean().unstack().round(3)
    print("\nF1 medio por familia de perturbación:\n", resumen)
    print("\nResultados en", salida)


if __name__ == "__main__":
    main()
