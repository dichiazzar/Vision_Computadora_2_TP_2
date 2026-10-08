"""Análisis de confusiones entre valores (sugerencia del profesor: "qué cartas generan mayor confusión entre sí").

Para cada configuración, cada esquina real se cruza con la detección que más se le superpone (IoU >= 0.5, sin mirar
el valor, conf >= 0.5). Así se separan tres tipos de error:
  - confusión de valor: la esquina se encontró, pero con otro valor (p. ej. un 3 leído como 5);
  - esquina no encontrada (falso negativo de localización);
  - detección que no corresponde a ninguna esquina real (falso positivo).
Se guarda la matriz de confusión (real x predicho, con la columna "no detectada"), las confusiones más frecuentes y
las esquinas no detectadas por valor (filas con predicho = "no detectada"; las filas con real = "TOTAL" son los totales).

Uso (desde la carpeta del repo):
    python Experimentos/confusiones.py --dataset propio
Resultados: Experimentos/resultados/confusiones_<dataset>.csv y .png
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Experimentos"))
sys.path.insert(0, str(REPO / "Identificador_Cartas"))

import evaluar_real as E                        # noqa: E402
from clases import CLASES                       # noqa: E402
from robustez import DATASETS, configuraciones  # noqa: E402

ORDEN = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]


def cruzar(preds, gts):
    """Asignación voraz por IoU (sin mirar el valor). Devuelve pares (gt_valor | None, pred_valor | None)."""
    pares, usados_p = [], set()
    candidatos = []
    for i, (r, *g) in enumerate(gts):
        for j, p in enumerate(preds):
            v = E.iou((*p.sup_izq, *p.inf_der), g)
            if v >= 0.5:
                candidatos.append((v, i, j))
    usados_g = set()
    for v, i, j in sorted(candidatos, reverse=True):
        if i in usados_g or j in usados_p:
            continue
        usados_g.add(i)
        usados_p.add(j)
        pares.append((gts[i][0], preds[j].rank))
    pares += [(gts[i][0], None) for i in range(len(gts)) if i not in usados_g]
    pares += [(None, preds[j].rank) for j in range(len(preds)) if j not in usados_p]
    return pares


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="propio", choices=list(DATASETS))
    ap.add_argument("--configs", nargs="+", default=["A", "B", "A_propio", "B_propio"])
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--salida", default=str(REPO / "Experimentos" / "resultados"))
    args = ap.parse_args()

    carpeta, split = DATASETS[args.dataset]
    muestras = E.cargar_split(carpeta, split)
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filas, matrices = [], {}
    for nombre, crear in configuraciones(args.configs):
        det = crear()
        det.conf = args.conf
        pares = []
        for ruta, gts in muestras:
            pares += cruzar(det.detectar(cv2.imread(str(ruta))), gts)
        cont = Counter(pares)
        n_gt = sum(v for (g, _), v in cont.items() if g is not None)
        bien = sum(v for (g, p), v in cont.items() if g is not None and g == p)
        valor_mal = sum(v for (g, p), v in cont.items() if g is not None and p is not None and g != p)
        no_det = sum(v for (g, p), v in cont.items() if g is not None and p is None)
        fp = sum(v for (g, p), v in cont.items() if g is None)
        print(f"{nombre:9s} esquinas reales {n_gt}: bien {bien} ({bien / n_gt:.1%}) | valor equivocado {valor_mal} "
              f"({valor_mal / n_gt:.1%}) | no detectadas {no_det} ({no_det / n_gt:.1%}) | falsos positivos {fp}")
        confusiones = sorted(((v, g, p) for (g, p), v in cont.items() if g and p and g != p), reverse=True)
        print("   confusiones más frecuentes (real -> predicho):",
              ", ".join(f"{g}->{p} x{v}" for v, g, p in confusiones[:8]) or "ninguna")
        no_det_por_valor = Counter({g: v for (g, p), v in cont.items() if g and p is None})
        print("   no detectadas por valor:", dict(no_det_por_valor.most_common()))
        m = pd.DataFrame(0, index=ORDEN, columns=ORDEN + ["no det."])
        for (g, p), v in cont.items():
            if g is not None:
                m.loc[g, p if p is not None else "no det."] += v
        matrices[nombre] = m
        for v, g, p in confusiones:
            filas.append(dict(config=nombre, real=g, predicho=p, veces=v))
        for g, v in no_det_por_valor.most_common():
            filas.append(dict(config=nombre, real=g, predicho="no detectada", veces=v))
        filas.append(dict(config=nombre, real="TOTAL", predicho="bien", veces=bien))
        filas.append(dict(config=nombre, real="TOTAL", predicho="valor equivocado", veces=valor_mal))
        filas.append(dict(config=nombre, real="TOTAL", predicho="no detectada", veces=no_det))
        filas.append(dict(config=nombre, real="TOTAL", predicho="falso positivo", veces=fp))

    salida = Path(args.salida)
    pd.DataFrame(filas).to_csv(salida / f"confusiones_{args.dataset}.csv", index=False)
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.6 * len(matrices), 4.4))
    for ax, (nombre, m) in zip(np.atleast_1d(axes), matrices.items()):
        normal = m.div(m.sum(1).replace(0, 1), axis=0)
        ax.imshow(normal.values, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(m.columns)), m.columns, fontsize=7, rotation=90)
        ax.set_yticks(range(len(m.index)), m.index, fontsize=7)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if m.iat[i, j]:
                    ax.text(j, i, m.iat[i, j], ha="center", va="center", fontsize=6,
                            color="white" if normal.iat[i, j] > 0.5 else "black")
        ax.set_title(nombre, fontsize=10)
        ax.set_xlabel("predicho")
    np.atleast_1d(axes)[0].set_ylabel("real")
    fig.suptitle(f"Confusiones entre valores - test {args.dataset} ({len(muestras)} imágenes)", fontsize=10)
    fig.tight_layout()
    fig.savefig(salida / f"confusiones_{args.dataset}.png", dpi=130)
    print("\nResultados en", salida)


if __name__ == "__main__":
    main()
