"""Evaluación de los pipelines A y B sobre el dataset de fotos reales (por defecto Dataset Real/teogopk; con --dataset/--real, otro export YOLO, p. ej. fotos propias).

Métricas, para cada configuración y cada variante de la imagen (original, rotada 90°/180°, al 50%):

  Nivel esquina (la caja tiene que coincidir con IoU >= 0.5 Y tener el valor correcto):
    - mAP50       promedio sobre los 13 valores (umbral de confianza bajo, curva PR completa)
    - P / R / F1  en el punto de operación (conf >= --conf, el que usa la aplicación)
    - R_loc       recall de localización ignorando el valor (¿encontró la esquina?)
  Nivel mano (lo que importa para el Blackjack):
    - cartas_ok   % de imágenes donde las cartas de la Casa y del Jugador son exactamente las reales
    - puntaje_ok  % de imágenes donde los dos puntajes de Blackjack son los reales
    Las cartas "reales" se arman con las etiquetas aplicando el mismo emparejamiento de esquinas.
  Latencia media por imagen (ms) en el punto de operación.

Configuraciones: A y B con los pesos base, y A_ft / B_ft con los del fine-tuning si existen.

Uso (desde la carpeta del repo):
    python Experimentos/evaluar_real.py
    python Experimentos/evaluar_real.py --split valid --sin-variantes
Resultados en Experimentos/resultados/.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Identificador_Cartas"))

from class_carta import emparejar_esquinas          # noqa: E402
from class_corner import Corner                     # noqa: E402
from class_player import Entidad                    # noqa: E402
import detectores as D                              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clases import normalizar_clase                 # noqa: E402

IOU_MIN = 0.5
CONF_AP = 0.001          # umbral bajo para calcular mAP


# ============================================================
# DATOS
# ============================================================

def cargar_split(dataset, split):
    """Lista de (ruta_imagen, [(rank, x1, y1, x2, y2), ...]) en píxeles."""
    nombres = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))["names"]
    if isinstance(nombres, dict):
        nombres = [nombres[k] for k in sorted(nombres, key=int)]
    nombres = [normalizar_clase(n) for n in nombres]     # '10h' / 'AS' / ... -> valor
    muestras = []
    for img_path in sorted((dataset / split / "images").glob("*")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl = dataset / split / "labels" / f"{img_path.stem}.txt"
        gts = []
        if lbl.exists():
            for linea in lbl.read_text().splitlines():
                p = linea.split()
                if len(p) < 5:
                    continue
                c, xc, yc, bw, bh = int(float(p[0])), *map(float, p[1:5])
                gts.append((nombres[c], (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h))
        muestras.append((img_path, gts))
    return muestras


def transformar(img, gts, variante):
    """Aplica la variante a la imagen y a las cajas reales."""
    h, w = img.shape[:2]
    if variante == "original":
        return img, gts
    if variante == "rot90":      # horario: (x, y) -> (h - y, x)
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), [(r, h - y2, x1, h - y1, x2) for r, x1, y1, x2, y2 in gts]
    if variante == "rot180":
        return cv2.rotate(img, cv2.ROTATE_180), [(r, w - x2, h - y2, w - x1, h - y1) for r, x1, y1, x2, y2 in gts]
    if variante == "escala50":
        return cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), \
            [(r, x1 / 2, y1 / 2, x2 / 2, y2 / 2) for r, x1, y1, x2, y2 in gts]
    raise ValueError(variante)


# ============================================================
# MÉTRICAS
# ============================================================

def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def emparejar(preds, gts, con_valor):
    """Asignación voraz por confianza. Devuelve una lista de booleanos (TP/FP) alineada con preds ordenadas."""
    usados, tps = set(), []
    for p in sorted(preds, key=lambda e: -e.conf):
        caja = (*p.sup_izq, *p.inf_der)
        mejor, mejor_iou = None, IOU_MIN
        for k, (r, *g) in enumerate(gts):
            if k in usados or (con_valor and r != p.rank):
                continue
            v = iou(caja, g)
            if v >= mejor_iou:
                mejor, mejor_iou = k, v
        if mejor is not None:
            usados.add(mejor)
        tps.append(mejor is not None)
    return tps


def average_precision(tps_conf, n_gt):
    """AP de todos los puntos (estilo COCO/VOC) a partir de [(conf, es_tp)]."""
    if n_gt == 0:
        return None
    if not tps_conf:
        return 0.0
    tps_conf.sort(key=lambda t: -t[0])
    tp = np.cumsum([t for _, t in tps_conf])
    fp = np.cumsum([not t for _, t in tps_conf])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def manos(esquinas, alto):
    cartas, _ = emparejar_esquinas(esquinas)
    casa, jugador = Entidad("Casa"), Entidad("Jugador")
    for c in cartas:
        (casa if c.centro[1] < alto / 2 else jugador).agregar_carta(c)
    return casa, jugador


def misma_mano(a, b):
    return Counter(c.rank for c in a.cartas) == Counter(c.rank for c in b.cartas)


# ============================================================
# EVALUACIÓN
# ============================================================

def evaluar(detector, muestras, variante, conf_op):
    por_clase = {}                    # rank -> [(conf, tp)]
    n_gt = Counter()
    tp_op = fp_op = fn_op = loc_tp = 0
    cartas_ok = puntaje_ok = 0
    tiempos, detalle = [], []

    for img_path, gts0 in muestras:
        if callable(variante):        # perturbación arbitraria (Experimentos/robustez.py): (img, gts, ruta) -> (img, gts)
            img, gts = variante(cv2.imread(str(img_path)), gts0, img_path)
        else:
            img, gts = transformar(cv2.imread(str(img_path)), gts0, variante)
        alto = img.shape[0]

        detector.conf = CONF_AP
        todas = detector.detectar(img)
        detector.conf = conf_op
        t0 = time.perf_counter()
        ops = detector.detectar(img)              # punto de operación (se cronometra esta llamada)
        tiempos.append((time.perf_counter() - t0) * 1000)

        # mAP50 por valor
        for r in {g[0] for g in gts} | {p.rank for p in todas}:
            preds_r = [p for p in todas if p.rank == r]
            gts_r = [g for g in gts if g[0] == r]
            tps = emparejar(preds_r, gts_r, con_valor=True)
            por_clase.setdefault(r, []).extend(zip(sorted((p.conf for p in preds_r), reverse=True), tps))
            n_gt[r] += len(gts_r)

        # P / R / F1 en el punto de operación
        tps = emparejar(ops, gts, con_valor=True)
        tp_op += sum(tps)
        fp_op += len(tps) - sum(tps)
        fn_op += len(gts) - sum(tps)
        loc_tp += sum(emparejar(ops, gts, con_valor=False))

        # Manos
        gt_corners = [Corner([x1, y1, x2, y2], r) for r, x1, y1, x2, y2 in gts]
        casa_gt, jug_gt = manos(gt_corners, alto)
        casa_pr, jug_pr = manos(ops, alto)
        ok_c = misma_mano(casa_gt, casa_pr) and misma_mano(jug_gt, jug_pr)
        ok_p = casa_gt.puntaje == casa_pr.puntaje and jug_gt.puntaje == jug_pr.puntaje
        cartas_ok += ok_c
        puntaje_ok += ok_p
        detalle.append(dict(imagen=img_path.name, variante=getattr(variante, "nombre", variante), gt=f"{casa_gt} | {jug_gt}",
                            pred=f"{casa_pr} | {jug_pr}", cartas_ok=bool(ok_c), puntaje_ok=bool(ok_p)))

    aps = {r: average_precision(v, n_gt[r]) for r, v in por_clase.items()}
    aps = {r: a for r, a in aps.items() if a is not None}
    n_total = sum(n_gt.values())
    p = tp_op / max(tp_op + fp_op, 1)
    rr = tp_op / max(n_total, 1)
    n = len(muestras)
    fila = dict(mAP50=np.mean(list(aps.values())) if aps else 0.0, P=p, R=rr,
                F1=2 * p * rr / max(p + rr, 1e-9), R_loc=loc_tp / max(n_total, 1),
                cartas_ok=cartas_ok / n, puntaje_ok=puntaje_ok / n,
                ms_imagen=float(np.mean(tiempos)), imagenes=n, esquinas=n_total)
    return fila, {r: round(a, 4) for r, a in sorted(aps.items())}, detalle


def configuraciones():
    """A y B base, más los fine-tunings que existan: A_ft / B_ft (teogopk) y A_propio / B_propio (fotos propias)."""
    confs = [("A", lambda: D.DetectorA(nombre="A")),
             ("B", lambda: D.DetectorB(nombre="B"))]
    for etiqueta, sufijo in (("real", "ft"), ("propio", "propio")):
        y13, y1, cnn = D.pesos_finetune(etiqueta)
        if y13.exists():
            confs.append((f"A_{sufijo}", lambda y13=y13, s=sufijo: D.DetectorA(pesos=y13, nombre=f"A_{s}")))
        if y1.exists() and cnn.exists():
            confs.append((f"B_{sufijo}", lambda y1=y1, cnn=cnn, s=sufijo:
                          D.DetectorB(pesos_yolo=y1, pesos_cnn=cnn, nombre=f"B_{s}")))
    return confs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="propio",
                    help="propio (Dataset Real/roboflow), teogopk (Dataset Real/teogopk) u otra carpeta")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.5, help="umbral del punto de operación")
    ap.add_argument("--sin-variantes", action="store_true", help="sólo las imágenes originales")
    ap.add_argument("--salida", default=str(REPO / "Experimentos" / "resultados"))
    args = ap.parse_args()

    nombre_dataset = args.dataset if args.dataset in ("propio", "teogopk") else Path(args.dataset).name
    dataset = {"propio": REPO / "Dataset Real" / "roboflow",
               "teogopk": REPO / "Dataset Real" / "teogopk"}.get(args.dataset, Path(args.dataset))
    salida = Path(args.salida)
    print("dataset:", dataset)
    if not (dataset / "data.yaml").exists():
        sys.exit(f"No encontré {dataset / 'data.yaml'}: exportar el dataset real de Roboflow ahí (ver Dataset Real/README.md)")
    muestras = cargar_split(dataset, args.split)
    if not muestras:
        sys.exit(f"No hay imágenes en {dataset / args.split / 'images'}")
    variantes = ["original"] if args.sin_variantes else ["original", "rot90", "rot180", "escala50"]
    print(f"{len(muestras)} imágenes de {args.split} | {sum(len(g) for _, g in muestras)} esquinas | variantes: {variantes}")

    filas, aps_todas, detalles = [], {}, []
    for nombre, crear in configuraciones():
        det = crear()
        for var in variantes:
            fila, aps, det_img = evaluar(det, muestras, var, args.conf)
            filas.append(dict(config=nombre, variante=var, **fila))
            aps_todas[f"{nombre}/{var}"] = aps
            detalles += [dict(config=nombre, **d) for d in det_img]
            print(f"  {nombre:5s} {var:9s} mAP50 {fila['mAP50']:.3f} | P {fila['P']:.3f} R {fila['R']:.3f} "
                  f"F1 {fila['F1']:.3f} | R_loc {fila['R_loc']:.3f} | cartas_ok {fila['cartas_ok']:.2f} "
                  f"puntaje_ok {fila['puntaje_ok']:.2f} | {fila['ms_imagen']:.0f} ms", flush=True)

    import pandas as pd
    salida.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(filas)
    df.to_csv(salida / f"eval_{nombre_dataset}_{args.split}.csv", index=False, float_format="%.4f")
    (salida / f"eval_{nombre_dataset}_{args.split}_ap_por_valor.json").write_text(json.dumps(aps_todas, indent=2))
    pd.DataFrame(detalles).to_csv(salida / f"eval_{nombre_dataset}_{args.split}_por_imagen.csv", index=False)

    print("\nResumen (promedio de las variantes):")
    print(df.groupby("config")[["mAP50", "F1", "R_loc", "cartas_ok", "puntaje_ok", "ms_imagen"]].mean().round(3))
    print("\nResultados en", salida)


if __name__ == "__main__":
    main()
