"""Evaluación de la clasificación de poses de la mano sobre imágenes reales de HaGRID.

HaGRID (Kapitanov et al., 2024; CC BY-SA 4.0) tiene fotos de personas haciendo gestos. Se usa una muestra
(hagrid-sample-30k-384p en Hugging Face) y se mapea cada clase a la pose esperada:
    one               -> APUNTAR   (gesto PEDIR)
    palm, stop, four  -> PALMA     (gesto PLANTARSE)
    el resto          -> OTRA      (no debe disparar nada: fist, like, dislike, peace, call, three)

Se comparan dos clasificadores sobre los mismos puntos de MediaPipe:
    reglas  -> gestos.clasificar_pose (cocientes de distancias; invariante a la rotación)
    mp      -> MediaPipe GestureRecognizer pre-entrenado (Pointing_Up -> APUNTAR, Open_Palm -> PALMA)
en la imagen original y rotada 90° y 180° (con cámara cenital la mano aparece en cualquier orientación).

Uso (desde la carpeta del repo):
    python Gestos/evaluar_gestos.py            # baja 40 imágenes por clase la primera vez
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from gestos import MODELOS, DetectorDeManos, clasificar_pose   # noqa: E402

DATASET = "cj-mills/hagrid-sample-30k-384p"
ETIQUETAS = {"call": 0, "dislike": 1, "fist": 2, "four": 3, "like": 4, "one": 7, "palm": 8, "peace": 9,
             "stop": 12, "three": 14}
ESPERADA = {"one": "APUNTAR", "palm": "PALMA", "stop": "PALMA", "four": "PALMA"}
MP_A_POSE = {"Pointing_Up": "APUNTAR", "Open_Palm": "PALMA"}
URL_RECONOCEDOR = ("https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/"
                   "latest/gesture_recognizer.task")


def bajar_hagrid(destino, n):
    """Baja n imágenes por clase con la API de filas de Hugging Face (el dataset está ordenado por clase)."""
    for nombre, label in ETIQUETAS.items():
        d = destino / nombre
        if d.exists() and len(list(d.glob("*.jpg"))) >= n:
            continue
        d.mkdir(parents=True, exist_ok=True)
        q = urllib.parse.urlencode({"dataset": DATASET, "config": "default", "split": "train",
                                    "offset": label * 1768 + 400, "length": n})
        with urllib.request.urlopen(f"https://datasets-server.huggingface.co/rows?{q}", timeout=120) as r:
            filas = json.load(r)["rows"]
        for fila in filas:
            if fila["row"]["label"] == label:
                with urllib.request.urlopen(fila["row"]["image"]["src"], timeout=60) as r:
                    (d / f"{fila['row_idx']}.jpg").write_bytes(r.read())
        print(f"  {nombre}: {len(list(d.glob('*.jpg')))} imágenes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default=str(AQUI / "datos_hagrid"))
    ap.add_argument("--n", type=int, default=40, help="imágenes por clase")
    ap.add_argument("--salida", default=str(AQUI.parent / "Experimentos" / "resultados" / "gestos_hagrid.csv"))
    args = ap.parse_args()

    datos = Path(args.datos)
    bajar_hagrid(datos, args.n)

    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    if not (MODELOS / "gesture_recognizer.task").exists():
        urllib.request.urlretrieve(URL_RECONOCEDOR, MODELOS / "gesture_recognizer.task")
    reconocedor = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELOS / "gesture_recognizer.task")), num_hands=1))
    manos = DetectorDeManos(num_manos=1, modo="imagen")

    rotaciones = {"original": None, "rot90": cv2.ROTATE_90_CLOCKWISE, "rot180": cv2.ROTATE_180}
    conteo = defaultdict(Counter)          # (metodo, rotacion, clase) -> Counter(pose predicha)
    for clase in ETIQUETAS:
        for f in sorted((datos / clase).glob("*.jpg"))[: args.n]:
            img0 = cv2.imread(str(f))
            for rot, codigo in rotaciones.items():
                img = img0 if codigo is None else cv2.rotate(img0, codigo)
                puntos = manos.detectar(img)
                conteo[("reglas", rot, clase)][clasificar_pose(puntos[0]) if puntos else "SIN_MANO"] += 1
                r = reconocedor.recognize(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                   data=np.ascontiguousarray(img[:, :, ::-1])))
                pred = MP_A_POSE.get(r.gestures[0][0].category_name, "OTRA") if r.gestures else "SIN_MANO"
                conteo[("mp", rot, clase)][pred] += 1

    filas = []
    for metodo in ("reglas", "mp"):
        for rot in rotaciones:
            fila = dict(metodo=metodo, rotacion=rot)
            for pose in ("APUNTAR", "PALMA"):
                positivas = [c for c, p in ESPERADA.items() if p == pose]
                tp = sum(conteo[(metodo, rot, c)][pose] for c in positivas)
                n_pos = sum(sum(conteo[(metodo, rot, c)].values()) for c in positivas)
                fp = sum(conteo[(metodo, rot, c)][pose] for c in ETIQUETAS if c not in positivas)
                fila[f"recall_{pose}"] = tp / max(n_pos, 1)
                fila[f"precision_{pose}"] = tp / max(tp + fp, 1)
            negativas = [c for c in ETIQUETAS if c not in ESPERADA]
            n_neg = sum(sum(conteo[(metodo, rot, c)].values()) for c in negativas)
            disparos = sum(conteo[(metodo, rot, c)][p] for c in negativas for p in ("APUNTAR", "PALMA"))
            fila["falsos_disparos"] = disparos / max(n_neg, 1)
            fila["sin_mano"] = sum(conteo[(metodo, rot, c)]["SIN_MANO"] for c in ETIQUETAS) / \
                sum(sum(conteo[(metodo, rot, c)].values()) for c in ETIQUETAS)
            filas.append(fila)

    import pandas as pd
    df = pd.DataFrame(filas)
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.salida, index=False, float_format="%.3f")
    print(df.round(3).to_string(index=False))
    print("\nDetalle (reglas, original):")
    for clase in ETIQUETAS:
        print(f"  {clase:8s} -> {dict(conteo[('reglas', 'original', clase)])}")
    print("\nResultados en", args.salida)


if __name__ == "__main__":
    main()
