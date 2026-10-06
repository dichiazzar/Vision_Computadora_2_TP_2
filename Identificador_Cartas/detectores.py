"""Los dos pipelines de detección de esquinas, con la misma interfaz: detectar(frame_bgr) -> [Corner].

Pipeline A: YOLO-13 detecta la esquina y clasifica su valor en un solo paso.
Pipeline B: YOLO-1 sólo localiza la esquina -> se recorta -> la CNN-13 (ResNet-18) clasifica el valor.

Con seguir=True el detector corre además ByteTrack (incluido en ultralytics): cada esquina recibe un
track_id que se mantiene entre cuadros (por posición y movimiento: filtro de Kalman + IoU).
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from class_corner import Corner

REPO = Path(__file__).resolve().parent.parent

PESOS_YOLO_13 = REPO / "YOLO - 13" / "runs" / "detect" / "blackjack-cv" / "yolo13-baseline-3" / "weights" / "best.pt"
PESOS_YOLO_1 = REPO / "YOLO - 1" / "runs" / "detect" / "blackjack-cv" / "yolo1-baseline" / "weights" / "best.pt"
PESOS_CNN_13 = REPO / "CNN - 13" / "runs" / "cnn13-resnet18" / "best.pt"

# Pesos después del fine-tuning con fotos reales (Experimentos/finetune_real.py). Cada fine-tuning tiene una etiqueta:
#   "real"   -> con el dataset público teogopk (otro mazo)
#   "propio" -> con fotos del mazo y la mesa de la demo (Dataset Real/roboflow)
ETIQUETAS_FT = {"real": "teogopk", "propio": "fotos propias"}


def pesos_finetune(etiqueta):
    """(YOLO-13, YOLO-1, CNN-13) ajustados con el dataset de esa etiqueta."""
    return (REPO / "YOLO - 13" / "runs" / "detect" / "blackjack-cv" / f"yolo13-finetune-{etiqueta}" / "weights" / "best.pt",
            REPO / "YOLO - 1" / "runs" / "detect" / "blackjack-cv" / f"yolo1-finetune-{etiqueta}" / "weights" / "best.pt",
            REPO / "CNN - 13" / "runs" / f"cnn13-finetune-{etiqueta}" / "best.pt")


PESOS_YOLO_13_FT, PESOS_YOLO_1_FT, PESOS_CNN_13_FT = pesos_finetune("real")

# Mismo margen que create_cnn_dataset.py usó para generar los recortes de entrenamiento
PADDING_RECORTE = 0.10
MAX_DET = 50          # máximo de esquinas por imagen (una mesa tiene muchas menos)
TRACKER = "bytetrack.yaml"


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DetectorA:
    nombre = "A: YOLO-13"

    def __init__(self, conf=0.5, pesos=PESOS_YOLO_13, nombre=None):
        self.modelo = YOLO(str(pesos))
        self.nombre = nombre or self.nombre
        self.conf = conf
        self.device = _device()

    def reiniciar_seguimiento(self):
        self.modelo.predictor = None          # descarta el estado de ByteTrack

    def detectar(self, frame, seguir=False):
        # agnostic_nms: si dos cajas casi coinciden, queda sólo la más confiable AUNQUE tengan valores distintos.
        # Por defecto YOLO sólo suprime cajas del mismo valor, y una esquina dudosa (un 3 invertido y medio tapado)
        # salía dos veces, como "3" y como "5": la "5" contaba como una carta de más.
        kw = dict(conf=self.conf, max_det=MAX_DET, device=self.device, agnostic_nms=True, verbose=False)
        r = (self.modelo.track(frame, persist=True, tracker=TRACKER, **kw) if seguir
             else self.modelo.predict(frame, **kw))[0]
        b = r.boxes
        ids = b.id.int().cpu().tolist() if b.id is not None else [None] * len(b)
        return [Corner(box, self.modelo.names[int(c)], float(s), track_id=tid)
                for box, c, s, tid in zip(b.xyxy.cpu().numpy().tolist(), b.cls.cpu().tolist(),
                                          b.conf.cpu().tolist(), ids)]


class DetectorB:
    nombre = "B: YOLO-1 + CNN-13"

    def __init__(self, conf=0.5, pesos_yolo=PESOS_YOLO_1, pesos_cnn=PESOS_CNN_13, nombre=None):
        from torchvision import models

        if not Path(pesos_cnn).exists():
            raise FileNotFoundError(f"No están los pesos de la CNN-13 ({pesos_cnn}). "
                                    "Entrenarla con: python \"CNN - 13/train_cnn_13.py\"")
        self.yolo = YOLO(str(pesos_yolo))
        self.nombre = nombre or self.nombre
        self.conf = conf
        self.device = _device()

        ckpt = torch.load(pesos_cnn, map_location="cpu", weights_only=False)  # checkpoint propio (train_cnn_13.py)
        self.clases = ckpt["classes"]
        self.img_size = ckpt["img_size"]
        self.mean = np.array(ckpt["mean"], np.float32)
        self.std = np.array(ckpt["std"], np.float32)
        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = torch.nn.Linear(self.cnn.fc.in_features, len(self.clases))
        self.cnn.load_state_dict(ckpt["state_dict"])
        self.cnn.to(self.device).eval()

    def _recortar(self, frame, box):
        """Recorte con el mismo padding y preprocesamiento que en el entrenamiento de la CNN."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        px, py = (x2 - x1) * PADDING_RECORTE, (y2 - y1) * PADDING_RECORTE
        x1, y1 = max(0, int(round(x1 - px))), max(0, int(round(y1 - py)))
        x2, y2 = min(w, int(round(x2 + px))), min(h, int(round(y2 + py)))
        if x2 <= x1 or y2 <= y1:      # caja degenerada pegada al borde (confianza muy baja): recorte vacío
            return np.zeros((3, self.img_size, self.img_size), np.float32)
        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        crop = (crop.astype(np.float32) / 255.0 - self.mean) / self.std
        return crop.transpose(2, 0, 1)

    def reiniciar_seguimiento(self):
        self.yolo.predictor = None

    @torch.no_grad()
    def detectar(self, frame, seguir=False):
        kw = dict(conf=self.conf, max_det=MAX_DET, device=self.device, verbose=False)
        r = (self.yolo.track(frame, persist=True, tracker=TRACKER, **kw) if seguir
             else self.yolo.predict(frame, **kw))[0]
        cajas = r.boxes.xyxy.cpu().numpy()
        if len(cajas) == 0:
            return []
        conf_det = r.boxes.conf.cpu().numpy()
        ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else [None] * len(cajas)
        x = torch.from_numpy(np.stack([self._recortar(frame, b) for b in cajas])).to(self.device)
        prob = torch.softmax(self.cnn(x), 1).cpu().numpy()
        esquinas = []
        for box, cd, p, tid in zip(cajas, conf_det, prob, ids):
            k = int(p.argmax())
            # confianza conjunta: que sea una esquina (YOLO-1) y que sea ese valor (CNN)
            esquinas.append(Corner(box.tolist(), self.clases[k], float(cd * p[k]), track_id=tid))
        return esquinas


def crear_detector(pipeline, conf=0.5, pesos="auto"):
    """pesos: "base" (entrenados con el dataset sintético), la etiqueta de un fine-tuning ("real", "propio") o
    "auto": los ajustados con fotos propias si existen (en el escenario de la demo el recall de YOLO-13 pasa de 0.66
    a 0.96), y si no los base."""
    if pesos == "auto":
        y13, y1, cnn = pesos_finetune("propio")
        listos = y13.exists() if pipeline.upper() == "A" else (y1.exists() and cnn.exists())
        pesos = "propio" if listos else "base"
    if pesos == "base":
        return DetectorA(conf=conf) if pipeline.upper() == "A" else DetectorB(conf=conf)
    y13, y1, cnn = pesos_finetune(pesos)
    if pipeline.upper() == "A":
        if not y13.exists():
            raise FileNotFoundError(f"No hay fine-tuning '{pesos}' de YOLO-13 ({y13})")
        return DetectorA(conf=conf, pesos=y13, nombre=f"A: YOLO-13 ft-{pesos}")
    if not (y1.exists() and cnn.exists()):
        raise FileNotFoundError(f"No hay fine-tuning '{pesos}' de YOLO-1 / CNN-13 ({y1}, {cnn})")
    return DetectorB(conf=conf, pesos_yolo=y1, pesos_cnn=cnn, nombre=f"B: YOLO-1 + CNN-13 ft-{pesos}")
