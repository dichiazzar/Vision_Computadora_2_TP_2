"""Reconocimiento de gestos de la mano del Jugador para controlar la partida.

Dos gestos (los que se usan en una mesa de Blackjack):
    PEDIR       "dame una carta"   -> pose APUNTAR: sólo el índice extendido (como señalar / golpear la mesa)
    PLANTARSE   "no quiero más"    -> pose PALMA (mano abierta, los cuatro dedos extendidos) EN MOVIMIENTO:
                                      pasar la mano sobre las cartas, como en una mesa real

Cómo funciona:
1. MediaPipe HandLandmarker (pre-entrenado, corre en CPU en tiempo real) da 21 puntos por mano.
2. La pose se clasifica con reglas sobre esos puntos: un dedo está extendido si la distancia muñeca->punta es
   bastante mayor que muñeca->nudillo. Son cocientes de distancias, así que NO dependen de la orientación de la
   mano ni de su tamaño en la imagen: funcionan igual con la cámara cenital, con la mano girada, cerca o lejos.
   Umbrales calibrados con HaGRID (ver evaluar_gestos.py): extendido > 1.45, doblado < 1.25.
3. Lógica temporal: APUNTAR tiene que sostenerse `cuadros_min` cuadros seguidos; PALMA tiene que desplazarse
   al menos un tamaño de palma (rápido) o sostenerse `cuadros_palma` cuadros (~1.2 s). Una mano abierta apoyada
   un instante no alcanza (en HaGRID la mayoría de los falsos "PALMA" eran la otra mano, relajada). Después de un
   gesto hay un tiempo muerto de `espera` cuadros.
   Sólo se aceptan manos en la zona del Jugador (mitad inferior de la imagen).
"""

import math
from pathlib import Path

import numpy as np

MODELOS = Path(__file__).resolve().parent / "modelos"
MODELO_MANOS = MODELOS / "hand_landmarker.task"
URL_MODELO = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/"
              "hand_landmarker.task")

EXTENDIDO, DOBLADO = 1.45, 1.25
DEDOS = [(8, 5), (12, 9), (16, 13), (20, 17)]      # (punta, nudillo) de índice, medio, anular y meñique
GESTO_DE_POSE = {"APUNTAR": "PEDIR", "PALMA": "PLANTARSE"}


def extension_dedos(puntos):
    """puntos: array (21, 2). Devuelve dist(muñeca, punta) / dist(muñeca, nudillo) para los 4 dedos."""
    m = puntos[0]
    return [math.dist(m, puntos[t]) / max(math.dist(m, puntos[n]), 1e-6) for t, n in DEDOS]


def clasificar_pose(puntos):
    """APUNTAR (sólo índice extendido), PALMA (los 4 dedos extendidos) u OTRA. El pulgar no se usa."""
    indice, medio, anular, menique = extension_dedos(puntos)
    if indice > EXTENDIDO and max(medio, anular, menique) < DOBLADO:
        return "APUNTAR"
    if min(indice, medio, anular, menique) > EXTENDIDO:
        return "PALMA"
    return "OTRA"


def descargar_modelo():
    if not MODELO_MANOS.exists():
        import urllib.request
        MODELOS.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL_MODELO, MODELO_MANOS)
    return MODELO_MANOS


class DetectorDeManos:
    """MediaPipe HandLandmarker. modo="video" para cámara/video (usa la continuidad entre cuadros)."""

    def __init__(self, num_manos=2, modo="video", conf=0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self.modo = modo
        opciones = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(descargar_modelo())),
            running_mode=vision.RunningMode.VIDEO if modo == "video" else vision.RunningMode.IMAGE,
            num_hands=num_manos, min_hand_detection_confidence=conf, min_hand_presence_confidence=conf)
        self.detector = vision.HandLandmarker.create_from_options(opciones)
        self._t_ms = 0

    def detectar(self, frame_bgr):
        """Devuelve una lista de arrays (21, 2) en píxeles, uno por mano."""
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        imagen = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self.modo == "video":
            self._t_ms += 66                       # timestamps crecientes (~15 FPS); MediaPipe sólo exige orden
            r = self.detector.detect_for_video(imagen, self._t_ms)
        else:
            r = self.detector.detect(imagen)
        return [np.array([[p.x * w, p.y * h] for p in mano]) for mano in r.hand_landmarks]


def tamanio_palma(puntos):
    """Distancia muñeca -> nudillo del dedo medio: escala de la mano en la imagen."""
    return max(math.dist(puntos[0], puntos[9]), 1e-6)


def centro_palma(puntos):
    return puntos[[0, 5, 9, 13, 17]].mean(0)


class ReconocedorDeGestos:
    """Convierte las manos detectadas cuadro a cuadro en gestos discretos ("PEDIR" / "PLANTARSE").

    PEDIR      pose APUNTAR sostenida `cuadros_min` cuadros seguidos.
    PLANTARSE  pose PALMA, de dos formas (plantarse="ambos", por defecto):
               - barrido: la palma abierta se DESPLAZA al menos `barrido` tamaños de palma (pasar la mano sobre
                 las cartas, como en una mesa real). Es la forma rápida.
               - sostenida: la palma abierta se mantiene `cuadros_palma` cuadros seguidos (~1.2 s a 10 FPS).
               Con plantarse="barrido" sólo vale el barrido (una mano abierta apoyada nunca dispara) y con
               plantarse="estatico" sólo la palma sostenida.
    Después de cada gesto hay `espera` cuadros sin aceptar otro, y además hay que SOLTAR la pose (retirar la mano
    o cambiarla) antes de que pueda dispararse otro: mantener la palma abierta no se planta una y otra vez.
    """

    def __init__(self, detector=None, cuadros_min=6, espera=20, zona_jugador=True, plantarse="ambos",
                 barrido=1.0, cuadros_palma=12, tolerancia=2):
        assert plantarse in ("ambos", "barrido", "estatico"), plantarse
        self.cuadros_palma = cuadros_palma
        self.detector = detector
        self.cuadros_min, self.espera_total, self.zona_jugador = cuadros_min, espera, zona_jugador
        self.plantarse, self.barrido, self.tolerancia = plantarse, barrido, tolerancia
        self.ultimas_manos = []       # (puntos, pose) del último cuadro, para dibujar
        self.manos_en_mesa = 0        # manos detectadas en el último cuadro, en cualquier zona
        self.reiniciar()

    def reiniciar(self):
        self.espera = 0
        self.apuntar = 0              # cuadros seguidos con APUNTAR
        self.palma = []               # centros de la palma abierta en los últimos cuadros
        self.huecos = 0               # cuadros sin PALMA tolerados dentro de un barrido
        self.soltar = False           # después de un gesto: esperar a que la mano deje la pose

    def actualizar(self, manos):
        """manos: lista de (puntos (21,2), pose) válidas en este cuadro. Devuelve el gesto disparado o None."""
        if self.espera > 0:
            self.espera -= 1
        apuntando = [p for p, pose in manos if pose == "APUNTAR"]
        abiertas = [p for p, pose in manos if pose == "PALMA"]
        if self.soltar:
            if not apuntando and not abiertas:
                self.soltar = False
            return None

        self.apuntar = self.apuntar + 1 if apuntando else 0

        if abiertas:
            p = abiertas[0]
            self.palma.append((centro_palma(p), tamanio_palma(p)))
            self.palma = self.palma[-max(15, self.cuadros_palma):]
            self.huecos = 0
        else:
            self.huecos += 1
            if self.huecos > self.tolerancia:
                self.palma = []

        gesto = None
        if self.apuntar >= self.cuadros_min:
            gesto = "PEDIR"
        elif self.plantarse != "barrido" and len(self.palma) >= self.cuadros_palma:
            gesto = "PLANTARSE"                                      # palma sostenida
        elif self.plantarse != "estatico" and len(self.palma) >= 3 and self.desplazamiento() >= self.barrido:
            gesto = "PLANTARSE"                                      # barrido

        if gesto and self.espera == 0:
            self.espera = self.espera_total
            self.apuntar, self.palma, self.soltar = 0, [], True
            return gesto
        return None

    def desplazamiento(self):
        """Mayor distancia entre posiciones de la palma abierta (últimos 15 cuadros), en tamaños de palma."""
        if len(self.palma) < 2:
            return 0.0
        centros = np.array([c for c, _ in self.palma[-15:]])
        escala = float(np.median([s for _, s in self.palma]))
        d = np.linalg.norm(centros[:, None] - centros[None], axis=-1).max()
        return float(d / escala)

    def procesar(self, frame):
        """Detecta las manos en el cuadro y devuelve el gesto (o None)."""
        manos = self.detector.detectar(frame)
        alto = frame.shape[0]
        self.manos_en_mesa = len(manos)          # cualquier zona (Casa o Jugador): lo usa el motor de reglas
        validas = [m for m in manos if not self.zona_jugador or m[:, 1].mean() > alto / 2]
        self.ultimas_manos = [(m, clasificar_pose(m)) for m in validas]
        return self.actualizar(self.ultimas_manos)

    def progreso(self, pose):
        """Avance hacia el disparo (0-1), para dibujar una barra."""
        if pose == "APUNTAR":
            return min(1.0, self.apuntar / self.cuadros_min)
        if pose == "PALMA":
            sostenida = len(self.palma) / self.cuadros_palma if self.plantarse != "barrido" else 0.0
            barrido = self.desplazamiento() / self.barrido if self.plantarse != "estatico" else 0.0
            return min(1.0, max(sostenida, barrido))
        return 0.0


CONEXIONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def dibujar_manos(frame, reconocedor):
    import cv2
    colores = {"APUNTAR": (0, 255, 255), "PALMA": (255, 0, 255), "OTRA": (200, 200, 200)}
    for puntos, pose in reconocedor.ultimas_manos:
        c = colores[pose]
        for a, b in CONEXIONES:
            cv2.line(frame, tuple(int(v) for v in puntos[a]), tuple(int(v) for v in puntos[b]), c, 2, cv2.LINE_AA)
        x, y = puntos[:, 0].min(), puntos[:, 1].min()
        etiqueta = {"APUNTAR": "APUNTAR -> PEDIR", "PALMA": "PALMA -> PLANTARSE", "OTRA": "mano"}[pose]
        cv2.putText(frame, etiqueta, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, etiqueta, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
        if pose != "OTRA":
            ancho = int(120 * reconocedor.progreso(pose))
            cv2.rectangle(frame, (int(x), int(y) - 6), (int(x) + 120, int(y) - 2), (60, 60, 60), -1)
            cv2.rectangle(frame, (int(x), int(y) - 6), (int(x) + ancho, int(y) - 2), c, -1)
    return frame
