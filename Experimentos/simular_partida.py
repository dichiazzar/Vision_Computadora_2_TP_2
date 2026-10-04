"""Simulación de partidas de Blackjack completas, para probar toda la cadena sin cartas ni cámara.

Arma la mesa cuadro a cuadro pegando recortes de cartas REALES (de las fotos de prueba del mazo Bicycle Dragon)
sobre un fondo de mármol, siguiendo un guion de reparto: aparecen cartas, la Casa tiene una carta boca abajo
que después da vuelta, se levantan las cartas, etc. Se agrega movimiento de cámara en mano, ruido y compresión.
Los gestos del Jugador se inyectan en los momentos del guion (o los detecta el módulo de gestos, si se usa).

Cada cuadro pasa por la cadena completa: detector -> seguimiento -> motor de reglas (juego.Partida), y al final
se compara el resultado de cada mano con el esperado.

Uso (desde la carpeta del repo):
    python Experimentos/simular_partida.py                       # pipeline A, guarda videos/partida_simulada.mp4
    python Experimentos/simular_partida.py --pipeline B --sin-video
    python Experimentos/simular_partida.py --gestos     # los gestos los hace una mano real (HaGRID) sobre la mesa
                                                        # y los reconoce Gestos/gestos.py (MediaPipe + reglas)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Identificador_Cartas"))
sys.path.insert(0, str(REPO / "Experimentos"))
sys.path.insert(0, str(REPO / "Gestos"))

from class_carta import emparejar_esquinas            # noqa: E402
from detectores import DetectorA, crear_detector      # noqa: E402
from generar_video_prueba import senoide              # noqa: E402
from juego import Partida                             # noqa: E402

ANCHO, ALTO = 1600, 900
CASA_X, CASA_Y = [560, 790, 1020], 40          # posiciones (esquina superior izquierda) de las cartas
JUG_X, JUG_Y = [480, 710, 940, 1170], 520

# Guion: (cuadros que dura el paso, cartas de la Casa, cartas del Jugador, gesto al comienzo del paso)
# "DORSO" = carta boca abajo (la cámara no la reconoce). Resultado esperado al final de cada mano.
MANOS = [
    dict(nombre="pide y gana 18 a 17", esperado="JUGADOR", pasos=[
        (20, [], [], None),
        (30, [], ["7"], None),
        (30, ["Q"], ["7"], None),
        (30, ["Q", "DORSO"], ["7"], None),
        (35, ["Q", "DORSO"], ["7", "K"], None),            # 17
        (35, ["Q", "DORSO"], ["7", "K"], "PEDIR"),
        (35, ["Q", "DORSO"], ["7", "K", "AH"], None),      # 18
        (30, ["Q", "DORSO"], ["7", "K", "AH"], "PLANTARSE"),
        (45, ["Q", "7"], ["7", "K", "AH"], None),          # la Casa da vuelta: 17, se planta
        (30, [], [], None),
    ]),
    dict(nombre="blackjack del Jugador", esperado="BLACKJACK", pasos=[
        (30, [], ["AD"], None),
        (30, ["7"], ["AD"], None),
        (30, ["7", "DORSO"], ["AD"], None),
        (35, ["7", "DORSO"], ["AD", "K"], None),           # blackjack
        (45, ["7", "Q"], ["AD", "K"], None),               # la Casa da vuelta: 17
        (30, [], [], None),
    ]),
    dict(nombre="el Jugador se pasa", esperado="CASA", pasos=[
        (30, [], ["K"], None),
        (30, ["AH"], ["K"], None),
        (30, ["AH", "DORSO"], ["K"], None),
        (35, ["AH", "DORSO"], ["K", "Q"], None),           # 20
        (35, ["AH", "DORSO"], ["K", "Q"], "PEDIR"),
        (45, ["AH", "DORSO"], ["K", "Q", "7"], None),      # 27: se pasa
        (30, [], [], None),
    ]),
]


def recortar_cartas():
    """Recortes de cartas reales: se detectan las esquinas en las fotos y se recorta cada carta completa."""
    det = DetectorA(conf=0.5)
    fuentes = {"prueba.jpeg": {"Q": "Q", "7": "7", "K": "K"},
               "ACES.jpeg": {"A": None}}
    cartas = {}
    for foto, nombres in fuentes.items():
        img = cv2.imread(str(REPO / foto))
        completas = [c for c in emparejar_esquinas(det.detectar(img))[0] if c.completa]
        ases = 0
        for c in sorted(completas, key=lambda c: (c.centro[1] > img.shape[0] / 2, c.centro[0])):
            x1, y1 = (int(v) - 8 for v in c.sup_izq)
            x2, y2 = (int(v) + 8 for v in c.inf_der)
            recorte = img[max(0, y1):y2, max(0, x1):x2].copy()
            if c.rank == "A":     # ACES.jpeg: arriba A corazones; abajo A diamantes (izq.) y A tréboles (der.)
                cartas[["AH", "AD", "AC"][ases]] = recorte
                ases += 1
            elif c.rank in nombres:
                cartas[c.rank] = recorte
    prueba = cv2.imread(str(REPO / "prueba.jpeg"))
    cartas["DORSO"] = prueba[22:272, 758:940].copy()           # carta boca abajo de prueba.jpeg
    return cartas


def fondo_marmol():
    """Fondo de mesa: un parche limpio de prueba.jpeg repetido en espejo."""
    parche = cv2.imread(str(REPO / "prueba.jpeg"))[320:880, 1000:1340]
    fila = np.hstack([parche, cv2.flip(parche, 1)] * 3)
    mosaico = np.vstack([fila, cv2.flip(fila, 0)])
    return cv2.resize(mosaico[:, :], (ANCHO, ALTO))


def componer(fondo, cartas, casa, jugador):
    img = fondo.copy()
    for nombres, xs, y in ((casa, CASA_X, CASA_Y), (jugador, JUG_X, JUG_Y)):
        for nombre, x in zip(nombres, xs):
            c = cartas[nombre]
            h, w = c.shape[:2]
            img[y:y + h, x:x + w] = c[: ALTO - y, : ANCHO - x]
    return img


# Manos reales de HaGRID (Gestos/datos_hagrid, las baja Gestos/evaluar_gestos.py) que MediaPipe detecta bien
MANOS_GESTO = {"PEDIR": ("one", "12812.jpg"), "PLANTARSE": ("palm", "14553.jpg")}


def cargar_manos():
    from gestos import DetectorDeManos
    from simular_gestos import recortar_mano
    det = DetectorDeManos(num_manos=1, modo="imagen")
    manos = {}
    for gesto, (clase, archivo) in MANOS_GESTO.items():
        img = cv2.imread(str(REPO / "Gestos" / "datos_hagrid" / clase / archivo))
        assert img is not None, f"falta {clase}/{archivo}: correr antes python Gestos/evaluar_gestos.py"
        m = recortar_mano(img, det)
        esc = 300 / max(m.shape[:2])
        manos[gesto] = cv2.resize(m, None, fx=esc, fy=esc)
    return manos


def pegar_mano(img, mano, gesto, k, cuadros=15):
    """Durante los primeros `cuadros` del paso pega la mano: quieta (PEDIR) o barriendo la mesa (PLANTARSE)."""
    if k >= cuadros:
        return img
    h, w = mano.shape[:2]
    x = 1150 if gesto == "PEDIR" else 850 + int(k * 500 / cuadros)
    y = ALTO - h - 15
    img = img.copy()
    img[y:y + h, x:x + w] = mano[:, : ANCHO - x]
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="A")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--salida", default=str(REPO / "videos" / "partida_simulada.mp4"))
    ap.add_argument("--sin-video", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gestos", action="store_true", help="gestos hechos por una mano real y reconocidos")
    args = ap.parse_args()

    cartas, fondo = recortar_cartas(), fondo_marmol()
    faltan = {"Q", "7", "K", "AH", "AD", "DORSO"} - set(cartas)
    assert not faltan, f"no se pudieron recortar: {faltan}"

    guion = [(paso, mano) for mano in MANOS for paso in mano["pasos"]]
    n = sum(p[0] for p, _ in guion)
    rng = np.random.default_rng(args.seed)
    t = np.arange(n) / 10
    ang, tx, ty = senoide(t, rng, 3.0), senoide(t, rng, 0.02 * ANCHO), senoide(t, rng, 0.02 * ALTO)

    reconocedor, manos = None, {}
    if args.gestos:
        from gestos import DetectorDeManos, ReconocedorDeGestos
        reconocedor = ReconocedorDeGestos(DetectorDeManos(num_manos=2, modo="video"))
        manos = cargar_manos()
    partida = Partida(crear_detector(args.pipeline, conf=args.conf), gestos=reconocedor)
    writer = None
    if not args.sin_video:
        Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
        from video import GrabadorVideo
        writer = GrabadorVideo(args.salida, 10, (ANCHO // 2 * 2, ALTO))

    resultados, i = [], 0
    for mano in MANOS:
        antes = dict(partida.juego.historial)
        for duracion, casa, jugador, gesto in mano["pasos"]:
            base = componer(fondo, cartas, casa, jugador)
            for k in range(duracion):
                M = cv2.getRotationMatrix2D((ANCHO / 2, ALTO / 2), ang[i], 1.0)
                M[:, 2] += (tx[i], ty[i])
                cuadro = cv2.warpAffine(base, M, (ANCHO, ALTO), borderMode=cv2.BORDER_REFLECT)
                ruido = rng.normal(0, 3, cuadro.shape).astype(np.int16)
                cuadro = np.clip(cuadro.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
                if args.gestos:
                    if gesto:
                        cuadro = pegar_mano(cuadro, manos[gesto], gesto, k)
                    vista = partida.procesar(cuadro, info=f"cuadro {i}")
                else:
                    vista = partida.procesar(cuadro, gesto=gesto if k == 5 else None, info=f"cuadro {i}")
                if writer is not None:
                    writer.write(vista)
                i += 1
        # el resultado se toma del historial: al levantar las cartas el motor ya empezó otra mano
        nuevos = [k for k, v in partida.juego.historial.items() if v > antes.get(k, 0)]
        obtenido = nuevos[0] if len(nuevos) == 1 else ("SIN TERMINAR" if not nuevos else "+".join(nuevos))
        resultados.append((mano["nombre"], mano["esperado"], obtenido))
    if writer is not None:
        writer.release()

    print("\nEventos de la partida:")
    for ev in partida.juego.eventos:
        print(f"  [{ev.cuadro:4d}] {ev.tipo:13s} {ev.texto}")
    print("\nResultados:")
    ok = 0
    for nombre, esperado, obtenido in resultados:
        ok += esperado == obtenido
        print(f"  {'OK ' if esperado == obtenido else 'MAL'} {nombre:28s} esperado {esperado:10s} obtenido {obtenido}")
    print(f"\n{ok}/{len(resultados)} manos correctas | saldo final {partida.juego.saldo:+.0f} (esperado +15)")
    if writer is not None:
        print("video:", args.salida)


if __name__ == "__main__":
    main()
