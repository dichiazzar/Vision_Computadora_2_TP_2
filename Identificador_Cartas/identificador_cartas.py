'''Identificador de cartas para Blackjack.

1. Un detector (pipeline A o B, ver detectores.py) encuentra las esquinas con índice y su valor.
2. Las esquinas se agrupan en cartas: dos esquinas del mismo valor en la diagonal de una carta
   forman una carta; una esquina sin pareja (la otra tapada) también cuenta como carta.
3. Cada carta se asigna a la Casa (mitad superior de la imagen) o al Jugador (mitad inferior)
   según su centro, se calcula el puntaje de Blackjack y quién gana.
4. En video / cámara se agrega seguimiento (seguimiento.py): ByteTrack + votación del valor por track, y se
   muestra la mano más frecuente de los últimos cuadros, para que el resultado no parpadee.

Uso (desde la carpeta del repo o desde Identificador_Cartas/):
    python Identificador_Cartas/identificador_cartas.py --imagen prueba.jpeg
    python Identificador_Cartas/identificador_cartas.py --imagen ACES_TAPADOS.jpeg --pipeline B
    python Identificador_Cartas/identificador_cartas.py --camara 0
    python Identificador_Cartas/identificador_cartas.py --video partida.mp4 --guardar salida.mp4
    python Identificador_Cartas/identificador_cartas.py --video prueba.mp4 --esperado "Q|7 K" --sin-ventana
    python Identificador_Cartas/identificador_cartas.py --camara 0 --juego --gestos      # partida de Blackjack
'''

import argparse
import sys
import time
import unicodedata
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from class_carta import emparejar_esquinas
from class_player import Entidad, determinar_ganador
from detectores import REPO, crear_detector
from seguimiento import Seguimiento
from video import GrabadorVideo

VERDE, AZUL, AMARILLO, ROJO, BLANCO, GRIS = (0, 255, 0), (255, 0, 0), (0, 255, 255), (0, 0, 255), (255, 255, 255), (160, 160, 160)


def analizar(frame, detector, seguimiento=None):
    """Detecta, arma las cartas y las reparte entre Casa y Jugador.
    Con `seguimiento`, el valor de cada esquina es el votado por su track a lo largo de los cuadros."""
    esquinas = detector.detectar(frame, seguir=seguimiento is not None)
    if seguimiento is not None:
        esquinas = seguimiento.votar(esquinas)
    cartas, diagonales = emparejar_esquinas(esquinas)

    casa, jugador = Entidad("Casa"), Entidad("Jugador")
    mitad = frame.shape[0] / 2
    for carta in sorted(cartas, key=lambda c: c.centro[0]):
        (casa if carta.centro[1] < mitad else jugador).agregar_carta(carta)
    return esquinas, diagonales, casa, jugador


def a_ascii(msg):
    """OpenCV sólo dibuja ASCII: 'pidió' -> 'pidio', '¡Blackjack!' -> 'Blackjack!'."""
    return unicodedata.normalize("NFKD", str(msg)).encode("ascii", "ignore").decode()


def texto(frame, msg, org, escala, color, grosor=2):
    msg = a_ascii(msg)
    cv2.putText(frame, msg, org, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grosor + 2, cv2.LINE_AA)
    cv2.putText(frame, msg, org, cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def dibujar(frame, esquinas, diagonales, casa, jugador, info="", mano=None, mensaje=None):
    """Dibuja las esquinas y cartas de este cuadro. `mano` = (casa, jugador) a mostrar como resultado
    (en video, la mano estabilizada); por defecto, la del cuadro."""
    frame = frame.copy()
    height, width = frame.shape[:2]

    # 1. Diagonales entre las dos esquinas de cada carta
    for esquina1, esquina2, distancia, angulo in diagonales:
        p1 = tuple(int(v) for v in esquina1.centroid)
        p2 = tuple(int(v) for v in esquina2.centroid)
        cv2.line(frame, p1, p2, AMARILLO, 2)
        medio = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        texto(frame, f"d={distancia:.0f}px a={angulo:.0f}", (medio[0] + 5, medio[1]), 0.45, AMARILLO, 1)

    # 2. Esquinas detectadas
    for esquina in esquinas:
        x1, y1 = map(int, esquina.sup_izq)
        x2, y2 = map(int, esquina.inf_der)
        color = GRIS if getattr(esquina, "recordada", False) else VERDE    # gris = recordada, no detectada
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        etiqueta = f"{esquina.rank} {esquina.conf:.2f}"
        if esquina.track_id is not None:
            etiqueta = f"#{esquina.track_id} " + etiqueta
        texto(frame, etiqueta, (x1, y1 - 5), 0.5, color, 1)

    # 3. Cartas reconstruidas (línea punteada = carta con una sola esquina visible)
    for entidad in (casa, jugador):
        for carta in entidad.cartas:
            x1, y1 = map(int, carta.sup_izq)
            x2, y2 = map(int, carta.inf_der)
            cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), AZUL, 3 if carta.completa else 1)
            texto(frame, f"Carta: {carta.rank}", (x1, y2 + 22), 0.6, AZUL)

    # 4. Línea Casa / Jugador, puntajes y resultado
    if mano is not None:
        casa, jugador = mano
    cv2.line(frame, (0, height // 2), (width, height // 2), ROJO, 2)
    texto(frame, str(casa), (20, 30), 0.8, ROJO)
    texto(frame, str(jugador), (20, height // 2 + 30), 0.8, ROJO)
    if mensaje is None:
        mensaje = determinar_ganador(jugador, casa)
    texto(frame, mensaje, (20, height - 20), 0.9 if len(mensaje) < 60 else 0.6, BLANCO)
    if info:
        (ancho_txt, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 3)
        texto(frame, info, (width - ancho_txt - 15, 30), 0.55, BLANCO, 1)
    return frame


def resolver_ruta(p):
    """Acepta rutas relativas a la carpeta actual o a la raíz del repo."""
    p = Path(p)
    return p if p.exists() else REPO / p


def modo_imagen(args, detector):
    ruta = resolver_ruta(args.imagen)
    frame = cv2.imread(str(ruta))
    if frame is None:
        raise FileNotFoundError(f"No se pudo leer la imagen {ruta}")

    esquinas, diagonales, casa, jugador = analizar(frame, detector)
    print(f"[{detector.nombre}] {ruta.name}: {len(esquinas)} esquinas")
    print(" ", casa)
    print(" ", jugador)
    print(" ", determinar_ganador(jugador, casa))

    resultado = dibujar(frame, esquinas, diagonales, casa, jugador, detector.nombre)
    if args.guardar:
        cv2.imwrite(args.guardar, resultado)
        print("guardado en", args.guardar)
    if not args.sin_ventana:
        cv2.imshow("Detector de cartas", resultado)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def abrir_fuente(fuente, args):
    """Abre la cámara (pidiendo la resolución de --resolucion; por defecto OpenCV usa 640x480 y las esquinas de
    las cartas quedan chicas) o el archivo de video."""
    cap = cv2.VideoCapture(fuente)
    if isinstance(fuente, int) and cap.isOpened():
        ancho, alto = (int(v) for v in args.resolucion.lower().split("x"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
        print(f"cámara {fuente}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    return cap


class ControlSenal:
    """Detecta una cámara que no envía imagen (cuadros casi negros seguidos), p. ej. el celular por Vínculo a
    Windows cuando su transmisión está tomada por otra app o falta aceptar la notificación en el teléfono."""

    def __init__(self, activo, cuadros=30):
        self.activo, self.cuadros, self.negros, self.avisado = activo, cuadros, 0, False

    def revisar(self, frame, vista):
        if not self.activo:
            return
        self.negros = self.negros + 1 if float(frame.mean()) < 8 else 0
        if self.negros >= self.cuadros:
            lineas = ["LA CAMARA NO ENVIA IMAGEN",
                      "- Cerrar Phone Link / la app Camara si muestran el celular",
                      "- Aceptar la notificacion de Vinculo a Windows en el celular (desbloqueado)",
                      "- O probar otra camara: --camara 0 / 1 / 2"]
            for k, txt in enumerate(lineas):
                texto(vista, txt, (40, 120 + 40 * k), 0.9 if k == 0 else 0.7, (0, 0, 255) if k == 0 else BLANCO)
            if not self.avisado:
                print("La cámara no envía imagen (cuadros negros). " + " ".join(lineas[1:]))
                self.avisado = True
        elif self.negros == 0:
            self.avisado = False


def grabar_crudo(grabador, frame, args, cap):
    """--guardar-crudo: graba la imagen de la cámara TAL CUAL llega (sin cajas, textos ni esqueletos), con más
    calidad que el video anotado. Sirve para juntar material de entrenamiento con el escenario exacto de la demo
    (después: python "Dataset Real/extraer_cuadros.py"). Devuelve el grabador (lo crea en el primer cuadro)."""
    if not args.guardar_crudo:
        return None
    if grabador is None:
        fps = cap.get(cv2.CAP_PROP_FPS) if args.video else 15
        grabador = GrabadorVideo(args.guardar_crudo, fps or 15, frame.shape[1::-1],
                                 tiempo_real=args.camara is not None, crf=18)
        print("grabando la imagen cruda en", args.guardar_crudo)
    grabador.write(frame)
    return grabador


def parsear_mano(texto_mano):
    """'Q|7 K' -> (('Q',), ('7', 'K')): cartas esperadas de Casa | Jugador."""
    casa, jugador = texto_mano.split("|")
    return tuple(sorted(casa.split())), tuple(sorted(jugador.split()))


def modo_video(args, detector):
    fuente = args.camara if args.camara is not None else str(resolver_ruta(args.video))
    cap = abrir_fuente(fuente, args)
    senal = ControlSenal(activo=args.camara is not None)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {fuente!r}")

    seguimiento = None if args.sin_seguimiento else Seguimiento(args.ventana, args.ventana, args.memoria)
    detector.reiniciar_seguimiento()
    esperado = parsear_mano(args.esperado) if args.esperado else None
    stats = dict(cuadros=0, cambios_cuadro=0, cambios_mostrada=0, ok_cuadro=0, ok_mostrada=0)
    previa_cuadro = previa_mostrada = None

    writer, crudo, fps, t_prev = None, None, 0.0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_cuadros and stats["cuadros"] >= args.max_cuadros):
                break
            crudo = grabar_crudo(crudo, frame, args, cap)
            esquinas, diagonales, casa, jugador = analizar(frame, detector, seguimiento)
            mano_cuadro = Seguimiento.clave(casa, jugador)
            if seguimiento is not None:
                casa_m, jugador_m, estabilidad = seguimiento.estabilizar(casa, jugador)
            else:
                casa_m, jugador_m, estabilidad = casa, jugador, None
            mano_mostrada = Seguimiento.clave(casa_m, jugador_m)

            # estadísticas: cuántas veces cambia lo que se ve y, si se conoce la mano real, cuántas veces acierta
            stats["cuadros"] += 1
            stats["cambios_cuadro"] += previa_cuadro is not None and mano_cuadro != previa_cuadro
            stats["cambios_mostrada"] += previa_mostrada is not None and mano_mostrada != previa_mostrada
            previa_cuadro, previa_mostrada = mano_cuadro, mano_mostrada
            if esperado:
                stats["ok_cuadro"] += mano_cuadro == esperado
                stats["ok_mostrada"] += mano_mostrada == esperado

            t = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(t - t_prev, 1e-6) if fps else 1 / max(t - t_prev, 1e-6)
            t_prev = t
            info = f"{detector.nombre} | {fps:.1f} FPS"
            if estabilidad is not None:
                info += f" | estable {estabilidad:.0%}"
            resultado = dibujar(frame, esquinas, diagonales, casa, jugador, info, mano=(casa_m, jugador_m))
            senal.revisar(frame, resultado)

            if args.guardar:
                if writer is None:
                    out_fps = cap.get(cv2.CAP_PROP_FPS) if args.video else max(1, round(fps))
                    writer = GrabadorVideo(args.guardar, out_fps or 20, tiempo_real=args.camara is not None,
                                           tamanio=resultado.shape[1::-1])
                writer.write(resultado)
            if not args.sin_ventana:
                cv2.imshow("Detector de cartas (q para salir, r reinicia)", resultado)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (ord("q"), 27):
                    break
                if tecla == ord("r") and seguimiento is not None:      # nueva mano
                    seguimiento = Seguimiento(args.ventana, args.ventana, args.memoria)
                    detector.reiniciar_seguimiento()
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if crudo is not None:
            crudo.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    n = max(stats["cuadros"], 1)
    print(f"[{detector.nombre}] {stats['cuadros']} cuadros | seguimiento: {'no' if seguimiento is None else 'sí'}")
    print(f"  cambios de mano: por cuadro {stats['cambios_cuadro']} | mostrada {stats['cambios_mostrada']}")
    if esperado:
        print(f"  mano correcta: por cuadro {stats['ok_cuadro'] / n:.1%} | mostrada {stats['ok_mostrada'] / n:.1%}")
    if seguimiento is not None:
        print(f"  tracks de esquinas creados: {len(seguimiento.tracks_totales)}")
    return stats


def modo_juego(args, detector):
    """Partida de Blackjack con la cámara (o un video): reglas + gestos (--gestos) o teclado (p / l)."""
    from juego import Partida
    from blackjack import Reglas

    gestos = None
    if args.gestos:
        sys.path.insert(0, str(REPO / "Gestos"))
        from gestos import DetectorDeManos, ReconocedorDeGestos
        gestos = ReconocedorDeGestos(DetectorDeManos(num_manos=2, modo="video"),
                                     plantarse=args.plantarse)
    partida = Partida(detector, Reglas(casa_pide_17_blando=args.pide_17_blando), ventana=args.ventana,
                      memoria=args.memoria, gestos=gestos)
    fuente = args.camara if args.camara is not None else str(resolver_ruta(args.video))
    cap = abrir_fuente(fuente, args)
    senal = ControlSenal(activo=args.camara is not None)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {fuente!r}")
    writer, crudo, fps, t_prev, tecla = None, None, 0.0, time.perf_counter(), 255
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            crudo = grabar_crudo(crudo, frame, args, cap)
            gesto = {ord("p"): "PEDIR", ord("l"): "PLANTARSE"}.get(tecla)
            t = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(t - t_prev, 1e-6) if fps else 1 / max(t - t_prev, 1e-6)
            t_prev = t
            vista = partida.procesar(frame, gesto=gesto, info=f"{detector.nombre} | {fps:.1f} FPS")
            senal.revisar(frame, vista)
            if args.guardar:
                if writer is None:
                    out_fps = cap.get(cv2.CAP_PROP_FPS) if args.video else max(1, round(fps))
                    writer = GrabadorVideo(args.guardar, out_fps or 15, tiempo_real=args.camara is not None,
                                           tamanio=vista.shape[1::-1])
                writer.write(vista)
            tecla = 255
            if not args.sin_ventana:
                cv2.imshow("Blackjack (p pide, l se planta, n nueva mano, r partida nueva, q sale)", vista)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (ord("q"), 27):
                    break
                if tecla == ord("n"):        # descarta la mano en curso; conserva saldo e historial
                    partida.nueva_mano()
                if tecla == ord("r"):        # partida nueva: saldo en 0
                    partida.reiniciar_partida()
                    print("--- partida reiniciada ---")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if crudo is not None:
            crudo.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    j = partida.juego
    print(f"Manos jugadas: {sum(j.historial.values())} | {dict(j.historial)} | saldo {j.saldo:+.0f}")
    for ev in j.eventos:
        print(f"  [{ev.cuadro:5d}] {ev.tipo:13s} {ev.texto}")


def main():
    p = argparse.ArgumentParser(description="Identificador de cartas para Blackjack")
    fuente = p.add_mutually_exclusive_group(required=True)
    fuente.add_argument("--imagen", help="ruta a una imagen")
    fuente.add_argument("--camara", type=int, help="índice de la cámara (0 = la integrada)")
    fuente.add_argument("--video", help="ruta a un video")
    p.add_argument("--pipeline", default="A", choices=["A", "B", "a", "b"],
                   help="A = YOLO-13 | B = YOLO-1 + CNN-13")
    p.add_argument("--conf", type=float, default=0.5, help="confianza mínima de detección")
    p.add_argument("--resolucion", default="1280x720", help="cámara: resolución pedida (p. ej. 1920x1080)")
    p.add_argument("--pesos", default="auto", choices=["auto", "base", "real", "propio"],
                   help="auto = propio si existe, si no base | base = dataset sintético | "
                        "real = ajustados con teogopk | propio = ajustados con fotos propias")
    p.add_argument("--guardar", help="ruta para guardar el resultado (imagen o .mp4)")
    p.add_argument("--guardar-crudo", help="video/cámara: grabar también la imagen limpia, sin dibujos (.mp4)")
    p.add_argument("--sin-ventana", action="store_true", help="no abrir ventana (sólo imprimir / guardar)")
    p.add_argument("--sin-seguimiento", action="store_true", help="video: procesar cada cuadro por separado")
    p.add_argument("--ventana", type=int, default=15, help="video: cuadros para la votación y la mano mostrada")
    p.add_argument("--memoria", type=int, default=10, help="video: cuadros que se recuerda una esquina perdida")
    p.add_argument("--esperado", help='video: mano real "Casa|Jugador", p. ej. "Q|7 K", para medir el acierto')
    p.add_argument("--max-cuadros", type=int, default=0, help="video: cortar después de N cuadros")
    p.add_argument("--juego", action="store_true", help="video/cámara: partida de Blackjack con reglas")
    p.add_argument("--gestos", action="store_true", help="juego: controlar con gestos de la mano (MediaPipe)")
    p.add_argument("--plantarse", default="ambos", choices=["ambos", "barrido", "estatico"],
                   help="juego: plantarse pasando la mano (barrido), con la palma quieta ~1 s (estatico) o ambos")
    p.add_argument("--pide-17-blando", action="store_true", help="juego: la Casa pide con 17 blando")
    args = p.parse_args()

    detector = crear_detector(args.pipeline, conf=args.conf, pesos=args.pesos)
    if args.imagen:
        modo_imagen(args, detector)
    elif args.juego:
        modo_juego(args, detector)
    else:
        modo_video(args, detector)


if __name__ == "__main__":
    main()
