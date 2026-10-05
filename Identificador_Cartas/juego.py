"""Partida de Blackjack completa: cámara -> cartas -> motor de reglas, con gestos (o teclado) del Jugador.

Une las piezas del proyecto en un solo paso por cuadro:
    detector (pipeline A o B) -> seguimiento (votación, memoria, mediana) -> blackjack.Blackjack (reglas)
y dibuja sobre el cuadro las cartas, la etapa de la mano, la instrucción para el crupier / Jugador y el saldo.
Los gestos llegan desde afuera (módulo de gestos o teclado) con `procesar(frame, gesto="PEDIR" | "PLANTARSE")`.
"""

from types import SimpleNamespace

import cv2

from blackjack import Blackjack, Reglas, puntaje
from class_player import Entidad
from seguimiento import Seguimiento

COLOR_ETAPA = {"ESPERANDO": (200, 200, 200), "REPARTO": (0, 200, 255), "JUGADOR": (0, 255, 0),
               "CASA": (255, 160, 0), "FIN": (0, 255, 255)}


def _entidad(nombre, valores):
    e = Entidad(nombre)
    for v in valores:
        e.agregar_carta(SimpleNamespace(rank=v))
    return e


class Partida:
    def __init__(self, detector, reglas=None, ventana=15, memoria=10, confirmar=5, cuadros_vacia=20, gestos=None,
                 confirmar_no_pedida=20):
        """gestos: un Gestos.gestos.ReconocedorDeGestos (opcional). Si está, los gestos se detectan en cada cuadro."""
        self.gestos = gestos
        from identificador_cartas import analizar, dibujar   # import diferido (evita import circular)
        self._analizar, self._dibujar = analizar, dibujar
        self.detector = detector
        self.ventana, self.memoria = ventana, memoria
        self._config = dict(reglas=reglas or Reglas(), confirmar=confirmar, cuadros_vacia=cuadros_vacia,
                            confirmar_no_pedida=confirmar_no_pedida)
        self.juego = Blackjack(**self._config)
        self.reiniciar_vision()
        self.ultimo_gesto = None

    def nueva_mano(self):
        """Descarta la mano en curso (las cartas que hay en la mesa se toman como un reparto nuevo).
        Conserva el saldo y el historial."""
        self.juego._nueva_mano()
        self.juego._evento("etapa", "mano reiniciada")
        self.reiniciar_vision()

    def reiniciar_partida(self):
        """Partida nueva: saldo en 0, historial y eventos vacíos."""
        self.juego = Blackjack(**self._config)
        self.ultimo_gesto = None
        self.reiniciar_vision()
        if self.gestos is not None:
            self.gestos.reiniciar()

    def reiniciar_vision(self):
        self.seguimiento = Seguimiento(self.ventana, self.ventana, self.memoria)
        self.detector.reiniciar_seguimiento()

    def procesar(self, frame, gesto=None, info=""):
        """Procesa un cuadro (y un gesto opcional). Devuelve el cuadro dibujado."""
        esquinas, diagonales, casa, jugador = self._analizar(frame, self.detector, self.seguimiento)
        casa_m, jugador_m, _ = self.seguimiento.estabilizar(casa, jugador)
        if self.gestos is not None:
            detectado = self.gestos.procesar(frame)     # se corre siempre (MediaPipe usa la continuidad)
            gesto = gesto or detectado
        if gesto and self.juego.gesto(gesto):
            self.ultimo_gesto = (gesto, self.juego.cuadro)
        etapa_antes = self.juego.etapa
        self.juego.observar([c.rank for c in casa_m.cartas], [c.rank for c in jugador_m.cartas])
        if etapa_antes == "FIN" and self.juego.etapa == "ESPERANDO":
            self.reiniciar_vision()                     # mano nueva: se descartan los tracks viejos

        j = self.juego
        vista = self._dibujar(frame, esquinas, diagonales, casa, jugador, info,
                              mano=(_entidad("Casa", j.casa), _entidad("Jugador", j.jugador)),
                              mensaje=j.instruccion())
        self._panel(vista)
        if self.gestos is not None:
            from gestos import dibujar_manos
            dibujar_manos(vista, self.gestos)
        return vista

    def _panel(self, vista):
        """Recuadro con la etapa, los puntajes, el saldo y los últimos eventos."""
        j = self.juego
        h, w = vista.shape[:2]
        x0, y0, ancho, alto = w - 430, 50, 415, 175
        capa = vista.copy()
        cv2.rectangle(capa, (x0, y0), (x0 + ancho, y0 + alto), (25, 25, 25), -1)
        cv2.addWeighted(capa, 0.65, vista, 0.35, 0, vista)

        from identificador_cartas import a_ascii

        def linea(txt, y, color=(255, 255, 255), escala=0.5, grosor=1):
            cv2.putText(vista, a_ascii(txt), (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)

        linea(f"BLACKJACK | {j.etapa}", y0 + 24, COLOR_ETAPA.get(j.etapa, (255, 255, 255)), 0.65, 2)
        tj, blando = puntaje(j.jugador)
        tc, _ = puntaje(j.casa)
        linea(f"Jugador {tj}{' (blando)' if blando and tj < 21 else ''} | Casa {tc}", y0 + 50)
        linea(f"Saldo {j.saldo:+.0f}  (G {j.historial['JUGADOR'] + j.historial['BLACKJACK']} / "
              f"P {j.historial['CASA']} / E {j.historial['EMPATE']})", y0 + 72)
        if self.ultimo_gesto and j.cuadro - self.ultimo_gesto[1] < 30:
            linea(f"Gesto: {self.ultimo_gesto[0]}", y0 + 94, (0, 255, 255), 0.55, 2)
        for k, ev in enumerate(j.eventos[-3:]):
            color = (80, 80, 255) if ev.tipo == "irregularidad" else (190, 190, 190)
            linea(ev.texto[:52], y0 + 118 + 20 * k, color, 0.45)
