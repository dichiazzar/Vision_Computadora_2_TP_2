"""Motor de reglas de Blackjack (un Jugador contra la Casa), alimentado por la cámara y los gestos.

La cámara es la fuente de verdad de las cartas: el motor no genera cartas, recibe en cada cuadro la mano
observada (ya estabilizada por seguimiento.py) y los gestos del jugador ("PEDIR" / "PLANTARSE"), y decide
en qué etapa está la mano, qué tiene que hacer cada uno y quién gana.

Etapas:
    ESPERANDO   mesa vacía, se espera el reparto
    REPARTO     se reparten 2 cartas al Jugador y 1 visible a la Casa (la otra va boca abajo: la cámara no la ve)
    JUGADOR     el Jugador pide cartas (gesto PEDIR) o se planta (gesto PLANTARSE)
    CASA        la Casa da vuelta su carta y pide hasta llegar a 17 (regla configurable para el 17 blando)
    FIN         resultado; cuando se levantan las cartas de la mesa empieza otra mano

Robustez frente a la visión:
    - Una mano observada se acepta recién cuando se repite `confirmar` cuadros seguidos (filtra ruido).
    - Si el cambio agrega cartas que nadie pidió (en el turno del Jugador, una carta suya sin haberla pedido o una
      de la Casa; en el turno de la Casa, una del Jugador), tiene que repetirse `confirmar_no_pedida` cuadros
      (~2 s): una lectura fantasma de unos instantes no cambia la mano.
    - Durante una mano las cartas sólo se agregan: si una carta deja de verse (la tapa una mano, un reflejo)
      se asume que sigue ahí. La mano termina cuando la mesa queda vacía `cuadros_vacia` cuadros.
    - Si una carta cambia de valor (lectura corregida) se acepta la nueva lectura estable.
    - Cartas que aparecen sin que nadie las pida se aceptan pero se registran como "irregularidad".
"""

from collections import Counter
from dataclasses import dataclass, field

PUNTOS = {"A": 1, "J": 10, "Q": 10, "K": 10, **{str(i): i for i in range(2, 11)}}


def puntaje(cartas):
    """(total, blando): el As vale 11 si no se pasa de 21; blando = hay un As contando 11."""
    total = sum(PUNTOS[c] for c in cartas)
    if "A" in cartas and total + 10 <= 21:
        return total + 10, True
    return total, False


def es_blackjack(cartas):
    return len(cartas) == 2 and puntaje(cartas)[0] == 21


@dataclass
class Reglas:
    casa_pide_17_blando: bool = False   # False = la Casa se planta en todo 17 ("S17", lo más común)
    pago_blackjack: float = 1.5         # el blackjack natural paga 3:2
    apuesta: float = 10.0               # apuesta fija por mano (para llevar el saldo)


@dataclass
class Evento:
    cuadro: int
    tipo: str        # etapa | carta | gesto | irregularidad | resultado
    texto: str


@dataclass
class Blackjack:
    reglas: Reglas = field(default_factory=Reglas)
    confirmar: int = 5          # cuadros iguales para aceptar un cambio en la mesa
    confirmar_no_pedida: int = 20   # ídem si el cambio agrega cartas que nadie pidió (más desconfiado)
    cuadros_vacia: int = 20     # cuadros con la mesa vacía para dar la mano por terminada

    def __post_init__(self):
        self.saldo = 0.0
        self.historial = Counter()       # resultados de todas las manos
        self.eventos = []
        self.cuadro = 0
        self._nueva_mano()

    # ------------------------------------------------------------------ estado
    def _nueva_mano(self):
        self.etapa = "ESPERANDO"
        self.jugador, self.casa = [], []            # cartas aceptadas (lista de valores)
        self.esperando_carta = False                # el Jugador pidió y todavía no llegó la carta
        self.resultado = None
        self._candidata, self._repeticiones, self._vacia = None, 0, 0

    def _evento(self, tipo, texto):
        self.eventos.append(Evento(self.cuadro, tipo, texto))

    def _etapa(self, nueva):
        if nueva != self.etapa:
            self.etapa = nueva
            self._evento("etapa", nueva)

    # ------------------------------------------------------------------ entradas
    def gesto(self, nombre):
        """Gesto del Jugador: "PEDIR" o "PLANTARSE". Se ignora fuera de su turno y mientras espera la carta que
        pidió (la mano del crupier repartiendo podría parecer un gesto)."""
        if self.etapa != "JUGADOR" or self.esperando_carta:
            return False
        if nombre == "PEDIR":
            self.esperando_carta = True
            self._evento("gesto", "el Jugador pide carta")
            return True
        if nombre == "PLANTARSE":
            self._evento("gesto", f"el Jugador se planta con {puntaje(self.jugador)[0]}")
            self._etapa("CASA")
            self._avanzar_casa()
            return True
        return False

    def observar(self, casa, jugador):
        """Mano observada en este cuadro (listas de valores de Casa y Jugador). Devuelve la etapa."""
        self.cuadro += 1
        casa, jugador = sorted(casa), sorted(jugador)

        # mesa vacía: fin de la mano (o seguir esperando)
        if not casa and not jugador:
            self._vacia += 1
            if self._vacia >= self.cuadros_vacia and self.etapa != "ESPERANDO":
                if self.etapa != "FIN":
                    self._evento("irregularidad", "se levantaron las cartas antes de terminar la mano")
                self._nueva_mano()
            return self.etapa
        self._vacia = 0
        if self.etapa == "FIN":
            return self.etapa

        # debounce: la observación tiene que repetirse `confirmar` cuadros
        clave = (tuple(casa), tuple(jugador))
        if clave == self._candidata:
            self._repeticiones += 1
        else:
            self._candidata, self._repeticiones = clave, 1
        requeridas = self.confirmar_no_pedida if self._agrega_no_pedidas(casa, jugador) else self.confirmar
        if self._repeticiones != requeridas:        # se procesa una sola vez, al confirmarse
            return self.etapa

        nuevas_j = self._actualizar(self.jugador, jugador, "Jugador")
        nuevas_c = self._actualizar(self.casa, casa, "Casa")
        self._procesar(nuevas_c, nuevas_j)
        return self.etapa

    def _agrega_no_pedidas(self, casa, jugador):
        """¿La observación agrega cartas que nadie pidió en esta etapa?"""
        def agregadas(aceptadas, observadas):
            return sum((Counter(observadas) - Counter(aceptadas)).values()) if len(observadas) > len(aceptadas) else 0
        nuevas_j, nuevas_c = agregadas(self.jugador, jugador), agregadas(self.casa, casa)
        if self.etapa == "JUGADOR":
            return nuevas_c > 0 or (nuevas_j > 0 and not self.esperando_carta)
        if self.etapa == "CASA":
            return nuevas_j > 0
        return False

    def _actualizar(self, aceptadas, observadas, quien):
        """Incorpora las cartas nuevas. Las que dejaron de verse se mantienen (tapadas).
        Devuelve la cantidad de cartas nuevas (-1 si se corrigió la lectura de alguna carta)."""
        faltan = Counter(aceptadas) - Counter(observadas)       # aceptadas que ya no se ven
        sobran = Counter(observadas) - Counter(aceptadas)       # observadas que no estaban
        nuevas = sum(sobran.values())
        # misma cantidad pero distinto valor: la visión corrigió una lectura
        if len(observadas) == len(aceptadas) and faltan:
            aceptadas[:] = observadas
            self._evento("carta", f"{quien}: se corrige la lectura -> {' '.join(observadas)}")
            return -1
        for v in sobran.elements():
            aceptadas.append(v)
            self._evento("carta", f"{quien} recibe {v}")
        return nuevas

    # ------------------------------------------------------------------ lógica del juego
    def _procesar(self, nuevas_casa, nuevas_jugador):
        if self.etapa == "ESPERANDO":
            self._etapa("REPARTO")

        if self.etapa == "REPARTO":
            if len(self.casa) > 1:
                self._evento("irregularidad", "la Casa tiene dos cartas visibles en el reparto "
                                              "(la segunda va boca abajo)")
            if len(self.jugador) >= 2 and len(self.casa) >= 1:
                if es_blackjack(self.jugador):
                    self._evento("carta", "¡Blackjack del Jugador!")
                    self._etapa("CASA")             # la Casa sólo da vuelta su carta
                    self._avanzar_casa()
                else:
                    self._etapa("JUGADOR")
                    self._turno_jugador_actualizado()
            return

        if self.etapa == "JUGADOR":
            if nuevas_casa > 0:
                self._evento("irregularidad", "la Casa recibió carta durante el turno del Jugador")
            if nuevas_jugador < 0:              # lectura corregida: recalcular
                self._turno_jugador_actualizado()
            elif nuevas_jugador > 0:
                if not self.esperando_carta:
                    self._evento("irregularidad", "carta al Jugador sin que la pidiera")
                self.esperando_carta = False
                self._turno_jugador_actualizado()
            return

        if self.etapa == "CASA":
            if nuevas_jugador > 0:
                self._evento("irregularidad", "el Jugador recibió carta en el turno de la Casa")
            self._avanzar_casa()

    def _turno_jugador_actualizado(self):
        total, _ = puntaje(self.jugador)
        if total > 21:
            self._terminar("CASA", f"el Jugador se pasa con {total}")
        elif total == 21:
            self._evento("gesto", "el Jugador llega a 21: se planta automáticamente")
            self._etapa("CASA")
            self._avanzar_casa()

    def _casa_debe_pedir(self):
        total, blando = puntaje(self.casa)
        return total < 17 or (total == 17 and blando and self.reglas.casa_pide_17_blando)

    def _avanzar_casa(self):
        if len(self.casa) < 2:          # todavía no dio vuelta su carta
            return
        total_j, total_c = puntaje(self.jugador)[0], puntaje(self.casa)[0]
        bj_j, bj_c = es_blackjack(self.jugador), es_blackjack(self.casa)
        if bj_j or bj_c:
            if bj_j and bj_c:
                self._terminar("EMPATE", "ambos tienen blackjack")
            elif bj_j:
                self._terminar("BLACKJACK", "blackjack del Jugador")
            else:
                self._terminar("CASA", "blackjack de la Casa")
            return
        if self._casa_debe_pedir():
            return
        if total_c > 21:
            self._terminar("JUGADOR", f"la Casa se pasa con {total_c}")
        elif total_j > total_c:
            self._terminar("JUGADOR", f"{total_j} a {total_c}")
        elif total_j < total_c:
            self._terminar("CASA", f"{total_c} a {total_j}")
        else:
            self._terminar("EMPATE", f"empate en {total_j}")

    def _terminar(self, ganador, detalle):
        pago = {"JUGADOR": 1.0, "BLACKJACK": self.reglas.pago_blackjack, "CASA": -1.0, "EMPATE": 0.0}[ganador]
        self.saldo += pago * self.reglas.apuesta
        self.historial[ganador] += 1
        self.resultado = (ganador, detalle)
        self._evento("resultado", f"{ganador}: {detalle} ({pago * self.reglas.apuesta:+.0f})")
        self._etapa("FIN")

    # ------------------------------------------------------------------ para mostrar en pantalla
    def instruccion(self):
        """Qué tiene que pasar ahora, para mostrar en pantalla."""
        tj, tc = puntaje(self.jugador)[0], puntaje(self.casa)[0]
        if self.etapa == "ESPERANDO":
            return "Repartir: 2 cartas al Jugador y 1 a la Casa"
        if self.etapa == "REPARTO":
            return "Repartiendo..."
        if self.etapa == "JUGADOR":
            if self.esperando_carta:
                return f"Jugador ({tj}) pidió carta: repartirle una"
            return f"Jugador ({tj}): pedir carta o plantarse"
        if self.etapa == "CASA":
            if len(self.casa) < 2:
                return "Casa: dar vuelta la carta oculta"
            return f"Casa ({tc}): {'debe pedir carta' if self._casa_debe_pedir() else 'se planta'}"
        ganador, detalle = self.resultado
        texto = {"JUGADOR": "Gana el Jugador", "BLACKJACK": "¡Blackjack! Gana el Jugador",
                 "CASA": "Gana la Casa", "EMPATE": "Empate"}[ganador]
        return f"{texto} ({detalle}). Levantar las cartas para otra mano"
