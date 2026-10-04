"""Estabilización temporal para video: votación por track, memoria de esquinas y mediana por carta.

En video, cada cuadro se detecta por separado y el resultado "parpadea": una esquina se confunde un instante
(un 8 que por un cuadro es 6), una esquina se pierde por desenfoque, etc. Se estabiliza en dos niveles:

1. Valor de cada esquina: ByteTrack le da a cada esquina un track_id estable. El valor se decide por
   VOTACIÓN ponderada por confianza sobre los últimos `ventana_voto` cuadros de ese track.
2. Memoria: una esquina que se vio hace menos de `memoria` cuadros y en este cuadro no se detectó
   (desenfoque, reflejo) se mantiene en su última posición. Una carta retirada desaparece a los ~0.7 s.
3. Mano mostrada: cada carta se decide por la mediana de los últimos `ventana_mano` cuadros (ver estabilizar).
"""

from collections import Counter, defaultdict, deque
from copy import copy
from types import SimpleNamespace

from class_player import Entidad


def _cerca(e1, e2, factor=1.5):
    """¿Están dos esquinas a menos de `factor` veces su tamaño? Dos esquinas de cartas distintas están
    siempre mucho más lejos (la diagonal de una carta mide ~6 tamaños de esquina)."""
    dx, dy = e1.centroid[0] - e2.centroid[0], e1.centroid[1] - e2.centroid[1]
    return (dx * dx + dy * dy) ** 0.5 < factor * max(e1.tamanio, e2.tamanio)


class Seguimiento:
    def __init__(self, ventana_voto=15, ventana_mano=15, memoria=10):
        self.ventana_voto = ventana_voto
        self.memoria = memoria
        self.ultima = {}                                               # track_id -> última esquina vista
        self.votos = defaultdict(lambda: deque(maxlen=ventana_voto))   # track_id -> [(valor, conf), ...]
        self.visto = {}                                                # track_id -> último cuadro visto
        self.manos = deque(maxlen=ventana_mano)
        self.cuadro = 0
        self.tracks_totales = set()

    def votar(self, esquinas):
        """Reemplaza el valor de cada esquina seguida por el más votado de su track y agrega las esquinas
        recordadas (vistas hace poco, perdidas en este cuadro). Devuelve la nueva lista de esquinas."""
        self.cuadro += 1
        for e in esquinas:
            if e.track_id is None:          # el tracker todavía no la confirmó: se usa el valor del cuadro
                continue
            self.votos[e.track_id].append((e.rank_detectado, e.conf))
            self.visto[e.track_id] = self.cuadro
            self.tracks_totales.add(e.track_id)
            score = Counter()
            for valor, conf in self.votos[e.track_id]:
                score[valor] += conf
            e.rank = score.most_common(1)[0][0]
            self.ultima[e.track_id] = e

        presentes = {e.track_id for e in esquinas}
        recordadas = []
        for tid, e in self.ultima.items():
            # si ByteTrack le dio otro id a la misma esquina, la copia vieja queda junto a una detección actual
            # (aunque la cámara se haya movido unos píxeles): no se recuerda, si no la carta saldría duplicada
            superpuesta = any(_cerca(e, d) for d in esquinas)
            if tid not in presentes and not superpuesta and self.cuadro - self.visto[tid] <= self.memoria:
                r = copy(e)
                r.recordada = True
                recordadas.append(r)
        # olvidar tracks que ByteTrack ya descartó
        for tid in [t for t, c in self.visto.items() if self.cuadro - c > 4 * self.ventana_voto]:
            del self.visto[tid]
            self.votos.pop(tid, None)
            self.ultima.pop(tid, None)
        return esquinas + recordadas

    @staticmethod
    def clave(casa, jugador):
        return tuple(sorted(c.rank for c in casa.cartas)), tuple(sorted(c.rank for c in jugador.cartas))

    def estabilizar(self, casa, jugador):
        """Agrega la mano de este cuadro y devuelve (casa, jugador, estabilidad) estabilizados.

        Cada carta se decide por separado: para cada lado y cada valor, la cantidad mostrada es la MEDIANA de
        cuántas cartas de ese valor hubo en los últimos cuadros. Así, una esquina perdida en un cuadro y una
        lectura equivocada en otro no se combinan en una mano equivocada (lo que pasaría tomando la mano
        completa más frecuente). estabilidad = fracción de los últimos cuadros iguales a la mano mostrada."""
        self.manos.append(self.clave(casa, jugador))

        def mediana_por_valor(lado):
            conteos = [Counter(mano[lado]) for mano in self.manos]
            valores = sorted(set().union(*conteos))
            resultado = []
            for v in valores:
                n = sorted(c[v] for c in conteos)
                resultado += [v] * n[len(n) // 2]
            return tuple(resultado)

        valores_casa, valores_jugador = mediana_por_valor(0), mediana_por_valor(1)
        n = sum(m == (valores_casa, valores_jugador) for m in self.manos)
        casa_est, jugador_est = Entidad("Casa"), Entidad("Jugador")
        for r in valores_casa:
            casa_est.agregar_carta(SimpleNamespace(rank=r))
        for r in valores_jugador:
            jugador_est.agregar_carta(SimpleNamespace(rank=r))
        return casa_est, jugador_est, n / len(self.manos)
