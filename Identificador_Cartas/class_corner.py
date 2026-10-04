import math

# ============================================================
# GEOMETRÍA DE UNA CARTA (calibrada con las fotos de prueba)
# ============================================================
# Las dos esquinas con índice de una misma carta están en vértices opuestos (diagonal).
# En lugar de umbrales en píxeles (que dependen de la resolución y de la altura de la cámara)
# se usan medidas RELATIVAS al tamaño de la esquina detectada:
#
#   ratio = distancia entre centroides / tamaño de la esquina (raíz del área de la bbox)
#
# En las fotos de prueba los pares reales dan ratio 5.5 - 6.1, para cualquier distancia a la cámara.
RATIO_ESPERADO = 5.8
RATIO_MIN, RATIO_MAX = 4.9, 7.0

# Ángulo de la diagonal (0-180°, medido desde la horizontal, eje y hacia abajo):
#  - carta vertical (índice más alto que ancho): diagonal "\"  -> ~50°
#  - carta horizontal (rotada 90°, índice más ancho que alto): diagonal "/" -> ~140°
ANGULO_VERTICAL = 50.0
ANGULO_HORIZONTAL = 140.0
TOLERANCIA_ANGULO = 22.0

# Dos esquinas con valores DISTINTOS sólo se unen si la geometría es casi perfecta (costo bajo).
# En las fotos de prueba los pares reales de cartas derechas tienen costo ~0.1-0.3.
COSTO_MAX_VALOR_DISTINTO = 0.5


class Corner:
    def __init__(self, coordenadas, rank, conf=1.0, track_id=None):
        self.track_id = track_id      # id de ByteTrack (None si no hay seguimiento)
        self.rank_detectado = rank    # valor de este cuadro; rank puede cambiar por la votación del track
        self.sup_izq = (coordenadas[0], coordenadas[1])
        self.inf_der = (coordenadas[2], coordenadas[3])
        self.rank = rank
        self.conf = conf
        self.centroid = (
            (coordenadas[0] + coordenadas[2]) / 2,
            (coordenadas[1] + coordenadas[3]) / 2
        )
        self.ancho = coordenadas[2] - coordenadas[0]
        self.alto = coordenadas[3] - coordenadas[1]
        self.tamanio = math.sqrt(max(self.ancho * self.alto, 1.0))
        # El índice (valor + palo) es más alto que ancho cuando la carta está vertical
        self.vertical = self.alto >= self.ancho

    def __eq__(self, other):
        return self.sup_izq == other.sup_izq and self.inf_der == other.inf_der

    __hash__ = object.__hash__

    def longitud_diagonal(self, other):
        """Distancia entre centroides, en píxeles y relativa al tamaño medio de las dos esquinas."""
        distancia = math.dist(self.centroid, other.centroid)
        ratio = distancia / ((self.tamanio + other.tamanio) / 2)
        return distancia, ratio

    def angulo_diagonal(self, other):
        """Ángulo de la recta entre centroides, en grados dentro de [0, 180)."""
        dx = other.centroid[0] - self.centroid[0]
        dy = other.centroid[1] - self.centroid[1]
        return math.degrees(math.atan2(dy, dx)) % 180

    def angulo_esperado(self, other):
        """Ángulo de la diagonal esperado según la orientación de la carta.
        Si una esquina parece vertical y la otra horizontal (carta muy girada), se acepta cualquiera."""
        if self.vertical and other.vertical:
            return [ANGULO_VERTICAL]
        if not self.vertical and not other.vertical:
            return [ANGULO_HORIZONTAL]
        return [ANGULO_VERTICAL, ANGULO_HORIZONTAL]

    def score_par(self, other):
        """¿Pueden ser las dos esquinas de la misma carta?

        Devuelve (es_par, costo, distancia, angulo). Cuanto menor el costo, más probable el par.
        """
        distancia, ratio = self.longitud_diagonal(other)
        angulo = self.angulo_diagonal(other)

        desvio_angulo = min(abs(angulo - a) for a in self.angulo_esperado(other))
        ok = RATIO_MIN <= ratio <= RATIO_MAX and desvio_angulo <= TOLERANCIA_ANGULO
        costo = abs(ratio - RATIO_ESPERADO) / (RATIO_MAX - RATIO_MIN) + desvio_angulo / TOLERANCIA_ANGULO

        if self.rank != other.rank:
            # Valores distintos: el detector pudo leer mal una de las dos esquinas (p. ej. la invertida y
            # desenfocada). Si la geometría es casi perfecta se acepta igual como la misma carta, con un costo
            # mayor para que siempre se prefieran los pares del mismo valor. La carta toma el valor de la
            # esquina más confiable (ver Carta).
            ok = ok and costo <= COSTO_MAX_VALOR_DISTINTO
            costo += 1.0
        return ok, costo, distancia, angulo
