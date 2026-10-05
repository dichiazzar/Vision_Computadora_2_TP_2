class Carta:
    """Una carta reconstruida a partir de una o dos esquinas con el mismo valor.

    Con dos esquinas la bbox de la carta es la unión de ambas. Con una sola (la otra está tapada
    o fuera de cuadro) la carta igual cuenta para la mano, y su bbox es la de esa esquina.
    """

    def __init__(self, corner1, corner2=None, rank=None):

        self.esquinas = [corner1] if corner2 is None else [corner1, corner2]
        # si las dos esquinas no coinciden en el valor, manda la más confiable
        self.rank = rank if rank is not None else max(self.esquinas, key=lambda e: e.conf).rank
        self.conf = min(e.conf for e in self.esquinas)

        self.sup_izq = (
            min(e.sup_izq[0] for e in self.esquinas),
            min(e.sup_izq[1] for e in self.esquinas)
        )

        self.inf_der = (
            max(e.inf_der[0] for e in self.esquinas),
            max(e.inf_der[1] for e in self.esquinas)
        )

        self.centro = (
            (self.sup_izq[0] + self.inf_der[0]) / 2,
            (self.sup_izq[1] + self.inf_der[1]) / 2
        )

    @property
    def completa(self):
        return len(self.esquinas) == 2


def _iou(a, b):
    ix = max(0.0, min(a.inf_der[0], b.inf_der[0]) - max(a.sup_izq[0], b.sup_izq[0]))
    iy = max(0.0, min(a.inf_der[1], b.inf_der[1]) - max(a.sup_izq[1], b.sup_izq[1]))
    inter = ix * iy
    return inter / max(a.ancho * a.alto + b.ancho * b.alto - inter, 1e-6)


def quitar_duplicadas(esquinas, iou_min=0.3):
    """Dos esquinas del MISMO valor que se superponen son la misma esquina vista dos veces (p. ej. una copia
    recordada por el seguimiento y la detección nueva, algo corrida porque una mano la tapaba): queda la más
    confiable. Dos esquinas reales nunca se superponen así (los índices de dos cartas no ocupan el mismo lugar)."""
    quedan = []
    for e in sorted(esquinas, key=lambda x: -x.conf):
        if not any(e.rank == q.rank and _iou(e, q) > iou_min for q in quedan):
            quedan.append(e)
    return quedan


def _dentro_de(punto, carta, margen=0.15):
    """¿El punto cae dentro de la carta (su caja achicada un `margen` por lado)?"""
    (x1, y1), (x2, y2) = carta.sup_izq, carta.inf_der
    mx, my = (x2 - x1) * margen, (y2 - y1) * margen
    return x1 + mx <= punto[0] <= x2 - mx and y1 + my <= punto[1] <= y2 - my


def emparejar_esquinas(esquinas):
    """Agrupa las esquinas detectadas en cartas.

    0. Se quitan las esquinas duplicadas (mismo valor, superpuestas).

    1. Se evalúan TODOS los pares posibles (no sólo el primero que aparece); los de distinto valor sólo
       con geometría casi perfecta y siempre detrás de los del mismo valor (ver Corner.score_par).
    2. Se aceptan de menor a mayor costo, sin reutilizar esquinas (emparejamiento voraz global).
       Así, con varias cartas del mismo valor cerca, cada esquina se une con su pareja más probable.
    3. Las esquinas que quedan sin pareja se cuentan como cartas de una sola esquina, salvo que caigan DENTRO de
       una carta completa del mismo valor: ahí no puede verse un índice real (lo taparía esa carta), así que es una
       detección falsa sobre la cara de la carta (p. ej. el dibujo de una K medio tapada por una mano).

    Devuelve (cartas, diagonales) donde diagonales son los pares aceptados (para dibujar).
    """
    esquinas = quitar_duplicadas(esquinas)
    candidatos = []
    for i in range(len(esquinas)):
        for j in range(i + 1, len(esquinas)):
            ok, costo, distancia, angulo = esquinas[i].score_par(esquinas[j])
            if ok:
                candidatos.append((costo, i, j, distancia, angulo))

    usadas, cartas, diagonales = set(), [], []
    for costo, i, j, distancia, angulo in sorted(candidatos):
        if i in usadas or j in usadas:
            continue
        usadas.update((i, j))
        cartas.append(Carta(esquinas[i], esquinas[j]))
        diagonales.append((esquinas[i], esquinas[j], distancia, angulo))

    completas = list(cartas)
    for k, esquina in enumerate(esquinas):
        if k in usadas:
            continue
        if any(c.rank == esquina.rank and _dentro_de(esquina.centroid, c) for c in completas):
            continue                     # índice "dentro" de una carta del mismo valor: detección falsa
        cartas.append(Carta(esquina))

    return cartas, diagonales
