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


def emparejar_esquinas(esquinas):
    """Agrupa las esquinas detectadas en cartas.

    1. Se evalúan TODOS los pares posibles (no sólo el primero que aparece); los de distinto valor sólo
       con geometría casi perfecta y siempre detrás de los del mismo valor (ver Corner.score_par).
    2. Se aceptan de menor a mayor costo, sin reutilizar esquinas (emparejamiento voraz global).
       Así, con varias cartas del mismo valor cerca, cada esquina se une con su pareja más probable.
    3. Las esquinas que quedan sin pareja se cuentan como cartas de una sola esquina.

    Devuelve (cartas, diagonales) donde diagonales son los pares aceptados (para dibujar).
    """
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

    for k, esquina in enumerate(esquinas):
        if k not in usadas:
            cartas.append(Carta(esquina))

    return cartas, diagonales
