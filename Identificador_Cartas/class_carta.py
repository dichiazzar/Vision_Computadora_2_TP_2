class Carta:

    def __init__(self, corner1, corner2, rank):

        self.rank = rank

        self.sup_izq = (
            min(corner1.sup_izq[0], corner2.sup_izq[0]),
            min(corner1.sup_izq[1], corner2.sup_izq[1])
        )

        self.inf_der = (
            max(corner1.inf_der[0], corner2.inf_der[0]),
            max(corner1.inf_der[1], corner2.inf_der[1])
        )