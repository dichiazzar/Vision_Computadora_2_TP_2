import math

class Corner:
    def __init__(self, coordenadas, rank):
        self.sup_izq = (coordenadas[0], coordenadas[1])
        self.inf_der = (coordenadas[2], coordenadas[3])
        self.rank = rank
        self.centroid = (
            (coordenadas[0] + coordenadas[2]) / 2,
            (coordenadas[1] + coordenadas[3]) / 2
        )

    def __eq__(self, other):
        return self.sup_izq == other.sup_izq and self.inf_der == other.inf_der

    def longitud_diagonal(self, other):
        res = 0 
        dx = other.centroid[0] - self.centroid[0]
        dy = other.centroid[1] - self.centroid[1]
    
        distancia = math.sqrt((dx)**2 + (dy)**2)

        if distancia > 235 and distancia < 270: 
            res = 1
        return [res, distancia]

    def angulo_diagonal(self, other):
        res = 0
        dx = other.centroid[0] - self.centroid[0]
        dy = other.centroid[1] - self.centroid[1]
        angulo_rad = math.atan2(dy,dx)

        angulo = math.degrees(angulo_rad)

        if angulo < 0:
            angulo = angulo + 180
        if angulo > 40 and angulo < 70:
            res = 1
        return [res, angulo]


