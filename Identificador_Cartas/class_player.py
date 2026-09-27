from class_corner import Corner
from class_carta import Carta
from collections import defaultdict

class Entidad:
    def __init__(self, nombre):
        self.nombre = nombre
        self.esquinas = defaultdict(list)
        self.cartas = []
        self.puntaje = 0
        self.diagonales = []
        self.estado = 0. #0 es no excedido, 1 es excedido

    def agregar_esquina(self, esquina):
        c = 0
        candidatas = self.esquinas[esquina.rank]
        if len(candidatas) >= 1:
            c = self.determinar_carta(esquina, candidatas)
        if c == 0:
            #self.puntaje = self.puntaje + esquina.rank
            candidatas.append(esquina)

    def determinar_carta(self, esquina, esquinas_candidatas):
        for par in esquinas_candidatas:
            l, distanica = esquina.longitud_diagonal(par)
            a, angulo = esquina.angulo_diagonal(par)
            self.diagonales.append(
            (esquina, par, distanica, angulo))
            print(
                f"{self.nombre} | "
                f"Rank: {esquina.rank} | "
                f"Distancia: {distanica:.2f}px | "
                f"Ángulo: {angulo:.2f}°")
            score = l + a
            print(score)
            if score > 1:
                carta = Carta(esquina, par, esquina.rank)
                self.cartas.append(carta)
                esquinas_candidatas.remove(par)
                return True
        return False

    def determinar_estado(self):
        if self.puntaje > 21:
            self.estado = 1


