PUNTOS_BJ = {"A": 1, "J": 10, "Q": 10, "K": 10, **{str(i): i for i in range(2, 11)}}


class Entidad:
    """Un participante de la mano (Jugador o Casa) con sus cartas y su puntaje de Blackjack."""

    def __init__(self, nombre):
        self.nombre = nombre
        self.cartas = []

    def agregar_carta(self, carta):
        self.cartas.append(carta)

    @property
    def puntaje(self):
        """Suma de Blackjack: J/Q/K valen 10 y un As vale 11 si no se pasa de 21 (si no, 1)."""
        total = sum(PUNTOS_BJ[c.rank] for c in self.cartas)
        if any(c.rank == "A" for c in self.cartas) and total + 10 <= 21:
            total += 10
        return total

    @property
    def excedido(self):
        return self.puntaje > 21

    @property
    def blackjack(self):
        return len(self.cartas) == 2 and self.puntaje == 21

    def __str__(self):
        cartas = " ".join(c.rank for c in self.cartas) or "-"
        return f"{self.nombre}: {cartas} = {self.puntaje}"


def determinar_ganador(jugador, casa):
    """Resultado de la mano según las reglas básicas de Blackjack."""
    if not jugador.cartas or not casa.cartas:
        return "Esperando cartas"
    if jugador.excedido:
        return "El Jugador se pasó: gana la Casa"
    if casa.excedido:
        return "La Casa se pasó: gana el Jugador"
    if jugador.blackjack and not casa.blackjack:
        return "Blackjack: gana el Jugador"
    if casa.blackjack and not jugador.blackjack:
        return "Blackjack: gana la Casa"
    if jugador.puntaje > casa.puntaje:
        return "Gana el Jugador"
    if jugador.puntaje < casa.puntaje:
        return "Gana la Casa"
    return "Empate"
