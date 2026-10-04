"""Clases del proyecto y normalización de los nombres de otros datasets.

Los datasets externos nombran las clases de muchas formas: con palo ("10h", "AS", "QC"), en minúsculas,
o con palabras ("ace", "king of hearts"). Para Blackjack sólo importa el valor, así que todo se lleva
a las 13 clases de YOLO-13 / CNN-13.
"""

import re

CLASES = ["10", "2", "3", "4", "5", "6", "7", "8", "9", "A", "J", "K", "Q"]   # orden de YOLO-13 / CNN-13

_PALABRAS = {"ACE": "A", "JACK": "J", "QUEEN": "Q", "KING": "K", "TWO": "2", "THREE": "3", "FOUR": "4",
             "FIVE": "5", "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9", "TEN": "10"}


def normalizar_clase(nombre):
    """'10h' -> '10', 'AS' -> 'A', 'king of hearts' -> 'K', '7' -> '7'. Error si no se reconoce."""
    n = str(nombre).strip().upper()
    if n in CLASES:
        return n
    primera = re.split(r"[\s_\-]+", n)[0]
    if primera in _PALABRAS:
        return _PALABRAS[primera]
    sin_palo = re.sub(r"(H|S|D|C|HEARTS?|SPADES?|DIAMONDS?|CLUBS?)$", "", n)
    if sin_palo in CLASES:
        return sin_palo
    raise ValueError(f"No se reconoce la clase {nombre!r} como un valor de carta")
