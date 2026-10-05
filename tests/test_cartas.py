"""Tests del armado de cartas a partir de esquinas (Identificador_Cartas/class_carta.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Identificador_Cartas"))

from class_carta import emparejar_esquinas, quitar_duplicadas   # noqa: E402
from class_corner import Corner                                  # noqa: E402


def esquina(cx, cy, rank, conf=0.9, w=26, h=60):
    """Esquina vertical (índice más alto que ancho) centrada en (cx, cy)."""
    return Corner([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], rank, conf)


def carta(x, y, rank):
    """Las dos esquinas de una carta vertical: arriba a la izquierda y abajo a la derecha (diagonal ~50°,
    distancia ~5.8 tamaños de esquina, como en las fotos reales)."""
    return [esquina(x, y, rank), esquina(x + 147, y + 175, rank)]


def valores(esquinas):
    return sorted(c.rank for c in emparejar_esquinas(esquinas)[0])


def test_carta_completa():
    assert valores(carta(100, 100, "K")) == ["K"]


def test_dos_cartas_del_mismo_valor_separadas():
    assert valores(carta(100, 100, "K") + carta(400, 100, "K")) == ["K", "K"]


def test_esquina_duplicada_superpuesta_es_una_sola():
    # la misma esquina vista dos veces, algo corrida (p. ej. copia recordada + detección nueva)
    esq = carta(100, 100, "K") + [esquina(106, 104, "K", conf=0.6)]
    assert len(quitar_duplicadas(esq)) == 2
    assert valores(esq) == ["K"]


def test_esquinas_superpuestas_de_distinto_valor_no_se_fusionan_aca():
    # eso lo resuelve el NMS agnóstico del detector; acá sólo se fusionan las del mismo valor
    assert len(quitar_duplicadas([esquina(100, 100, "3"), esquina(102, 101, "5")])) == 2


def test_indice_falso_dentro_de_una_carta_del_mismo_valor_se_descarta():
    # una "K" detectada sobre el dibujo de la propia K (centro de la carta): no es otra carta
    assert valores(carta(100, 100, "K") + [esquina(175, 190, "K", conf=0.5)]) == ["K"]


def test_esquina_suelta_de_otro_valor_dentro_de_una_carta_si_cuenta():
    # un índice de OTRO valor sobre la cara de una carta: puede ser una carta encima (superpuesta)
    assert valores(carta(100, 100, "K") + [esquina(175, 190, "7")]) == ["7", "K"]


def test_esquina_suelta_fuera_de_las_cartas_cuenta_como_carta():
    assert valores(carta(100, 100, "K") + [esquina(600, 120, "K")]) == ["K", "K"]
