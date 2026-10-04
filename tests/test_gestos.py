"""Tests de la clasificación de poses y de la lógica temporal de los gestos (Gestos/gestos.py).
Usan manos sintéticas de 21 puntos (no hace falta MediaPipe ni cámara)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Gestos"))

from gestos import ReconocedorDeGestos, clasificar_pose   # noqa: E402


def mano(extendidos, escala=100.0, angulo=0.0, centro=(300.0, 500.0)):
    """Mano esquemática: muñeca en el origen, nudillos a 1 unidad, puntas a 2 (extendido) o 0.8 (doblado).
    extendidos = (indice, medio, anular, menique)."""
    p = np.zeros((21, 2))
    p[1:5] = [[-0.3, -0.2], [-0.5, -0.4], [-0.6, -0.6], [-0.7, -0.8]]           # pulgar
    for k, (x, ext) in enumerate(zip([-0.3, -0.1, 0.1, 0.3], extendidos)):
        base = 5 + 4 * k
        largo = 2.0 if ext else 0.8
        p[base:base + 4] = [[x, -1.0], [x, -1.0 - (largo - 1) / 3 if ext else -1.2],
                            [x, -1.0 - 2 * (largo - 1) / 3 if ext else -1.1], [x, -largo]]
    a = np.radians(angulo)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return p @ R.T * escala + np.array(centro)


APUNTAR = (True, False, False, False)
PALMA = (True, True, True, True)
PUNO = (False, False, False, False)
PAZ = (True, True, False, False)


@pytest.mark.parametrize("angulo", [0, 37, 90, 180, 270])
@pytest.mark.parametrize("escala", [40, 150])
def test_pose_invariante_a_rotacion_y_escala(angulo, escala):
    assert clasificar_pose(mano(APUNTAR, escala, angulo)) == "APUNTAR"
    assert clasificar_pose(mano(PALMA, escala, angulo)) == "PALMA"
    assert clasificar_pose(mano(PUNO, escala, angulo)) == "OTRA"
    assert clasificar_pose(mano(PAZ, escala, angulo)) == "OTRA"


def correr(rec, cuadros):
    """cuadros: lista de manos (o None). Devuelve los gestos disparados."""
    gestos = []
    for m in cuadros:
        manos = [] if m is None else [(m, clasificar_pose(m))]
        g = rec.actualizar(manos)
        if g:
            gestos.append(g)
    return gestos


def test_apuntar_sostenido_pide_una_sola_vez():
    rec = ReconocedorDeGestos(cuadros_min=6, espera=20)
    assert correr(rec, [mano(APUNTAR)] * 15) == ["PEDIR"]


def test_apuntar_breve_no_dispara():
    rec = ReconocedorDeGestos(cuadros_min=6)
    assert correr(rec, [mano(APUNTAR)] * 4 + [None] * 3 + [mano(APUNTAR)] * 4) == []


def test_palma_quieta_con_solo_barrido_no_se_planta():
    rec = ReconocedorDeGestos(plantarse="barrido")
    quieta = [mano(PALMA, centro=(300 + np.sin(k), 500)) for k in range(40)]   # temblor de 1 px
    assert correr(rec, quieta) == []


def test_palma_sostenida_se_planta_por_defecto():
    rec = ReconocedorDeGestos(cuadros_palma=12)
    assert correr(rec, [mano(PALMA)] * 11) == []                               # todavía no
    rec = ReconocedorDeGestos(cuadros_palma=12)
    assert correr(rec, [mano(PALMA)] * 20) == ["PLANTARSE"]                    # una sola vez


def test_palma_breve_no_se_planta():
    rec = ReconocedorDeGestos(cuadros_palma=12)
    assert correr(rec, [mano(PALMA)] * 8 + [None] * 5 + [mano(PALMA)] * 8) == []


def test_barrido_con_la_palma_se_planta():
    rec = ReconocedorDeGestos(barrido=1.0)
    barrido = [mano(PALMA, centro=(300 + 15 * k, 500)) for k in range(12)]     # 1.8 palmas en 12 cuadros
    assert correr(rec, barrido) == ["PLANTARSE"]


def test_barrido_con_el_puno_no_hace_nada():
    rec = ReconocedorDeGestos()
    assert correr(rec, [mano(PUNO, centro=(300 + 15 * k, 500)) for k in range(12)]) == []


def test_solo_estatico_no_acepta_barrido_rapido():
    rec = ReconocedorDeGestos(plantarse="estatico", cuadros_palma=12)
    assert correr(rec, [mano(PALMA, centro=(300 + 15 * k, 500)) for k in range(8)]) == []


def test_espera_entre_gestos():
    rec = ReconocedorDeGestos(cuadros_min=6, espera=20)
    pedir = [mano(APUNTAR)] * 8
    pausa = [None] * 5
    assert correr(rec, pedir + pausa + pedir) == ["PEDIR"]                    # el segundo cae en la espera
    rec = ReconocedorDeGestos(cuadros_min=6, espera=20)
    assert correr(rec, pedir + [None] * 25 + pedir) == ["PEDIR", "PEDIR"]


def test_mantener_la_pose_no_repite_el_gesto():
    rec = ReconocedorDeGestos(cuadros_palma=12, espera=5)
    assert correr(rec, [mano(PALMA)] * 60) == ["PLANTARSE"]                    # palma abierta 6 s: una vez
    rec = ReconocedorDeGestos(cuadros_palma=12, espera=5)
    assert correr(rec, [mano(PALMA)] * 15 + [None] * 3 + [mano(PALMA)] * 15) == ["PLANTARSE", "PLANTARSE"]
