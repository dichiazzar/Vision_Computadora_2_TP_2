"""Tests del motor de reglas (Identificador_Cartas/blackjack.py). Correr con:  python -m pytest tests"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Identificador_Cartas"))

from blackjack import Blackjack, Reglas, es_blackjack, puntaje   # noqa: E402

CONF = 3   # cuadros para confirmar en los tests


def nuevo(**reglas):
    return Blackjack(reglas=Reglas(**reglas), confirmar=CONF, cuadros_vacia=5)


def ver(bj, casa, jugador, n=CONF):
    """La cámara ve la misma mesa durante n cuadros."""
    for _ in range(n):
        bj.observar(casa.split(), jugador.split())
    return bj.etapa


def repartir(bj, casa, jugador):
    """Reparto típico: cartas que van apareciendo una a una."""
    j = jugador.split()
    ver(bj, "", j[0])
    ver(bj, casa, j[0])
    return ver(bj, casa, " ".join(j))


# ---------------------------------------------------------------- puntaje
@pytest.mark.parametrize("cartas,esperado", [
    (["A", "K"], (21, True)), (["A", "A"], (12, True)), (["A", "6"], (17, True)),
    (["A", "6", "K"], (17, False)), (["10", "Q", "2"], (22, False)), (["A", "A", "9"], (21, True)),
])
def test_puntaje(cartas, esperado):
    assert puntaje(cartas) == esperado


def test_es_blackjack():
    assert es_blackjack(["A", "J"])
    assert not es_blackjack(["A", "5", "5"])     # 21 con tres cartas no es blackjack


# ---------------------------------------------------------------- flujo de la mano
def test_reparto_llega_al_turno_del_jugador():
    bj = nuevo()
    assert repartir(bj, "9", "10 6") == "JUGADOR"
    assert bj.jugador == ["10", "6"] and bj.casa == ["9"]


def test_jugador_pide_y_se_pasa():
    bj = nuevo()
    repartir(bj, "9", "10 6")
    assert bj.gesto("PEDIR")
    assert ver(bj, "9", "10 6 K") == "FIN"
    assert bj.resultado[0] == "CASA"
    assert bj.saldo == -10


def test_jugador_se_planta_y_la_casa_pide_hasta_17():
    bj = nuevo()
    repartir(bj, "9", "10 8")
    bj.gesto("PLANTARSE")
    assert bj.etapa == "CASA"
    ver(bj, "9 4", "10 8")                     # la Casa da vuelta: 13 -> debe pedir
    assert bj.etapa == "CASA" and "debe pedir" in bj.instruccion()
    assert ver(bj, "9 4 5", "10 8") == "FIN"   # 18 contra 18
    assert bj.resultado[0] == "EMPATE" and bj.saldo == 0


def test_casa_se_pasa():
    bj = nuevo()
    repartir(bj, "6", "10 7")
    bj.gesto("PLANTARSE")
    ver(bj, "6 10", "10 7")
    ver(bj, "6 10 9", "10 7")
    assert bj.resultado[0] == "JUGADOR" and bj.saldo == 10


@pytest.mark.parametrize("pide_17_blando,etapa", [(False, "FIN"), (True, "CASA")])
def test_regla_17_blando(pide_17_blando, etapa):
    bj = nuevo(casa_pide_17_blando=pide_17_blando)
    repartir(bj, "A", "10 9")
    bj.gesto("PLANTARSE")
    assert ver(bj, "A 6", "10 9") == etapa     # A+6 = 17 blando


def test_blackjack_del_jugador_paga_3_a_2():
    bj = nuevo()
    assert repartir(bj, "9", "A K") == "CASA"  # no juega: la Casa sólo da vuelta su carta
    assert ver(bj, "9 7", "A K") == "FIN"
    assert bj.resultado[0] == "BLACKJACK" and bj.saldo == 15


def test_ambos_blackjack_empatan():
    bj = nuevo()
    repartir(bj, "A", "K A")
    ver(bj, "A Q", "K A")
    assert bj.resultado[0] == "EMPATE"


def test_21_se_planta_automaticamente():
    bj = nuevo()
    repartir(bj, "10", "5 6")
    bj.gesto("PEDIR")
    assert ver(bj, "10", "5 6 K") == "CASA"


# ---------------------------------------------------------------- robustez frente a la visión
def test_ruido_de_un_cuadro_no_cambia_nada():
    bj = nuevo()
    repartir(bj, "9", "10 6")
    ver(bj, "9", "10 6 K", n=1)                # un cuadro con una carta fantasma
    ver(bj, "9", "10 6")
    assert bj.jugador == ["10", "6"] and bj.etapa == "JUGADOR"


def test_carta_tapada_se_mantiene():
    bj = nuevo()
    repartir(bj, "9", "10 6")
    ver(bj, "9", "10")                         # una mano tapa el 6
    assert bj.jugador == ["10", "6"] and bj.etapa == "JUGADOR"


def test_correccion_de_lectura():
    bj = nuevo()
    repartir(bj, "9", "10 6")
    ver(bj, "9", "10 8")                       # el "6" era un "8" mal leído al principio
    assert sorted(bj.jugador) == ["10", "8"]


def test_carta_no_pedida_es_irregularidad_pero_se_acepta():
    bj = nuevo()
    repartir(bj, "9", "10 2")
    ver(bj, "9", "10 2 3")
    assert bj.jugador == ["10", "2", "3"]
    assert any(e.tipo == "irregularidad" and "sin que la pidiera" in e.texto for e in bj.eventos)


def test_gesto_fuera_de_turno_se_ignora():
    bj = nuevo()
    assert not bj.gesto("PEDIR")               # todavía no se repartió
    repartir(bj, "9", "10 8")
    bj.gesto("PLANTARSE")
    assert not bj.gesto("PEDIR")               # ya es turno de la Casa


def test_mesa_vacia_inicia_otra_mano_y_mantiene_el_saldo():
    bj = nuevo()
    repartir(bj, "9", "10 6")
    bj.gesto("PEDIR")
    ver(bj, "9", "10 6 K")                     # se pasa: -10
    ver(bj, "", "", n=5)                       # se levantan las cartas
    assert bj.etapa == "ESPERANDO" and bj.saldo == -10
    repartir(bj, "6", "10 10")
    bj.gesto("PLANTARSE")
    ver(bj, "6 10", "10 10")
    ver(bj, "6 10 K", "10 10")                 # la Casa se pasa: +10
    assert bj.saldo == 0 and bj.historial["CASA"] == 1 and bj.historial["JUGADOR"] == 1


def test_gestos_ignorados_mientras_espera_la_carta():
    bj = nuevo()
    repartir(bj, "9", "10 2")
    assert bj.gesto("PEDIR")
    assert not bj.gesto("PLANTARSE")           # la mano del crupier repartiendo no cuenta como gesto
    assert not bj.gesto("PEDIR")
    ver(bj, "9", "10 2 5")                     # llega la carta: vuelve a aceptar gestos
    assert bj.gesto("PLANTARSE") and bj.etapa == "CASA"
