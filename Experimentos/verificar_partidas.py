"""Registro de partidas reales: prueba de regresión del sistema completo con videos de juegos de verdad.

Cada partida del registro (Experimentos/partidas_reales/registro.yaml) es un video CRUDO grabado con
--guardar-crudo más su resultado esperado (verificado contra lo que se vio en vivo): ganador, cartas finales de cada
lado, puntajes, gestos del Jugador e irregularidades. El script reproduce cada video cuadro a cuadro por la cadena
completa (detector -> seguimiento -> reglas, con el reconocedor de gestos) y verifica que dé exactamente lo mismo.
Si después de cambiar algo (detector, pesos, emparejamiento, seguimiento, reglas, gestos) alguna partida ya no da
lo mismo, el script lo marca y termina con error.

Uso (desde la carpeta del repo):
    python Experimentos/verificar_partidas.py                       # verifica todas (~1-2 min por partida en CPU)
    python Experimentos/verificar_partidas.py --solo crudo_12       # una sola
    python Experimentos/verificar_partidas.py --agregar videos/crudo_13.mp4 --ganador JUGADOR \\
        --descripcion "pide con 12 y gana 19 a 18"
      (reproduce el video, verifica que el ganador sea el indicado, lo copia al registro y guarda el resultado)
"""

import argparse
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import yaml

REPO = Path(__file__).resolve().parent.parent
CARPETA = REPO / "Experimentos" / "partidas_reales"
REGISTRO = CARPETA / "registro.yaml"
sys.path.insert(0, str(REPO / "Identificador_Cartas"))
sys.path.insert(0, str(REPO / "Gestos"))

GANADORES = ["JUGADOR", "BLACKJACK", "CASA", "EMPATE"]


def gesto_corto(texto):
    if "pide carta" in texto:
        return "PEDIR"
    if "llega a 21" in texto:
        return "PLANTA_21"
    if "se planta" in texto:
        return "PLANTARSE"
    return texto


def reproducir(video, pipeline="A", conf=0.3, pesos="auto"):
    """Pasa el video por la cadena completa y devuelve el resumen de la partida."""
    from blackjack import puntaje
    from detectores import crear_detector
    from gestos import DetectorDeManos, ReconocedorDeGestos
    from juego import Partida

    partida = Partida(crear_detector(pipeline, conf=conf, pesos=pesos),
                      gestos=ReconocedorDeGestos(DetectorDeManos(num_manos=2, modo="video")))
    cap = cv2.VideoCapture(str(video))
    cuadros = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        partida.procesar(frame)
        cuadros += 1
    cap.release()
    j = partida.juego
    ganadores = [g for g in GANADORES for _ in range(j.historial[g])]
    return {
        "cuadros": cuadros,
        "ganador": ganadores[0] if len(ganadores) == 1 else (ganadores or "SIN TERMINAR"),
        "jugador": sorted(j.jugador),
        "casa": sorted(j.casa),
        "puntaje_jugador": puntaje(j.jugador)[0],
        "puntaje_casa": puntaje(j.casa)[0],
        "gestos": [gesto_corto(e.texto) for e in j.eventos if e.tipo == "gesto"],
        "irregularidades": [e.texto for e in j.eventos if e.tipo == "irregularidad"],
        "eventos": [f"[{e.cuadro}] {e.tipo}: {e.texto}" for e in j.eventos],
    }


CAMPOS = ["ganador", "jugador", "casa", "puntaje_jugador", "puntaje_casa", "gestos", "irregularidades"]


def comparar(esperado, obtenido):
    """Lista de diferencias (campo, esperado, obtenido)."""
    return [(c, esperado[c], obtenido[c]) for c in CAMPOS if esperado.get(c) != obtenido[c]]


def cargar():
    if not REGISTRO.exists():
        return {"partidas": []}
    return yaml.safe_load(REGISTRO.read_text(encoding="utf-8")) or {"partidas": []}


def guardar(reg):
    CARPETA.mkdir(parents=True, exist_ok=True)
    REGISTRO.write_text("# Partidas reales para la prueba de regresión (ver Experimentos/verificar_partidas.py).\n"
                        "# El resultado esperado se verificó contra el panel que se vio en vivo.\n"
                        + yaml.safe_dump(reg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def agregar(args):
    origen = Path(args.agregar)
    if not origen.exists():
        sys.exit(f"No existe {origen}")
    reg = cargar()
    nombre = args.nombre or origen.stem
    if any(p["nombre"] == nombre for p in reg["partidas"]):
        sys.exit(f"Ya hay una partida llamada {nombre} en el registro")
    print(f"reproduciendo {origen} ...", flush=True)
    r = reproducir(origen, args.pipeline, args.conf, args.pesos)
    if r["ganador"] != args.ganador:
        print("\n".join(r["eventos"]))
        sys.exit(f"El sistema da ganador {r['ganador']} y se indicó {args.ganador}: no se agrega. "
                 "Revisar la partida (puede ser un error del sistema a corregir).")
    CARPETA.mkdir(parents=True, exist_ok=True)
    destino = CARPETA / f"{nombre}.mp4"
    shutil.copy2(origen, destino)
    reg["partidas"].append({"nombre": nombre, "video": destino.name, "descripcion": args.descripcion or "",
                            "esperado": {c: r[c] for c in CAMPOS}})
    guardar(reg)
    print(f"agregada {nombre}: {r['ganador']} | jugador {r['jugador']} ({r['puntaje_jugador']}) | "
          f"casa {r['casa']} ({r['puntaje_casa']}) | gestos {r['gestos']}")


def verificar(args):
    reg = cargar()
    partidas = [p for p in reg["partidas"] if not args.solo or p["nombre"] in args.solo]
    if not partidas:
        sys.exit("No hay partidas para verificar (agregar con --agregar)")
    fallas = 0
    for p in partidas:
        t0 = time.time()
        r = reproducir(CARPETA / p["video"], args.pipeline, args.conf, args.pesos)
        dif = comparar(p["esperado"], r)
        fallas += bool(dif)
        estado = "OK   " if not dif else "FALLA"
        print(f"{estado} {p['nombre']:10s} {r['ganador']:9s} jugador {r['jugador']} ({r['puntaje_jugador']}) | "
              f"casa {r['casa']} ({r['puntaje_casa']}) | {time.time() - t0:.0f} s  {p.get('descripcion', '')}",
              flush=True)
        for campo, esp, obt in dif:
            print(f"      {campo}: esperado {esp} | obtenido {obt}")
        if dif and args.detalle:
            print("      eventos:\n        " + "\n        ".join(r["eventos"]))
    print(f"\n{len(partidas) - fallas}/{len(partidas)} partidas dan el resultado esperado")
    sys.exit(1 if fallas else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agregar", help="video crudo de una partida nueva para sumar al registro")
    ap.add_argument("--ganador", choices=GANADORES, help="con --agregar: quién ganó en la partida real")
    ap.add_argument("--nombre", help="con --agregar: nombre en el registro (por defecto, el del archivo)")
    ap.add_argument("--descripcion", help="con --agregar: breve descripción de la mano")
    ap.add_argument("--solo", nargs="+", help="verificar sólo estas partidas")
    ap.add_argument("--detalle", action="store_true", help="mostrar los eventos de las partidas que fallan")
    ap.add_argument("--pipeline", default="A")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--pesos", default="auto")
    args = ap.parse_args()
    if args.agregar:
        if not args.ganador:
            sys.exit("con --agregar hay que indicar --ganador (lo que pasó en la partida real)")
        agregar(args)
    else:
        verificar(args)


if __name__ == "__main__":
    main()
