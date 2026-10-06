"""Experimento de seguimiento: ¿cuánto estabiliza el video el seguimiento (ByteTrack + votación + memoria)?

Para cada foto de prueba se genera un video de cámara en mano (generar_video_prueba.py) y se procesa con los
pipelines A y B, con y sin seguimiento. La mano real es conocida y es la misma en todo el video.
Métricas:
  - mano correcta (%): cuadros en que la mano mostrada (Casa y Jugador) es exactamente la real
  - cambios de mano: cuántas veces cambia la mano mostrada (parpadeo; lo ideal es 0)

Uso (desde la carpeta del repo):
    python Experimentos/evaluar_seguimiento.py
Resultados en Experimentos/resultados/seguimiento_videos.csv
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Identificador_Cartas"))

import identificador_cartas as I   # noqa: E402
from detectores import crear_detector   # noqa: E402

VIDEOS = {  # foto (en Fotos_prueba/) -> mano real "Casa|Jugador"
    "prueba.jpeg": "Q|7 K",
    "ACES.jpeg": "A|A A",
    "aces torcidos.jpeg": "A|A A",
    "ACES_TAPADOS.jpeg": "A|A A",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta-videos", default=str(REPO / "videos"))
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--salida", default=str(REPO / "Experimentos" / "resultados" / "seguimiento_videos.csv"))
    args = ap.parse_args()

    carpeta = Path(args.carpeta_videos)
    filas = []
    for pipeline in ("A", "B"):
        detector = crear_detector(pipeline, conf=args.conf)
        for foto, esperado in VIDEOS.items():
            video = carpeta / (Path(foto).stem.replace(" ", "_") + ".mp4")
            if not video.exists():
                subprocess.run([sys.executable, str(REPO / "Experimentos" / "generar_video_prueba.py"),
                                "--foto", str(REPO / "Fotos_prueba" / foto), "--salida", str(video)], check=True)
            for seguir in (False, True):
                ns = argparse.Namespace(camara=None, video=str(video), guardar=None, sin_ventana=True,
                                        sin_seguimiento=not seguir, ventana=15, memoria=10,
                                        esperado=esperado, max_cuadros=0)
                st = I.modo_video(ns, detector)
                filas.append(dict(pipeline=pipeline, video=video.name, seguimiento=seguir, cuadros=st["cuadros"],
                                  mano_correcta=round(st["ok_mostrada"] / st["cuadros"], 4),
                                  cambios_de_mano=st["cambios_mostrada"]))

    import pandas as pd
    df = pd.DataFrame(filas)
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.salida, index=False)
    print(df.to_string(index=False))
    print("\nPromedio por pipeline y seguimiento:")
    print(df.groupby(["pipeline", "seguimiento"])[["mano_correcta", "cambios_de_mano"]].mean().round(3))
    print("\nResultados en", args.salida)


if __name__ == "__main__":
    main()
