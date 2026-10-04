"""Grabación de video compatible con WhatsApp, PowerPoint, navegadores y celulares (H.264).

OpenCV graba .mp4 con el códec "mp4v" (MPEG-4 Part 2), que muchas apps rechazan ("file not supported"), y
H.264 directo desde OpenCV depende de librerías que no siempre están (OpenH264). Por eso se graba en un archivo
temporal y al cerrar se convierte a H.264 (yuv420p, +faststart) con el ffmpeg que trae el paquete imageio-ffmpeg.
Si ffmpeg no está disponible, queda el .mp4 original (mp4v) y se avisa.

Convertir un video ya grabado:
    python Identificador_Cartas/video.py partida.mp4                  # crea partida_h264.mp4
    python Identificador_Cartas/video.py partida.mp4 --reemplazar     # lo reemplaza
    python Identificador_Cartas/video.py partida.mp4 --fps 8          # corrige la velocidad (se veía acelerado)

Con la cámara, los cuadros se procesan a la velocidad que da la PC (~5-15 por segundo) y no a una tasa fija; con
tiempo_real=True el grabador mide la duración real y al convertir ajusta la velocidad para que el video dure lo
mismo que la partida.
"""

import argparse
import subprocess
import time
from pathlib import Path

import cv2


def ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def fps_de(ruta):
    cap = cv2.VideoCapture(str(ruta))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def convertir_h264(origen, destino, crf=23, fps=None):
    """Convierte a H.264. Con `fps`, reinterpreta el video a esa velocidad (cambia la duración, no los cuadros).
    Devuelve True si salió bien."""
    exe = ffmpeg()
    if exe is None:
        return False
    tiempo = []
    if fps:
        factor = fps_de(origen) / fps
        tiempo = ["-vf", f"setpts={factor:.6f}*PTS", "-r", f"{fps:.3f}"]
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(origen), *tiempo, "-c:v", "libx264", "-preset", "veryfast",
           "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(destino)]
    return subprocess.run(cmd).returncode == 0 and Path(destino).exists()


class GrabadorVideo:
    """Como cv2.VideoWriter (método write / release), pero el archivo final queda en H.264."""

    def __init__(self, ruta, fps, tamanio, tiempo_real=False):
        self.tiempo_real, self.cuadros, self.t0 = tiempo_real, 0, None
        self.ruta = Path(ruta)
        self.temporal = self.ruta.with_name(self.ruta.stem + "_tmp_mp4v.mp4")
        self.writer = cv2.VideoWriter(str(self.temporal), cv2.VideoWriter_fourcc(*"mp4v"), fps, tamanio)

    def write(self, cuadro):
        if self.t0 is None:
            self.t0 = time.perf_counter()
        self.cuadros += 1
        self.writer.write(cuadro)

    def release(self):
        self.writer.release()
        fps = None
        if self.tiempo_real and self.cuadros > 1:
            fps = (self.cuadros - 1) / max(time.perf_counter() - self.t0, 1e-3)
        if convertir_h264(self.temporal, self.ruta, fps=fps):
            self.temporal.unlink()
            extra = f", {fps:.1f} FPS reales" if fps else ""
            print(f"video guardado (H.264{extra}): {self.ruta}")
        else:
            self.temporal.replace(self.ruta)
            print(f"video guardado en mp4v (no se pudo convertir a H.264: pip install imageio-ffmpeg): {self.ruta}")


def main():
    ap = argparse.ArgumentParser(description="Convierte un video a H.264 (compatible con WhatsApp, etc.)")
    ap.add_argument("video")
    ap.add_argument("--reemplazar", action="store_true", help="reemplazar el archivo original")
    ap.add_argument("--crf", type=int, default=23, help="calidad: menor = mejor y más pesado (18-28)")
    ap.add_argument("--fps", type=float, help="velocidad real a la que se grabó (corrige videos acelerados)")
    args = ap.parse_args()

    origen = Path(args.video)
    destino = origen.with_name(origen.stem + "_h264.mp4")
    if not convertir_h264(origen, destino, args.crf, args.fps):
        raise SystemExit("No se pudo convertir: instalar con  pip install imageio-ffmpeg")
    if args.reemplazar:
        destino.replace(origen)
        destino = origen
    print(f"{destino}  ({destino.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
