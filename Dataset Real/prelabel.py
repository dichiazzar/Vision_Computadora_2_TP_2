"""Pre-etiquetado de las fotos reales con YOLO-13, para corregirlas en Roboflow.

Lee las fotos de  Dataset Real/fotos/{train,valid,test}/  y genera  Dataset Real/para_roboflow/{train,valid,test}/
con cada imagen (orientación EXIF aplicada y lado mayor <= MAX_LADO) y su .txt en formato YOLO con las
esquinas que detectó YOLO-13. Las clases tienen el MISMO orden que YOLO-13 / CNN-13.

IMPORTANTE: las etiquetas son una propuesta del modelo. Hay que revisar TODAS las cajas en Roboflow
(agregar las que faltan, borrar las falsas, corregir valores). Si no, el test real quedaría sesgado a favor
del pipeline A.

Uso:
    python "Dataset Real/prelabel.py"
    python "Dataset Real/prelabel.py" --conf 0.3
"""

import argparse
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PESOS_YOLO_13 = REPO / "YOLO - 13" / "runs" / "detect" / "blackjack-cv" / "yolo13-baseline-3" / "weights" / "best.pt"
SPLITS = ["train", "valid", "test"]
EXTENSIONES = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
MAX_LADO = 1600   # las fotos del celular (4000 px) se achican; las esquinas siguen midiendo > 25 px


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fotos", default=str(HERE / "fotos"))
    p.add_argument("--salida", default=str(HERE / "para_roboflow"))
    p.add_argument("--conf", type=float, default=0.4)
    args = p.parse_args()

    modelo = YOLO(str(PESOS_YOLO_13))
    nombres = [modelo.names[i] for i in range(len(modelo.names))]
    fotos, salida = Path(args.fotos), Path(args.salida)

    total_img, total_cajas = 0, 0
    for split in SPLITS:
        imagenes = sorted(f for f in (fotos / split).glob("*") if f.suffix.lower() in EXTENSIONES) \
            if (fotos / split).exists() else []
        if not imagenes:
            print(f"{split}: sin fotos en {fotos / split}")
            continue
        (salida / split / "images").mkdir(parents=True, exist_ok=True)
        (salida / split / "labels").mkdir(parents=True, exist_ok=True)

        for f in imagenes:
            img = cv2.imread(str(f))          # cv2 aplica la rotación EXIF de las fotos del celular
            if img is None:
                print("  no se pudo leer", f.name, "(si es .heic, exportarla como .jpg)")
                continue
            h, w = img.shape[:2]
            escala = min(1.0, MAX_LADO / max(h, w))
            if escala < 1:
                img = cv2.resize(img, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
                h, w = img.shape[:2]

            r = modelo.predict(img, conf=args.conf, verbose=False)[0]
            filas = []
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                filas.append(f"{int(c)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                             f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")

            nombre = f"{split}_{f.stem}".replace(" ", "_")
            cv2.imwrite(str(salida / split / "images" / f"{nombre}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (salida / split / "labels" / f"{nombre}.txt").write_text("\n".join(filas))
            total_img, total_cajas = total_img + 1, total_cajas + len(filas)
        print(f"{split}: {len(imagenes)} fotos")

    # data.yaml para que Roboflow reconozca las clases con el mismo orden que YOLO-13
    (salida / "data.yaml").write_text(yaml.safe_dump(
        {"train": "train/images", "val": "valid/images", "test": "test/images",
         "nc": len(nombres), "names": nombres}, sort_keys=False))
    print(f"\n{total_img} imágenes, {total_cajas} esquinas pre-etiquetadas -> {salida}")
    print("Subir cada carpeta de split a Roboflow y REVISAR todas las cajas.")


if __name__ == "__main__":
    main()
