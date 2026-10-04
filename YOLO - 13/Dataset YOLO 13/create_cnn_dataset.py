from pathlib import Path
from PIL import Image
import yaml
from collections import defaultdict

# ============================================================
# CONFIGURACIÓN
# ============================================================

# El script debe estar dentro de la carpeta del dataset YOLO
YOLO_DATASET = Path(__file__).resolve().parent

# Dataset nuevo: <repo>/CNN - 13/Dataset CNN-13 (donde lo usa CNN - 13/train_cnn_13.py)
CNN_DATASET = YOLO_DATASET.parent.parent / "CNN - 13" / "Dataset CNN-13"

# Padding alrededor de cada bounding box.
# 0.10 = agregar 10% del ancho/alto de la bbox a cada lado.
PADDING = 0.10

SPLITS = ["train", "valid", "test"]


# ============================================================
# LEER CLASES DESDE data.yaml
# ============================================================

yaml_path = YOLO_DATASET / "data.yaml"

with open(yaml_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

class_names = data["names"]

print("Clases encontradas:")
for class_id, class_name in enumerate(class_names):
    print(f"  {class_id:2d} -> {class_name}")


# ============================================================
# CREAR CARPETAS
# ============================================================

for split in SPLITS:
    for class_name in class_names:
        (CNN_DATASET / split / str(class_name)).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# ESTADÍSTICAS
# ============================================================

counts = defaultdict(lambda: defaultdict(int))
skipped = 0


# ============================================================
# PROCESAR DATASET
# ============================================================

image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

for split in SPLITS:

    images_dir = YOLO_DATASET / split / "images"
    labels_dir = YOLO_DATASET / split / "labels"

    print(f"\nProcesando {split}...")

    if not images_dir.exists():
        raise FileNotFoundError(
            f"No encontré la carpeta: {images_dir}"
        )

    if not labels_dir.exists():
        raise FileNotFoundError(
            f"No encontré la carpeta: {labels_dir}"
        )

    images = [
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ]

    print(f"Imágenes encontradas: {len(images)}")

    for image_index, image_path in enumerate(images):

        label_path = labels_dir / f"{image_path.stem}.txt"

        # Puede haber imágenes sin anotaciones
        if not label_path.exists():
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error abriendo {image_path.name}: {e}")
            skipped += 1
            continue

        img_width, img_height = image.size

        with open(label_path, "r", encoding="utf-8") as f:
            annotations = f.readlines()

        for bbox_index, annotation in enumerate(annotations):

            parts = annotation.strip().split()

            if len(parts) < 5:
                skipped += 1
                continue

            class_id = int(float(parts[0]))

            x_center = float(parts[1])
            y_center = float(parts[2])
            bbox_width = float(parts[3])
            bbox_height = float(parts[4])

            if class_id < 0 or class_id >= len(class_names):
                skipped += 1
                continue

            class_name = str(class_names[class_id])

            # ------------------------------------------------
            # YOLO normalizado -> píxeles
            # ------------------------------------------------

            x_center_px = x_center * img_width
            y_center_px = y_center * img_height

            bbox_width_px = bbox_width * img_width
            bbox_height_px = bbox_height * img_height

            x1 = x_center_px - bbox_width_px / 2
            y1 = y_center_px - bbox_height_px / 2
            x2 = x_center_px + bbox_width_px / 2
            y2 = y_center_px + bbox_height_px / 2

            # ------------------------------------------------
            # PADDING
            # ------------------------------------------------

            pad_x = bbox_width_px * PADDING
            pad_y = bbox_height_px * PADDING

            x1 -= pad_x
            y1 -= pad_y
            x2 += pad_x
            y2 += pad_y

            # Limitar al tamaño de la imagen
            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = min(img_width, int(round(x2)))
            y2 = min(img_height, int(round(y2)))

            if x2 <= x1 or y2 <= y1:
                skipped += 1
                continue

            # ------------------------------------------------
            # CROP
            # ------------------------------------------------

            crop = image.crop((x1, y1, x2, y2))

            # Nombre único y trazable
            output_name = (
                f"{image_path.stem}"
                f"_bbox{bbox_index:02d}"
                f".jpg"
            )

            output_path = (
                CNN_DATASET
                / split
                / class_name
                / output_name
            )

            crop.save(
                output_path,
                "JPEG",
                quality=95
            )

            counts[split][class_name] += 1

        if (image_index + 1) % 500 == 0:
            print(
                f"  {image_index + 1}/{len(images)} imágenes procesadas"
            )


# ============================================================
# RESULTADOS
# ============================================================

print("\n" + "=" * 65)
print("DATASET CNN-13 GENERADO")
print("=" * 65)

for split in SPLITS:

    total = sum(counts[split].values())

    print(f"\n{split.upper()}")
    print("-" * 30)

    for class_name in class_names:
        n = counts[split][str(class_name)]
        print(f"{class_name:>3}: {n:5d}")

    print(f"{'TOTAL':>3}: {total:5d}")

print("\nCrops descartados:", skipped)
print("\nDataset guardado en:")
print(CNN_DATASET.resolve())