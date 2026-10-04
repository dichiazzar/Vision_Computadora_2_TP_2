from pathlib import Path

import torch
from ultralytics import YOLO

# Carpeta de este script: las rutas no dependen de desde dónde se lo ejecute
HERE = Path(__file__).resolve().parent

# Ruta al dataset
DATASET_PATH = HERE / "Dataset YOLO - 1" / "data.yaml"

# Modelo preentrenado
MODEL_NAME = "yolo11s.pt"

# Identificación del experimento
PROJECT_NAME = "blackjack-cv"
RUN_NAME = "yolo1-baseline"

# Hiperparámetros principales
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16
PATIENCE = 15

# GPU NVIDIA (cuda), Apple Silicon (mps) o CPU, según lo que haya
if torch.cuda.is_available():
    DEVICE = 0
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"No encontré data.yaml en: {DATASET_PATH}"
    )

print("=" * 60)
print("ENTRENAMIENTO YOLO-1")
print("=" * 60)

print(f"Modelo:       {MODEL_NAME}")
print(f"Dataset:      {DATASET_PATH}")
print(f"Epochs:       {EPOCHS}")
print(f"Image size:   {IMAGE_SIZE}")
print(f"Batch size:   {BATCH_SIZE}")
print(f"Patience:     {PATIENCE}")
print(f"Device:       {DEVICE}")
print(f"Proyecto:     {PROJECT_NAME}")
print(f"Run:          {RUN_NAME}")

print("=" * 60)


model = YOLO(MODEL_NAME)

results = model.train(
    # Dataset
    data=str(DATASET_PATH),

    # Entrenamiento
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    patience=PATIENCE,

    # Hardware
    device=DEVICE,

    # Optimización
    optimizer="auto",

    # Output: <carpeta del script>/runs/detect/blackjack-cv/<RUN_NAME>
    project=str(HERE / "runs" / "detect" / PROJECT_NAME),
    name=RUN_NAME,

    # Guardar checkpoints
    save=True,
    # Mostrar información durante entrenamiento
    verbose=True,
)



print("\n" + "=" * 60)
print("ENTRENAMIENTO TERMINADO")
print("=" * 60)

print("Resultados:")
print(results)

print(
    f"\nBuscá el checkpoint best.pt dentro de "
    f"{HERE / 'runs' / 'detect' / PROJECT_NAME / RUN_NAME / 'weights'}"
)