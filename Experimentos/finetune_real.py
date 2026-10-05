"""Fine-tuning de los tres modelos con las fotos reales (por defecto Dataset Real/teogopk; con --dataset/--real, otro export YOLO, p. ej. fotos propias), partiendo de los pesos ya entrenados.

Para no "olvidar" lo aprendido con el dataset sintético, cada época mezcla:
  - una muestra aleatoria del train sintético (--n-sintetico imágenes / recortes), y
  - el train real repetido --repetir-real veces (hay pocas fotos reales: así pesan más).
La validación (y la selección de best.pt) se hace SÓLO con el valid real. El test real no se toca:
se usa después con Experimentos/evaluar_real.py.

Modelos:
  yolo13 -> YOLO - 13/runs/detect/blackjack-cv/yolo13-finetune-real/weights/best.pt   (pipeline A)
  yolo1  -> YOLO - 1/runs/detect/blackjack-cv/yolo1-finetune-real/weights/best.pt     (pipeline B, detector)
  cnn    -> CNN - 13/runs/cnn13-finetune-real/best.pt                                  (pipeline B, clasificador)

Uso (desde la carpeta del repo):
    python Experimentos/finetune_real.py --modelo todos
    python Experimentos/finetune_real.py --modelo cnn --epochs 8
    python Experimentos/finetune_real.py --modelo yolo13 --device mps      # en la Mac
"""

import argparse
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import yaml

REPO = Path(__file__).resolve().parent.parent
# --dataset elige a la vez de dónde se leen las fotos y con qué etiqueta se guardan los pesos (no se pueden mezclar)
DATASETS = {"propio": (REPO / "Dataset Real" / "roboflow", "propio"),    # fotos del mazo de la demo
            "teogopk": (REPO / "Dataset Real" / "teogopk", "real")}      # dataset público (otro mazo)
REAL = DATASETS["propio"][0]
ETIQUETA = "propio"
TRABAJO = REPO / "Experimentos" / "finetune_data"
sys.path.insert(0, str(REPO / "Identificador_Cartas"))
sys.path.insert(0, str(REPO / "CNN - 13"))

import detectores as D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clases import CLASES, normalizar_clase  # noqa: E402
SINTETICO = {
    "yolo13": REPO / "YOLO - 13" / "Dataset YOLO 13",
    "yolo1": REPO / "YOLO - 1" / "Dataset YOLO - 1",
    "cnn": REPO / "CNN - 13" / "Dataset CNN-13",
}
EXT = {".jpg", ".jpeg", ".png"}


# ============================================================
# DATOS REALES: etiquetas reordenadas a las clases del proyecto
# ============================================================

def nombres_reales():
    nombres = yaml.safe_load((REAL / "data.yaml").read_text(encoding="utf-8"))["names"]
    if isinstance(nombres, dict):
        nombres = [nombres[k] for k in sorted(nombres, key=int)]
    return [normalizar_clase(n) for n in nombres]     # '10h' / 'AS' / ... -> valor


def preparar_real(destino, una_clase):
    """Copia train/valid reales a `destino` con las etiquetas en el orden de CLASES (o todas = 0).
    Roboflow exporta sólo las clases presentes y en su propio orden, así que hay que remapear por nombre."""
    nombres = nombres_reales()
    mapa = {i: (0 if una_clase else CLASES.index(n)) for i, n in enumerate(nombres)}
    imagenes = {}
    for split in ("train", "valid"):
        src = REAL / split
        (destino / split / "images").mkdir(parents=True, exist_ok=True)
        (destino / split / "labels").mkdir(parents=True, exist_ok=True)
        imagenes[split] = []
        for img in sorted((src / "images").glob("*")):
            if img.suffix.lower() not in EXT:
                continue
            dst = destino / split / "images" / img.name
            shutil.copy2(img, dst)
            lbl = src / "labels" / f"{img.stem}.txt"
            filas = []
            if lbl.exists():
                for linea in lbl.read_text().splitlines():
                    p = linea.split()
                    if len(p) >= 5:
                        filas.append(" ".join([str(mapa[int(float(p[0]))])] + p[1:5]))
            (destino / split / "labels" / f"{img.stem}.txt").write_text("\n".join(filas))
            imagenes[split].append(dst)
    print(f"  real: {len(imagenes['train'])} train / {len(imagenes['valid'])} valid")
    return imagenes


# ============================================================
# YOLO-13 y YOLO-1
# ============================================================

def finetune_yolo(modelo, args):
    from ultralytics import YOLO

    una_clase = modelo == "yolo1"
    base = D.PESOS_YOLO_1 if una_clase else D.PESOS_YOLO_13
    y13, y1, _ = D.pesos_finetune(ETIQUETA)
    destino_pesos = y1 if una_clase else y13
    run_dir = destino_pesos.parent.parent
    trabajo = TRABAJO / modelo
    if trabajo.exists():
        shutil.rmtree(trabajo)

    print(f"\n=== Fine-tuning {modelo} (desde {base.parent.parent.name}) ===")
    real = preparar_real(trabajo / "real", una_clase)
    sint = sorted(p for p in (SINTETICO[modelo] / "train" / "images").glob("*") if p.suffix.lower() in EXT)
    random.Random(args.seed).shuffle(sint)
    train = sint[: args.n_sintetico] + real["train"] * args.repetir_real
    random.Random(args.seed).shuffle(train)
    (trabajo / "train.txt").write_text("\n".join(str(p) for p in train))
    (trabajo / "val.txt").write_text("\n".join(str(p) for p in real["valid"]))
    data = trabajo / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(trabajo), "train": str(trabajo / "train.txt"),
                                    "val": str(trabajo / "val.txt"),
                                    "names": ["card-corner"] if una_clase else CLASES}, sort_keys=False))
    print(f"  train por época: {min(args.n_sintetico, len(sint))} sintéticas + "
          f"{len(real['train'])} reales x{args.repetir_real} | val: {len(real['valid'])} reales")

    t0 = time.time()
    YOLO(str(base)).train(
        data=str(data), epochs=args.epochs, imgsz=640, batch=args.batch_yolo, patience=args.patience,
        optimizer="SGD", lr0=args.lr_yolo, lrf=0.1, warmup_epochs=1, seed=args.seed,
        device=args.device, workers=args.workers, project=str(run_dir.parent), name=run_dir.name,
        exist_ok=True, plots=True, verbose=False)
    print(f"  listo en {(time.time() - t0) / 60:.1f} min -> {destino_pesos}")


# ============================================================
# CNN-13
# ============================================================

def recortar_reales(destino):
    """Recortes de las esquinas reales con el mismo padding que create_cnn_dataset.py."""
    nombres = nombres_reales()
    for split in ("train", "valid", "test"):
        for c in CLASES:
            (destino / split / c).mkdir(parents=True, exist_ok=True)
        if not (REAL / split / "images").exists():
            continue
        for img_path in sorted((REAL / split / "images").glob("*")):
            img = cv2.imread(str(img_path))
            lbl = REAL / split / "labels" / f"{img_path.stem}.txt"
            if img is None or not lbl.exists():
                continue
            h, w = img.shape[:2]
            for k, linea in enumerate(lbl.read_text().splitlines()):
                p = linea.split()
                if len(p) < 5:
                    continue
                clase = nombres[int(float(p[0]))]
                xc, yc, bw, bh = float(p[1]) * w, float(p[2]) * h, float(p[3]) * w, float(p[4]) * h
                px, py = bw * D.PADDING_RECORTE, bh * D.PADDING_RECORTE
                x1, y1 = max(0, round(xc - bw / 2 - px)), max(0, round(yc - bh / 2 - py))
                x2, y2 = min(w, round(xc + bw / 2 + px)), min(h, round(yc + bh / 2 + py))
                if x2 > x1 and y2 > y1:
                    cv2.imwrite(str(destino / split / clase / f"{img_path.stem}_bbox{k:02d}.jpg"),
                                img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])


def finetune_cnn(args):
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import ConcatDataset, DataLoader, Subset
    from torchvision import datasets

    import train_cnn_13 as T

    print("\n=== Fine-tuning CNN-13 (desde cnn13-resnet18) ===")
    crops = REPO / "CNN - 13" / "Dataset CNN-13-real"
    if crops.exists():
        shutil.rmtree(crops)
    recortar_reales(crops)

    torch.manual_seed(args.seed)
    device = T.get_device() if args.device in (None, "") else torch.device(
        "cuda" if str(args.device).isdigit() else args.device)
    base = torch.load(D.PESOS_CNN_13, map_location="cpu", weights_only=False)
    assert base["classes"] == CLASES
    train_tf, eval_tf = T.build_transforms(base["img_size"])

    sint = datasets.ImageFolder(SINTETICO["cnn"] / "train", train_tf)
    real_tr = datasets.ImageFolder(crops / "train", train_tf, allow_empty=True)
    real_va = datasets.ImageFolder(crops / "valid", eval_tf, allow_empty=True)
    assert sint.classes == real_tr.classes == real_va.classes == CLASES
    idx = np.random.default_rng(args.seed).permutation(len(sint))[: args.n_sintetico_cnn]
    train = ConcatDataset([Subset(sint, idx.tolist())] + [real_tr] * args.repetir_real)
    print(f"  recortes: {len(idx)} sintéticos + {len(real_tr)} reales x{args.repetir_real} | "
          f"val: {len(real_va)} reales")

    kw = dict(batch_size=128, num_workers=args.workers, persistent_workers=args.workers > 0)
    dl_tr, dl_va = DataLoader(train, shuffle=True, **kw), DataLoader(real_va, shuffle=False, **kw)

    model = T.build_model(len(CLASES))
    model.load_state_dict(base["state_dict"])
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_cnn, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(dl_tr))

    destino_cnn = D.pesos_finetune(ETIQUETA)[2]
    out = destino_cnn.parent
    out.mkdir(parents=True, exist_ok=True)
    _, acc0, _, _ = T.evaluate(model, dl_va, device, criterion)
    print(f"  exactitud en valid real ANTES del fine-tuning: {acc0:.4f}")
    mejor, sin_mejora = -1.0, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            criterion(model(x), y).backward()
            opt.step()
            sched.step()
        vl, va, _, _ = T.evaluate(model, dl_va, device, criterion)
        print(f"  época {ep:2d}/{args.epochs} | val real loss {vl:.4f} acc {va:.4f} | {time.time() - t0:.0f}s",
              flush=True)
        if va > mejor:
            mejor, sin_mejora = va, 0
            torch.save({**{k: v for k, v in base.items() if k != "state_dict"},
                        "state_dict": model.state_dict(), "epoch": ep, "val_acc": float(va),
                        "val_real_acc_antes": float(acc0), "finetune_args": vars(args)}, destino_cnn)
        else:
            sin_mejora += 1
            if sin_mejora >= args.patience:
                print("  early stopping")
                break
    print(f"  mejor exactitud en valid real: {mejor:.4f} (antes {acc0:.4f}) -> {destino_cnn}")


def main():
    global REAL, ETIQUETA
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="todos", choices=["todos", "yolo13", "yolo1", "cnn"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--n-sintetico", type=int, default=600, help="imágenes sintéticas por época (YOLO)")
    ap.add_argument("--n-sintetico-cnn", type=int, default=4000, help="recortes sintéticos por época (CNN)")
    ap.add_argument("--repetir-real", type=int, default=5, help="veces que se repite el train real por época")
    ap.add_argument("--batch-yolo", type=int, default=16)
    ap.add_argument("--lr-yolo", type=float, default=0.002)
    ap.add_argument("--lr-cnn", type=float, default=3e-4)
    ap.add_argument("--device", default=None, help="cpu | mps | 0 (GPU). Por defecto: automático")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="propio", choices=list(DATASETS),
                    help="propio = Dataset Real/roboflow (pesos *-finetune-propio) | "
                         "teogopk = Dataset Real/teogopk (pesos *-finetune-real)")
    ap.add_argument("--real", help="otra carpeta exportada de Roboflow (se guarda con --etiqueta)")
    ap.add_argument("--etiqueta", help="nombre de los pesos con --real (p. ej. 'propio2')")
    args = ap.parse_args()

    REAL, ETIQUETA = DATASETS[args.dataset]
    if args.real:
        if not args.etiqueta:
            sys.exit("con --real hay que indicar también --etiqueta (para no pisar otros pesos)")
        REAL, ETIQUETA = Path(args.real), args.etiqueta
    print(f"dataset real: {REAL} -> pesos '*-finetune-{ETIQUETA}'")

    if not (REAL / "data.yaml").exists():
        sys.exit(f"No encontré {REAL / 'data.yaml'}: exportar el dataset real de Roboflow ahí (ver Dataset Real/README.md)")
    if args.device is None:
        args.device = D._device() if D._device() != "cuda" else 0

    modelos = ["cnn", "yolo1", "yolo13"] if args.modelo == "todos" else [args.modelo]
    for m in modelos:
        finetune_cnn(args) if m == "cnn" else finetune_yolo(m, args)
    print("\nAhora evaluar en el test real:  python Experimentos/evaluar_real.py")


if __name__ == "__main__":
    main()
