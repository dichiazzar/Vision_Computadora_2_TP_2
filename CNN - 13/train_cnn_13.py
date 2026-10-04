"""Entrenamiento de la CNN-13 (pipeline B: YOLO-1 localiza la esquina -> esta CNN clasifica el valor).

Clasifica el recorte del índice de una carta en 13 clases (A, 2..10, J, Q, K) con una ResNet-18
pre-entrenada en ImageNet (transfer learning, se ajusta toda la red).

Uso (desde cualquier carpeta):
    python "CNN - 13/train_cnn_13.py"
    python "CNN - 13/train_cnn_13.py" --epochs 15 --img-size 96

Salida en CNN - 13/runs/<run>/:
    best.pt          pesos con mejor exactitud de validación (+ clases y preprocesamiento)
    history.csv      loss / accuracy por época
    metrics.json     exactitud y F1 macro en test, tiempos
    confusion_matrix.png
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

HERE = Path(__file__).resolve().parent
DATASET = HERE / "Dataset CNN-13"

# Normalización de ImageNet (la ResNet-18 se pre-entrenó con ella)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--img-size", type=int, default=96)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=4, help="early stopping (épocas sin mejorar en validación)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run", default="cnn13-resnet18")
    p.add_argument("--eval-only", action="store_true", help="no entrenar: evaluar en test el best.pt existente")
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms(size):
    # Sin flips: un índice espejado no es una carta válida (p. ej. el 7 o la J).
    # Las esquinas aparecen derechas e invertidas (180°) y el dataset ya trae ambas.
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.85, 1.1)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.03),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def build_model(num_classes):
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception as e:  # sin internet: se entrena desde cero (avisar en el informe)
        print(f"No se pudieron bajar los pesos de ImageNet ({e}); se entrena desde cero.")
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum, y_true, y_pred = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * len(y)
        y_true.append(y.cpu())
        y_pred.append(out.argmax(1).cpu())
    y_true, y_pred = torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()
    return loss_sum / len(y_true), float((y_true == y_pred).mean()), y_true, y_pred


def macro_f1(y_true, y_pred, n):
    f1s = []
    for c in range(n):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1s)), f1s


def save_confusion(y_true, y_pred, classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(classes)
    cm = np.zeros((n, n), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), classes)
    ax.set_yticks(range(n), classes)
    ax.set_xlabel("predicción")
    ax.set_ylabel("real")
    ax.set_title("CNN-13 — matriz de confusión normalizada (test)")
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    out_dir = HERE / "runs" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms(args.img_size)
    ds = {
        "train": datasets.ImageFolder(DATASET / "train", train_tf),
        "valid": datasets.ImageFolder(DATASET / "valid", eval_tf),
        "test": datasets.ImageFolder(DATASET / "test", eval_tf),
    }
    classes = ds["train"].classes  # orden alfabético: '10','2',...,'9','A','J','K','Q' (igual que YOLO-13)
    assert ds["valid"].classes == classes == ds["test"].classes

    kw = dict(batch_size=args.batch, num_workers=args.workers, persistent_workers=args.workers > 0)
    loaders = {
        "train": DataLoader(ds["train"], shuffle=True, **kw),
        "valid": DataLoader(ds["valid"], shuffle=False, **kw),
        "test": DataLoader(ds["test"], shuffle=False, **kw),
    }

    print("=" * 60)
    print(f"CNN-13 | device={device} | img={args.img_size} | epochs={args.epochs} | batch={args.batch}")
    print({k: len(v) for k, v in ds.items()}, "recortes | clases:", classes)
    print("=" * 60)

    model = build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    if not args.eval_only:
        train(model, loaders, criterion, classes, device, args, out_dir)

    # Evaluación final en test con los mejores pesos (archivo propio: weights_only=False es seguro)
    best = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    _, test_acc, y_true, y_pred = evaluate(model, loaders["test"], device, criterion)
    f1, f1s = macro_f1(y_true, y_pred, len(classes))
    save_confusion(y_true, y_pred, classes, out_dir / "confusion_matrix.png")

    # Latencia de la CNN sola (batch = 1), para comparar con el pipeline A
    model.eval()
    x = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        t0 = time.perf_counter()
        for _ in range(100):
            model(x)
    lat_ms = (time.perf_counter() - t0) * 10

    with open(out_dir / "history.csv", newline="") as f:
        history = list(csv.DictReader(f))
    metrics = dict(best_epoch=int(best["epoch"]), val_acc=float(best["val_acc"]), test_acc=test_acc,
                   test_macro_f1=f1, test_f1_por_clase=dict(zip(classes, f1s)),
                   latencia_ms_batch1=lat_ms, device=str(device), epocas_entrenadas=len(history),
                   train_min=sum(float(r["sec"]) for r in history) / 60, args=vars(args))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("=" * 60)
    print(f"TEST | acc {test_acc:.4f} | F1 macro {f1:.4f} | latencia {lat_ms:.1f} ms/recorte ({device})")
    print("Resultados en:", out_dir)


def train(model, loaders, criterion, classes, device, args, out_dir):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.epochs * len(loaders["train"]))

    history, best_acc, bad_epochs = [], -1.0, 0
    ckpt = {"classes": classes, "img_size": args.img_size, "mean": MEAN, "std": STD, "arch": "resnet18"}
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(y)
            tc += (out.argmax(1) == y).sum().item()
            tn += len(y)
        vl, va, _, _ = evaluate(model, loaders["valid"], device, criterion)
        row = dict(epoch=ep, train_loss=tl / tn, train_acc=tc / tn, val_loss=vl, val_acc=va, sec=time.time() - t0)
        history.append(row)
        print(f"época {ep:2d}/{args.epochs} | train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} | "
              f"val loss {vl:.4f} acc {va:.4f} | {row['sec']:.0f}s", flush=True)

        if va > best_acc:
            best_acc, bad_epochs = va, 0
            torch.save({**ckpt, "state_dict": model.state_dict(), "epoch": ep, "val_acc": va}, out_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stopping: {args.patience} épocas sin mejorar")
                break

    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)


if __name__ == "__main__":
    main()
