from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from imu_gesture.data import load_dataset, save_json, stratified_split
from imu_gesture.model import IMUCnnLstmNet


def configure_matplotlib_fonts() -> None:
    font_paths = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ]
    for font_path in font_paths:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK JP",
        "Droid Sans Fallback",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an IMU gesture classifier.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-flat-files", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def normalize_by_train(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x_train - mean) / std, (x_val - mean) / std, (x_test - mean) / std, mean, std


def split_train_val_test(
    y: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 <= val_ratio < 1.0 or not 0.0 <= test_ratio < 1.0:
        raise ValueError("--val-ratio and --test-ratio must be in [0, 1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("--val-ratio + --test-ratio must be less than 1")

    train_val_idx, test_idx = stratified_split(y, val_ratio=test_ratio, seed=seed + 1)
    if len(test_idx) == 0:
        raise ValueError("test split is empty; add more samples or reduce --test-ratio")

    adjusted_val_ratio = val_ratio / (1.0 - test_ratio) if test_ratio < 1.0 else 0.0
    train_rel_idx, val_rel_idx = stratified_split(
        y[train_val_idx],
        val_ratio=adjusted_val_ratio,
        seed=seed,
    )
    train_idx = train_val_idx[train_rel_idx]
    val_idx = train_val_idx[val_rel_idx]
    if len(val_idx) == 0:
        raise ValueError("validation split is empty; add more samples or reduce --val-ratio")
    if len(train_idx) == 0:
        raise ValueError("training split is empty; reduce validation/test ratios")
    return train_idx, val_idx, test_idx


def make_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(torch.from_numpy(x).permute(0, 2, 1), torch.from_numpy(y))


def evaluate(
    model: nn.Module,
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(x_tensor.to(device))
        loss = criterion(logits, y_tensor.to(device)).item()
        preds = logits.argmax(dim=1).cpu().numpy()
        acc = accuracy(logits.cpu(), y_tensor)
    return loss, acc, preds


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true.tolist(), y_pred.tolist()):
        matrix[true_label, pred_label] += 1
    return matrix


def save_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(history)


def plot_loss_curve(path: Path, history: list[dict[str, float]]) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.plot(epochs, train_loss, label="train loss", linewidth=2)
    ax.plot(epochs, val_loss, label="val loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion_matrix(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    fig_size = max(6, len(labels) * 1.1)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=140)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Test Confusion Matrix")

    threshold = matrix.max() / 2 if matrix.size and matrix.max() > 0 else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            color = "white" if value > threshold else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    configure_matplotlib_fonts()
    args = parse_args()
    set_seed(args.seed)

    x, y, labels, paths, feature_names = load_dataset(
        args.dataset,
        seq_len=args.seq_len,
        include_flat_files=args.include_flat_files,
    )
    train_idx, val_idx, test_idx = split_train_val_test(
        y,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    x_train, x_val, x_test, mean, std = normalize_by_train(
        x[train_idx],
        x[val_idx],
        x[test_idx],
    )
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    train_loader = DataLoader(
        make_dataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_x = torch.from_numpy(x_val).permute(0, 2, 1)
    val_y = torch.from_numpy(y_val)
    test_x = torch.from_numpy(x_test).permute(0, 2, 1)
    test_y = torch.from_numpy(y_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IMUCnnLstmNet(input_channels=x.shape[2], num_classes=len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_count = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            batch_count = len(batch_y)
            total_loss += loss.item() * batch_count
            total_acc += accuracy(logits.detach(), batch_y) * batch_count
            total_count += batch_count

        train_loss = total_loss / max(total_count, 1)
        train_acc = total_acc / max(total_count, 1)
        val_loss, val_acc, _ = evaluate(model, val_x, val_y, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d} | train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.3f} | val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_pred = evaluate(model, test_x, test_y, criterion, device)
    conf = confusion_matrix(y_test, test_pred, num_classes=len(labels))

    checkpoint = {
        "model_name": "IMUCnnLstmNet",
        "model_state": model.state_dict(),
        "input_channels": int(x.shape[2]),
        "num_classes": len(labels),
        "seq_len": args.seq_len,
        "labels": labels,
        "feature_names": feature_names,
    }
    torch.save(checkpoint, args.out_dir / "imu_gesture_model.pt")
    save_json(args.out_dir / "label_map.json", {str(i): label for i, label in enumerate(labels)})
    save_json(
        args.out_dir / "normalization.json",
        {
            "mean": mean.reshape(-1).astype(float).tolist(),
            "std": std.reshape(-1).astype(float).tolist(),
            "feature_names": feature_names,
        },
    )
    save_json(
        args.out_dir / "split.json",
        {
            "train": [str(paths[i]) for i in train_idx],
            "val": [str(paths[i]) for i in val_idx],
            "test": [str(paths[i]) for i in test_idx],
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        },
    )
    save_json(args.out_dir / "history.json", history)
    save_history_csv(args.out_dir / "history.csv", history)
    save_json(
        args.out_dir / "metrics.json",
        {
            "classes": labels,
            "samples": len(paths),
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "confusion_matrix": conf.astype(int).tolist(),
        },
    )
    plot_loss_curve(args.out_dir / "loss_curve.png", history)
    plot_confusion_matrix(args.out_dir / "confusion_matrix.png", conf, labels)

    print("test confusion matrix rows=true cols=pred")
    print("labels", json.dumps(labels, ensure_ascii=False))
    print(conf)
    print(
        json.dumps(
            {
                "classes": labels,
                "samples": len(paths),
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
