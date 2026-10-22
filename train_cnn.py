#!/usr/bin/env python
"""Train a small CNN baseline on MNIST for comparison with the FlyVis readout.

The CNN uses the same train/validation/test split and seed as
``train_mnist.py`` so the two test accuracies are directly comparable.  It is a
plain supervised baseline on raw pixels; no FlyVis simulation is involved.

    /home/bibu/flyvis/bin/python train_cnn.py

Outputs:

    runs/cnn_mnist/cnn.pt
    runs/cnn_mnist/metrics.json
    runs/cnn_mnist/test_predictions.pt
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=60_000)
    parser.add_argument("--test-size", type=int, default=10_000)
    parser.add_argument("--val-size", type=int, default=5_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "cnn_mnist")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(n_total: int, train_size: int, val_size: int, seed: int):
    """Match the split logic in train_mnist.py exactly."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pool = torch.randperm(n_total, generator=generator, device="cpu")[:train_size]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(pool), generator=generator, device="cpu")
    val_size = max(1, min(val_size, max(1, len(pool) // 5)))
    val_idx = [int(pool[i]) for i in order[:val_size]]
    train_idx = [int(pool[i]) for i in order[val_size:]]
    return train_idx, val_idx


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@torch.no_grad()
def evaluate(model, loader) -> float:
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        correct += (model(images).argmax(dim=1) == labels).sum().item()
        total += len(labels)
    model.train(was_training)
    return correct / total


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_source = MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_source = MNIST(args.data_dir, train=False, download=True, transform=transform)

    train_idx, val_idx = split_indices(
        len(train_source), min(args.train_size, len(train_source)), args.val_size, args.seed
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    test_idx = torch.randperm(len(test_source), generator=generator, device="cpu")[
        :min(args.test_size, len(test_source))
    ].tolist()

    train_loader = DataLoader(
        Subset(train_source, train_idx), batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        Subset(train_source, val_idx), batch_size=512,
        shuffle=False, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(test_source, test_idx), batch_size=512,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    print(f"Torch {torch.__version__} | device: {DEVICE}")
    print(f"train {len(train_idx)} | val {len(val_idx)} | test {len(test_idx)}")

    model = SmallCNN().to(DEVICE)
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"CNN parameters: {n_parameters:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = 0.0
    best_epoch = 0
    best_state = None
    history = []
    no_improvement = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            loss = F.cross_entropy(model(images), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        scheduler.step()
        train_loss = total_loss / len(train_idx)
        val_acc = evaluate(model, val_loader)
        history.append({"epoch": epoch, "loss": train_loss, "val_acc": val_acc})
        print(f"epoch {epoch:3d}/{args.epochs} | loss {train_loss:.4f} | val {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
            if args.patience and no_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_acc = evaluate(model, train_loader)
    val_acc = evaluate(model, val_loader)
    test_acc = evaluate(model, test_loader)

    model.eval()
    with torch.no_grad():
        predictions = []
        labels_all = []
        for images, labels in test_loader:
            predictions.append(model(images.to(DEVICE)).argmax(dim=1).cpu())
            labels_all.append(labels)
        test_pred = torch.cat(predictions)
        test_y = torch.cat(labels_all)
        confusion = torch.zeros(10, 10, dtype=torch.long)
        for target, pred in zip(test_y, test_pred):
            confusion[target, pred] += 1

    elapsed = time.time() - started
    metrics = {
        "torch_version": torch.__version__,
        "device": str(DEVICE),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "n_parameters": n_parameters,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "best_epoch": best_epoch,
        "best_val_accuracy": val_acc,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_seconds": elapsed,
        "history": history,
        "confusion_matrix": confusion.tolist(),
    }

    torch.save({"model_state": model.state_dict(), "config": metrics["config"]},
               args.out_dir / "cnn.pt")
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    torch.save({"predictions": test_pred, "labels": test_y},
               args.out_dir / "test_predictions.pt")

    print(f"Best epoch: {best_epoch} | val {val_acc:.4f} | test {test_acc:.4f}")
    print(f"Wrote {args.out_dir / 'cnn.pt'} and metrics.json")


if __name__ == "__main__":
    main()
