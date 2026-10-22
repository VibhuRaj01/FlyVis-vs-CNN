#!/usr/bin/env python
"""Train an MNIST digit classifier on top of a frozen pretrained FlyVis network.

The pretrained FlyVis model is a 45k-node connectome-constrained optic-lobe
simulation, not a CNN.  This script:

1. maps each 28x28 MNIST image onto the 721 photoreceptor positions of the
   FlyVis hexagonal lattice,
2. runs the frozen pretrained network for a short static-image sequence,
3. pools the activity of all non-photoreceptor cell types into a compact
   feature vector, and
4. trains a small MLP readout on those features.

Only the readout is trained.  The FlyVis parameters stay frozen at the
pretrained "flow/0000/000" checkpoint, which keeps the run fast and avoids
destroying the pretrained optic-flow representation.

Run from the repository root with the FlyVis virtual environment:

    /home/bibu/flyvis/bin/python train_mnist.py

Quick smoke test:

    /home/bibu/flyvis/bin/python train_mnist.py \
        --train-size 2000 --test-size 1000 --epochs 3 --no-cache
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

# These must be set before importing torch/flyvis.  FlyVis reads the .env file
# from the current working directory, but setting the variable here also makes
# the script work when launched from somewhere else.
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("FLYVIS_ROOT_DIR", str(ROOT / "data"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

import flyvis
from flyvis import NetworkView

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_CELL_TYPES = ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8")
HEX_EXTENT = 15
GRID_SIZE = 2 * HEX_EXTENT + 1  # 31x31 axial map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=60_000,
                        help="number of MNIST training images to use")
    parser.add_argument("--test-size", type=int, default=10_000,
                        help="number of MNIST test images to use")
    parser.add_argument("--val-size", type=int, default=5_000,
                        help="held-out validation images taken from the training pool")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10,
                        help="early-stopping patience in epochs; 0 disables it")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="FlyVis simulation batch size (32 is safe on 12 GB)")
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--n-frames", type=int, default=6,
                        help="number of repeated static frames per image")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="FlyVis integration time step (must be <= 0.02)")
    parser.add_argument("--pool", type=int, default=7,
                        help="spatial pool size per cell type (7 -> 7x7 map)")
    parser.add_argument("--cell-types", choices=["non-input", "output", "all"],
                        default="non-input")
    parser.add_argument("--hidden", type=int, default=128,
                        help="readout hidden size; 0 means a linear classifier")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".data")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "runs" / "flyvis_mnist" / "cache")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "flyvis_mnist")
    parser.add_argument("--no-cache", action="store_true",
                        help="recompute features even if a cache file exists")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subset(dataset: Dataset, size: int, seed: int) -> Subset:
    size = min(size, len(dataset))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator, device="cpu")[:size].tolist()
    return Subset(dataset, indices)


def load_cell_types(net, mode: str) -> List[str]:
    layer_index = net.connectome.nodes.layer_index
    available = [str(key) for key in layer_index.keys()]
    if mode == "output":
        return [t.decode() if isinstance(t, bytes) else str(t)
                for t in net.connectome.output_cell_types]
    if mode == "all":
        return available
    return [name for name in available if name not in INPUT_CELL_TYPES]


def build_hex_encoder(network_view, device: torch.device):
    """Return a grid_sample grid that maps a 28x28 image onto the R1 lattice."""
    u, v = network_view.get_uv("R1")
    u = torch.as_tensor(u, dtype=torch.float32, device=device)
    v = torch.as_tensor(v, dtype=torch.float32, device=device)
    gx = 2.0 * (u - u.min()) / (u.max() - u.min()) - 1.0
    gy = 2.0 * (v - v.min()) / (v.max() - v.min()) - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
    return grid, len(u)


def build_feature_maps(net, cell_types: Sequence[str], device: torch.device):
    """Precompute node indices and 31x31 scatter/pooling maps per cell type."""
    layer_index = net.connectome.nodes.layer_index
    indices: List[torch.Tensor] = []
    flat_positions: List[torch.Tensor] = []
    occupancy: List[torch.Tensor] = []

    for name in cell_types:
        idx_np = np.asarray(layer_index[name], dtype=np.int64)
        idx = torch.as_tensor(idx_np, dtype=torch.long)

        # Use the coordinates attached to these exact node indices.  Layer
        # index order and get_uv() order agree, but this is explicit and safe.
        u = np.asarray(net.connectome.nodes.u[idx_np])
        v = np.asarray(net.connectome.nodes.v[idx_np])
        rows = torch.as_tensor(np.rint(v + HEX_EXTENT), dtype=torch.long)
        cols = torch.as_tensor(np.rint(u + HEX_EXTENT), dtype=torch.long)
        flat = rows * GRID_SIZE + cols

        mask = torch.zeros(GRID_SIZE * GRID_SIZE, dtype=torch.float32)
        mask[flat] = 1.0

        indices.append(idx.to(device))
        flat_positions.append(flat.to(device))
        occupancy.append(mask.view(1, 1, GRID_SIZE, GRID_SIZE).to(device))

    return indices, flat_positions, occupancy


def pooled_features(
    activity: torch.Tensor,
    indices: Sequence[torch.Tensor],
    flat_positions: Sequence[torch.Tensor],
    occupancy: Sequence[torch.Tensor],
    pool: int,
) -> torch.Tensor:
    """Spatially pool each cell type's activity, then concatenate.

    ``activity`` has shape (batch, nodes).  Returns (batch, features) with both
    mean and max pooled maps for every selected cell type.
    """
    batch = activity.shape[0]
    features: List[torch.Tensor] = []

    for idx, flat, mask in zip(indices, flat_positions, occupancy):
        dense = torch.zeros(batch, GRID_SIZE * GRID_SIZE, device=activity.device)
        dense.scatter_(1, flat.unsqueeze(0).expand(batch, -1), activity[:, idx])
        dense = dense.view(batch, 1, GRID_SIZE, GRID_SIZE)

        # Mean over occupied hexals only (empty lattice positions stay zero).
        mean = F.adaptive_avg_pool2d(dense, (pool, pool))
        count = F.adaptive_avg_pool2d(mask, (pool, pool))
        mean = mean / (count + 1e-6)

        max_pooled = F.adaptive_max_pool2d(dense, (pool, pool))
        features.extend([mean.flatten(1), max_pooled.flatten(1)])

    return torch.cat(features, dim=1)


def preprocess_images(images: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Map (B,1,28,28) MNIST images onto the 721-element hex lattice."""
    batch = images.shape[0]
    sampled = F.grid_sample(
        images,
        grid.expand(batch, -1, -1, -1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[:, 0, 0]  # (B, 721)


def extract_features(
    dataset: Dataset,
    indices: Iterable[int],
    net,
    grid: torch.Tensor,
    feature_maps,
    n_frames: int,
    dt: float,
    pool: int,
    batch_size: int,
    device: torch.device,
    desc: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = list(indices)
    node_indices, flat_positions, occupancy = feature_maps
    initial_states = {}
    features: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    started = time.time()

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        images = torch.stack([dataset[i][0] for i in batch_indices]).to(device)
        target = torch.tensor([dataset[i][1] for i in batch_indices], dtype=torch.long)

        # One image is broadcast by Stimulus.add_input to all eight
        # photoreceptor types, so the last dimension is 721, not 8*721.
        hex_input = preprocess_images(images, grid)
        movie = hex_input[:, None, None, :].repeat(1, n_frames, 1, 1)

        batch = len(batch_indices)
        if batch not in initial_states:
            with torch.no_grad():
                initial_states[batch] = net.steady_state(1.0, dt, batch)

        with torch.no_grad():
            activity = net.simulate(movie, dt, initial_state=initial_states[batch])

        # Use the last few frames; the network has a short transient.
        late = activity[:, -min(3, n_frames):, :].mean(dim=1)
        features.append(pooled_features(late, node_indices, flat_positions, occupancy, pool)
                        .half().cpu())
        labels.append(target)

        done = start + batch
        if done % (batch_size * 10) == 0 or done == len(indices):
            elapsed = time.time() - started
            rate = done / max(elapsed, 1e-6)
            print(f"\r{desc}: {done}/{len(indices)} "
                  f"({rate:.1f} img/s, {elapsed:.0f}s)", end="", flush=True)

    print()
    return torch.cat(features), torch.cat(labels)


def cache_path(cache_dir: Path, split: str, size: int, seed: int,
               n_frames: int, pool: int, cell_types: str, dt: float) -> Path:
    return cache_dir / (
        f"{split}_n{size}_seed{seed}_f{n_frames}_p{pool}_dt{dt}_{cell_types}.pt"
    )


def get_features(args, dataset, size, split, net, grid, feature_maps, device):
    path = cache_path(args.cache_dir, split, size, args.seed,
                      args.n_frames, args.pool, args.cell_types, args.dt)
    if path.exists() and not args.no_cache:
        print(f"Loading cached {split} features from {path}")
        payload = torch.load(path, map_location="cpu")
        return payload["features"], payload["labels"]

    subset_ds = subset(dataset, size, args.seed)
    x, y = extract_features(
        subset_ds,
        range(len(subset_ds)),
        net,
        grid,
        feature_maps,
        n_frames=args.n_frames,
        dt=args.dt,
        pool=args.pool,
        batch_size=args.feature_batch_size,
        device=device,
        desc=f"extract {split}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": x, "labels": y}, path)
    print(f"Saved {split} features to {path}")
    return x, y


class Readout(nn.Module):
    def __init__(self, in_features: int, hidden: int, dropout: float):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 10),
            )
        else:
            self.net = nn.Linear(in_features, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


@torch.no_grad()
def evaluate(model, x, y, batch_size=1024):
    was_training = model.training
    model.eval()
    correct = 0
    for start in range(0, len(x), batch_size):
        logits = model(x[start:start + batch_size])
        correct += (logits.argmax(dim=1) == y[start:start + batch_size]).sum().item()
    model.train(was_training)
    return correct / len(x)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not (ROOT / "data" / "results" / "flow" / "0000" / "000" / "best_chkpt").exists():
        raise FileNotFoundError(
            "Pretrained FlyVis model not found. Run download_data.ipynb first "
            "or extract data/results_pretrained_models.zip into data/."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"FlyVis {flyvis.__version__} | data root: {flyvis.root_dir}")
    print(f"Torch {torch.__version__} | device: {DEVICE}")

    network_view = NetworkView("flow/0000/000")
    net = network_view.init_network(checkpoint="best")
    cell_types = load_cell_types(net, args.cell_types)
    feature_maps = build_feature_maps(net, cell_types, DEVICE)
    grid, n_hexals = build_hex_encoder(network_view, DEVICE)
    print(f"Cell types: {len(cell_types)} | hexals: {n_hexals}")

    transform = transforms.ToTensor()
    train_dataset = MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_dataset = MNIST(args.data_dir, train=False, download=True, transform=transform)

    train_x, train_y = get_features(
        args, train_dataset, args.train_size, "train", net, grid, feature_maps, DEVICE
    )
    test_x, test_y = get_features(
        args, test_dataset, args.test_size, "test", net, grid, feature_maps, DEVICE
    )

    # Split the training pool into train/validation.
    n_total = len(train_x)
    val_size = min(args.val_size, max(1, n_total // 5))
    val_size = max(1, val_size)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    order = torch.randperm(n_total, generator=generator, device="cpu")
    val_idx = order[:val_size]
    train_idx = order[val_size:]
    if len(train_idx) == 0:
        raise ValueError("Training split is empty; increase --train-size or decrease --val-size.")

    x_train = train_x[train_idx].float().to(DEVICE)
    y_train = train_y[train_idx].to(DEVICE)
    x_val = train_x[val_idx].float().to(DEVICE)
    y_val = train_y[val_idx].to(DEVICE)
    x_test = test_x.float().to(DEVICE)
    y_test = test_y.to(DEVICE)

    # Standardize using only training statistics.
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True) + 1e-6
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    print(f"Features: {x_train.shape[1]} | train {len(x_train)} | "
          f"val {len(x_val)} | test {len(x_test)}")

    # Re-seed before readout construction so results are identical whether the
    # FlyVis features came from the cache or were extracted in this run.
    set_seed(args.seed)
    model = Readout(x_train.shape[1], args.hidden, args.dropout).to(DEVICE)
    n_parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    batch_size = 256

    best_val = 0.0
    best_epoch = 0
    best_state = None
    history = []
    no_improvement = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(len(x_train), device=DEVICE)
        total_loss = 0.0

        for start in range(0, len(x_train), batch_size):
            batch = permutation[start:start + batch_size]
            logits = model(x_train[batch])
            loss = F.cross_entropy(logits, y_train[batch])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        train_loss = total_loss / len(x_train)
        train_acc = evaluate(model, x_train, y_train)
        val_acc = evaluate(model, x_val, y_val)
        history.append({"epoch": epoch, "loss": train_loss,
                        "train_acc": train_acc, "val_acc": val_acc})
        print(f"epoch {epoch:3d}/{args.epochs} | loss {train_loss:.4f} | "
              f"train {train_acc:.4f} | val {val_acc:.4f}")

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

    train_acc = evaluate(model, x_train, y_train)
    val_acc = evaluate(model, x_val, y_val)
    test_acc = evaluate(model, x_test, y_test)

    model.eval()
    with torch.no_grad():
        test_logits = model(x_test)
        test_pred = test_logits.argmax(dim=1)
        confusion = torch.zeros(10, 10, dtype=torch.long)
        for target, pred in zip(y_test.cpu(), test_pred.cpu()):
            confusion[target, pred] += 1

    elapsed = time.time() - started
    metrics = {
        "flyvis_version": flyvis.__version__,
        "torch_version": torch.__version__,
        "device": str(DEVICE),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "n_cell_types": len(cell_types),
        "n_features": int(x_train.shape[1]),
        "n_parameters": int(n_parameters),
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "n_test": int(len(x_test)),
        "best_epoch": best_epoch,
        "best_val_accuracy": val_acc,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_seconds": elapsed,
        "history": history,
        "confusion_matrix": confusion.tolist(),
    }

    torch.save({
        "model_state": model.state_dict(),
        "feature_mean": mean.cpu(),
        "feature_std": std.cpu(),
        "config": metrics["config"],
        "cell_types": cell_types,
    }, args.out_dir / "readout.pt")
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    torch.save({"predictions": test_pred.cpu(),
                "labels": y_test.cpu()}, args.out_dir / "test_predictions.pt")

    print(f"Best epoch: {best_epoch} | val {val_acc:.4f} | test {test_acc:.4f}")
    print(f"Wrote {args.out_dir / 'readout.pt'} and metrics.json")


if __name__ == "__main__":
    main()
