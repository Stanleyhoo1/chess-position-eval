"""Training setup for ChessEvaluationModel.

Defines the loss (Huber on a squashed target), the optimizer (Adam with a
warmup + cosine learning-rate decay), a minimal training
loop, and the dataset loader that turns the training CSV (columns
``fen,evaluation``, written by get_training_data.py) into model inputs using
the FEN encoding in chess_eval/encoding.py.

The finished model is saved as a checkpoint (``--out``, default
models/chess_eval.pt) that chess_eval.load_model can reload.

How much training happens:
    --epochs N     N full passes over the data (default 20). Each epoch does
                   ceil(rows / batch_size) optimizer updates.
    --updates N    stop after exactly N optimizer updates instead, regardless
                   of where that falls in an epoch.

Two architectures are available (see chess_eval/model.py): ``--model small`` is the
original two-convolution network, ``--model large`` (default) is the deeper
residual network. The checkpoint records which one was used.

Usage:
Run from the repository root:

    python -m training.train                          # all rows of data/training_data.csv, 20 epochs
    python -m training.train --limit 100 --updates 100   # quick test: first 100 rows, 100 updates
    python -m training.train --epochs 5 --batch-size 256 --out models/run1.pt
"""

import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from chess_eval.model import MODEL_CLASSES, ChessEvaluationModel, ChessEvaluationResNet
from chess_eval.encoding import EVAL_SCALE, TANH_SCALE, TARGET_TRANSFORMS, cp_to_target, target_to_cp, encode_fen, encode_fens


def load_dataset(csv_path, limit=None, use_cache=True, transform="tanh"):
    """Read a ``fen,evaluation`` CSV into a TensorDataset of
    (board, side_to_move, castling_rights, en_passant, target).

    Boards and metadata are uint8 (about 70 bytes per position, so 10M rows
    fit in ~0.7 GB); the model converts them to long/float itself. Targets
    are float32 in [-1, 1] (see cp_to_target; `transform` picks the mapping).

    The encoded tensors are cached next to the CSV (<csv>.encoded.pt, or
    <csv>.first<limit>.encoded.pt when --limit is used) and
    reused while the CSV is unchanged, so only the first run pays for encoding.
    """
    cache_path = csv_path + (".encoded.pt" if limit is None else f".first{limit}.encoded.pt")
    stat = os.stat(csv_path)
    # The cache stores raw centipawns so any transform can be applied on load.
    cache_key = {"size": stat.st_size, "mtime": stat.st_mtime, "limit": limit, "target": "centipawns"}
    if use_cache and os.path.exists(cache_path):
        cached = torch.load(cache_path)
        if cached.get("key") == cache_key:
            *inputs, cp = cached["tensors"]
            return TensorDataset(*inputs, cp_to_target(cp, transform))

    started = time.time()
    df = pd.read_csv(csv_path, usecols=["fen", "evaluation"], nrows=limit,
                     dtype={"fen": "string", "evaluation": "float32"})
    if len(df) == 0:
        raise ValueError(f"no rows read from {csv_path}")
    board, side, castling, en_passant = encode_fens(df["fen"])
    cp = df["evaluation"].to_numpy(dtype=np.float32).reshape(-1, 1).copy()   # writable for torch

    tensors = (
        torch.from_numpy(board),          # (n, 8, 8) uint8
        torch.from_numpy(side),           # (n, 1)    uint8
        torch.from_numpy(castling),       # (n, 4)    uint8
        torch.from_numpy(en_passant),     # (n,)      uint8
        torch.from_numpy(cp),             # (n, 1)    float32 centipawns
    )
    print(f"encoded {len(df):,} positions in {time.time() - started:.1f}s")
    if use_cache:
        torch.save({"key": cache_key, "tensors": tensors}, cache_path)
    *inputs, cp = tensors
    return TensorDataset(*inputs, cp_to_target(cp, transform))


class Batches:
    """Shuffled mini-batches drawn by indexing tensors that live on the
    training device. For small fixed-size samples this is far faster than a
    DataLoader, which collates Python objects one sample at a time."""

    def __init__(self, dataset, batch_size, device, shuffle=True):
        self.tensors = [t.to(device) for t in dataset.tensors]
        self.n = len(dataset)
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

    def __len__(self):
        return -(-self.n // self.batch_size)

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(self.n, device=self.device)
        else:
            order = torch.arange(self.n, device=self.device)
        for i in range(0, self.n, self.batch_size):
            idx = order[i:i + self.batch_size]
            yield tuple(t[idx] for t in self.tensors)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_training_components(model, lr=0.001, loss="huber", huber_delta=0.2):
    """Loss and optimizer.

    "mse":   squared error everywhere; a few huge misses dominate the gradient.
    "huber": squared error for |error| <= delta, absolute error beyond it, so
             outliers (mislabelled or mate positions) pull with bounded force.
             delta is in target units; 0.2 is about a 110 cp miss on a
             balanced position under the tanh transform.
    """
    if loss == "mse":
        criterion = nn.MSELoss()
    elif loss == "huber":
        criterion = nn.HuberLoss(delta=huber_delta)
    else:
        raise ValueError(f"unknown loss {loss!r}")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )
    return criterion, optimizer


def make_lr_schedule(kind, total_updates, warmup=0):
    """Return f(update_index) -> learning-rate multiplier in [0, 1].

    "constant": always 1.
    "cosine":   linear warmup from 0 over `warmup` updates, then cosine decay
                from 1 to 0 over the remaining updates, so the final updates
                take tiny steps that settle the weights instead of bouncing.
    """
    if kind == "constant":
        return lambda t: 1.0
    if kind == "cosine":
        decay_len = max(1, total_updates - warmup)

        def factor(t):
            if warmup and t < warmup:
                return (t + 1) / warmup
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (t - warmup) / decay_len)))
        return factor
    raise ValueError(f"unknown schedule {kind!r}")


def train_step(model, criterion, optimizer, batch, device, amp=False, transform="tanh"):
    """One optimizer update on a single batch.

    Returns (loss, squared_cp_error_sum, batch_rows): the loss in target units
    plus the ingredients for a true centipawn RMSE after inverting the target
    transform. With amp=True the forward pass runs in bfloat16 autocast (the
    loss and the weights stay float32), which is much faster on tensor-core GPUs.
    """
    board, side_to_move, castling_rights, en_passant, target = (t.to(device) for t in batch)

    model.train()
    optimizer.zero_grad()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        prediction = model(board, side_to_move, castling_rights, en_passant)   # (batch, 1)
    prediction = prediction.float()
    loss = criterion(prediction, target.float())                                # scalar
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        cp_err = target_to_cp(prediction.detach(), transform) - target_to_cp(target.float(), transform)
        sq_cp = (cp_err ** 2).sum().item()
    return loss.item(), sq_cp, target.shape[0]


@torch.no_grad()
def evaluate(model, batches, amp=False, transform="tanh", criterion=None):
    """Returns (mean loss in target units, RMSE in centipawns) over all batches.
    `criterion` defaults to MSE; pass the training criterion to match it."""
    model.eval()
    device = next(model.parameters()).device
    if criterion is None:
        criterion = nn.MSELoss(reduction="sum")
    else:
        criterion = type(criterion)(**({"delta": criterion.delta} if hasattr(criterion, "delta") else {}),
                                    reduction="sum")
    total, total_cp, count = 0.0, 0.0, 0
    for batch in batches:
        board, side_to_move, castling_rights, en_passant, target = (t.to(device) for t in batch)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            prediction = model(board, side_to_move, castling_rights, en_passant)
        prediction, target = prediction.float(), target.float()
        total += criterion(prediction, target).item()
        total_cp += ((target_to_cp(prediction, transform) - target_to_cp(target, transform)) ** 2).sum().item()
        count += target.shape[0]
    return total / max(count, 1), (total_cp / max(count, 1)) ** 0.5


def train(model, dataloader, epochs, device, log_every=1, max_updates=None,
          val_batches=None, amp=False, on_epoch_end=None, transform="tanh",
          lr=0.001, lr_schedule=None, loss="huber", huber_delta=0.2):
    """Train for `epochs` passes over `dataloader`, or until `max_updates`
    optimizer steps have been taken if that comes first.

    If `val_batches` is given, the validation loss is computed after every
    epoch. `on_epoch_end(epoch, train_loss, val_loss)` is called after each
    epoch (val_loss is None without validation data).

    `lr_schedule(update_index)` (see make_lr_schedule) scales `lr` before
    every update; None means a constant learning rate.

    Returns a dict with per-epoch lists "loss", "rmse_cp", "val_loss",
    "val_rmse_cp" and the total "updates". Losses are in target units; the
    rmse entries are true centipawn errors after inverting the transform.
    """
    criterion, optimizer = build_training_components(model, lr=lr, loss=loss, huber_delta=huber_delta)
    model.to(device)

    history = {"loss": [], "rmse_cp": [], "val_loss": [], "val_rmse_cp": [], "updates": 0}
    updates = 0
    for epoch in range(1, epochs + 1):
        total, total_cp, rows, count = 0.0, 0.0, 0, 0
        epoch_started = time.time()
        for batch in dataloader:
            if lr_schedule is not None:
                current_lr = lr * lr_schedule(updates)
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
            loss, sq_cp, n = train_step(model, criterion, optimizer, batch, device, amp=amp, transform=transform)
            total += loss
            total_cp += sq_cp
            rows += n
            count += 1
            updates += 1
            if max_updates is not None and updates >= max_updates:
                break
        mean_loss = total / max(count, 1)
        rmse = (total_cp / max(rows, 1)) ** 0.5
        history["loss"].append(mean_loss)
        history["rmse_cp"].append(rmse)
        history["updates"] = updates

        val_loss = None
        if val_batches is not None:
            val_loss, val_rmse = evaluate(model, val_batches, amp=amp, transform=transform, criterion=criterion)
            history["val_loss"].append(val_loss)
            history["val_rmse_cp"].append(val_rmse)

        if log_every and (epoch % log_every == 0 or epoch == epochs):
            val_text = f"  val {val_loss:.4f} (rmse {val_rmse:5.0f} cp)" if val_loss is not None else ""
            lr_text = f"  lr {optimizer.param_groups[0]['lr']:.2e}"
            print(f"epoch {epoch:>3}/{epochs}  updates {updates:>6}  "
                  f"loss {mean_loss:.4f} (rmse {rmse:5.0f} cp){val_text}{lr_text}  "
                  f"{time.time() - epoch_started:6.1f}s", flush=True)
        if on_epoch_end is not None:
            on_epoch_end(epoch, mean_loss, val_loss)
        if max_updates is not None and updates >= max_updates:
            break
    return history


def save_model(model, path, metadata=None):
    """Save the model weights plus training metadata as a checkpoint dict.

    Only the state_dict is stored (not the pickled class), so the checkpoint
    stays loadable after chess_eval/model.py changes as long as the layer names match.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": type(model).__name__,
        "model_config": model.config() if hasattr(model, "config") else {},
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/training_data.csv", help="fen,evaluation CSV to train on")
    parser.add_argument("--limit", type=int, help="only use the first N rows of --data")
    parser.add_argument("--epochs", type=int, default=20, help="full passes over the data")
    parser.add_argument("--updates", type=int, default=None,
                        help="stop after this many optimizer updates (overrides --epochs as the stopping point)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", default="models/chess_eval.pt", help="where to save the trained model")
    parser.add_argument("--model", default="large", choices=["small", "large"],
                        help="small = original 2-conv ChessEvaluationModel; large = residual ChessEvaluationResNet")
    parser.add_argument("--blocks", type=int, default=6, help="residual blocks (large model only)")
    parser.add_argument("--channels", type=int, default=128, help="conv channels (large model only)")
    parser.add_argument("--data-device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="where the dataset tensors live; auto keeps them on the GPU when they fit")
    parser.add_argument("--no-cache", action="store_true", help="re-encode the CSV even if a cache exists")
    parser.add_argument("--val-fraction", type=float, default=0.02,
                        help="fraction of rows held out for validation (0 disables)")
    parser.add_argument("--no-amp", action="store_true",
                        help="disable bfloat16 mixed precision (on by default when training on CUDA)")
    parser.add_argument("--target", default="tanh", choices=TARGET_TRANSFORMS,
                        help="how centipawns are mapped to the training target (see cp_to_target)")
    parser.add_argument("--loss", default="huber", choices=["huber", "mse"], help="training loss")
    parser.add_argument("--huber-delta", type=float, default=0.2,
                        help="Huber threshold in target units (errors beyond it count linearly)")
    parser.add_argument("--lr", type=float, default=0.001, help="peak learning rate")
    parser.add_argument("--schedule", default="cosine", choices=["cosine", "constant"],
                        help="learning-rate schedule over the whole run")
    parser.add_argument("--warmup", type=int, default=1000,
                        help="updates of linear warmup before the cosine decay starts")
    args = parser.parse_args()
    amp = device_is_cuda = torch.cuda.is_available() and not args.no_amp

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Sanity-check the encoder on the start position.
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board, side, castling, ep = encode_fen(start)
    assert board.shape == (8, 8) and board[0, 0] == 10 and board[7, 4] == 6 and board[3, 3] == 0
    assert side.tolist() == [1] and castling.tolist() == [1, 1, 1, 1] and ep.item() == 0

    dataset = load_dataset(args.data, limit=args.limit, use_cache=not args.no_cache, transform=args.target)
    data_bytes = sum(t.numel() * t.element_size() for t in dataset.tensors)
    if args.data_device == "auto":
        if device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info()
            data_device = device if data_bytes < 0.5 * free_bytes else torch.device("cpu")
        else:
            data_device = device
    else:
        data_device = torch.device(args.data_device)
    # Hold out the last val_fraction of rows (the CSV is already a random
    # sample, so the tail is a random subset).
    n_val = int(len(dataset) * args.val_fraction) if args.val_fraction > 0 else 0
    n_train = len(dataset) - n_val
    train_set = TensorDataset(*[t[:n_train] for t in dataset.tensors])
    val_set = TensorDataset(*[t[n_train:] for t in dataset.tensors]) if n_val else None

    dataloader = Batches(train_set, batch_size=args.batch_size, device=data_device)
    val_batches = Batches(val_set, batch_size=4096, device=data_device, shuffle=False) if val_set else None
    print(f"dataset tensors: {data_bytes / 1e6:.0f} MB on {data_device}; "
          f"{n_train:,} train / {n_val:,} validation rows; mixed precision {'on' if amp else 'off'}")

    updates_per_epoch = len(dataloader)
    if args.updates is not None:
        epochs = -(-args.updates // updates_per_epoch)          # ceil: enough epochs to reach the target
        planned_updates = args.updates
    else:
        epochs = args.epochs
        planned_updates = epochs * updates_per_epoch
    print(f"loaded {len(dataset):,} positions from {args.data}")
    print(f"batch size {args.batch_size} -> {updates_per_epoch} updates per epoch; "
          f"training for {planned_updates} updates ({epochs} epoch{'s' if epochs != 1 else ''})")
    print(f"target transform: {args.target}; loss {args.loss}"
          + (f" (delta {args.huber_delta})" if args.loss == "huber" else "")
          + f"; lr {args.lr} with {args.schedule} schedule"
          + (f" ({args.warmup} warmup updates)" if args.schedule == "cosine" else ""))
    lr_schedule = make_lr_schedule(args.schedule, planned_updates, warmup=args.warmup)

    if args.model == "small":
        model = ChessEvaluationModel()
    else:
        model = ChessEvaluationResNet(blocks=args.blocks, channels=args.channels)
    print(f"model: {type(model).__name__} with {sum(p.numel() for p in model.parameters()):,} parameters")
    started = time.time()
    best_path = os.path.splitext(args.out)[0] + "_best.pt"
    best = {"val": None, "epoch": None}

    target_meta = {"target_transform": args.target, "eval_scale": EVAL_SCALE, "tanh_scale": TANH_SCALE}

    def on_epoch_end(epoch, train_loss, val_loss):
        # Keep the checkpoint with the lowest validation loss as <out>_best.pt.
        if val_loss is not None and (best["val"] is None or val_loss < best["val"]):
            best["val"], best["epoch"] = val_loss, epoch
            save_model(model, best_path, metadata={"epoch": epoch, "train_loss": train_loss,
                                                   "val_loss": val_loss, "model": type(model).__name__,
                                                   **target_meta})

    hist = train(
        model, dataloader, epochs=epochs, device=device, log_every=max(1, epochs // 30),
        max_updates=args.updates, val_batches=val_batches, amp=amp, on_epoch_end=on_epoch_end,
        transform=args.target, lr=args.lr, lr_schedule=lr_schedule,
        loss=args.loss, huber_delta=args.huber_delta)
    train_seconds = time.time() - started
    losses, val_losses = hist["loss"], hist["val_loss"]
    updates_done = hist["updates"]
    print(f"finished: {updates_done} updates in {train_seconds:.1f}s, "
          f"loss {losses[0]:.4f} -> {losses[-1]:.4f}  "
          f"(rmse {hist['rmse_cp'][0]:.0f} -> {hist['rmse_cp'][-1]:.0f} cp)")
    if val_losses:
        best_idx = val_losses.index(best["val"])
        print(f"validation: rmse {hist['val_rmse_cp'][0]:.0f} -> {hist['val_rmse_cp'][-1]:.0f} cp; "
              f"best {hist['val_rmse_cp'][best_idx]:.0f} cp at epoch {best['epoch']} (saved to {best_path})")

    save_model(model, args.out, metadata={
        "data": args.data,
        "rows": len(dataset),
        "epochs": len(losses),
        "updates": updates_done,
        "batch_size": args.batch_size,
        "model": type(model).__name__,
        **target_meta,
        "loss": args.loss,
        "huber_delta": args.huber_delta if args.loss == "huber" else None,
        "lr": args.lr,
        "schedule": args.schedule,
        "warmup": args.warmup,
        "final_loss": losses[-1],
        "epoch_losses": losses,
        "epoch_rmse_cp": hist["rmse_cp"],
        "val_losses": val_losses,
        "val_rmse_cp": hist["val_rmse_cp"],
        "val_rows": n_val,
        "amp": amp,
        "train_seconds": round(train_seconds, 1),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(f"saved model to {args.out}")
