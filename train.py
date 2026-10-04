"""Training setup for ChessEvaluationModel.

Defines the loss (MSE) and optimizer (Adam, lr=0.001) and a minimal training
loop. Real dataset loading / preprocessing is not implemented here; the loop
works with any DataLoader whose batches are tuples of

    (board, side_to_move, castling_rights, en_passant, target)

with the input shapes documented in model.py and target of shape (batch, 1).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import ChessEvaluationModel


def build_training_components(model):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )
    return criterion, optimizer


def train_step(model, criterion, optimizer, batch, device):
    """One optimizer update on a single batch. Returns the batch loss as a float."""
    board, side_to_move, castling_rights, en_passant, target = (t.to(device) for t in batch)

    model.train()
    optimizer.zero_grad()
    prediction = model(board, side_to_move, castling_rights, en_passant)   # (batch, 1)
    loss = criterion(prediction, target.float())                            # scalar
    loss.backward()
    optimizer.step()
    return loss.item()


def train(model, dataloader, epochs, device, log_every=1):
    """Train for `epochs` passes over `dataloader`. Returns a list of mean epoch losses."""
    criterion, optimizer = build_training_components(model)
    model.to(device)

    epoch_losses = []
    for epoch in range(1, epochs + 1):
        total, count = 0.0, 0
        for batch in dataloader:
            total += train_step(model, criterion, optimizer, batch, device)
            count += 1
        mean_loss = total / max(count, 1)
        epoch_losses.append(mean_loss)
        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch:>3}/{epochs}  loss {mean_loss:.4f}")
    return epoch_losses


if __name__ == "__main__":
    # Smoke test: fit random positions with random targets. The loss should
    # fall as the network memorizes the dummy batch, which proves the loss,
    # optimizer and backward pass are wired correctly.
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    n = 64
    dataset = TensorDataset(
        torch.randint(0, 13, (n, 8, 8)),       # board
        torch.randint(0, 2, (n, 1)),           # side_to_move
        torch.randint(0, 2, (n, 4)),           # castling_rights
        torch.randint(0, 17, (n,)),            # en_passant
        torch.randn(n, 1),                     # target evaluation
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ChessEvaluationModel()
    losses = train(model, dataloader, epochs=20, device=device, log_every=5)

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"OK: loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")
