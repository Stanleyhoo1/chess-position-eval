"""Load a trained ChessEvaluationModel checkpoint and evaluate positions.

As a library:

    from chess_eval.load_model import load_model, evaluate_fen
    model, metadata = load_model("models/chess_eval.pt")
    score = evaluate_fen(model, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -")

Scores are centipawns from White's point of view, clipped to +-2500 (forced
mates were stored as +-2500). The network itself outputs a squashed target
(see chess_eval.encoding.cp_to_target); this module inverts whichever transform the
checkpoint records ("linear" for older checkpoints, "tanh" for newer ones).

From the command line (run from the repository root):

    python -m chess_eval.load_model                         # evaluates a few sample positions
    python -m chess_eval.load_model "8/8/2B2k2/p4p2/5P1p/Pb6/1P3KP1/8 w - -"
    python -m chess_eval.load_model --model models/run1.pt "<fen>" "<fen>"
"""

import argparse
import sys

import torch

from chess_eval.encoding import encode_fen, target_to_cp
from chess_eval.model import MODEL_CLASSES, ChessEvaluationModel


def load_model(path="models/chess_eval.pt", device=None):
    """Return (model, metadata). The model is on `device` and in eval mode."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)
    # Rebuild whichever architecture the checkpoint was trained with.
    model_cls = MODEL_CLASSES.get(checkpoint.get("model_class"), ChessEvaluationModel)
    model = model_cls(**checkpoint.get("model_config", {}))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    metadata = checkpoint.get("metadata", {})
    # Remember which target transform this checkpoint was trained with so
    # evaluate_fens can convert outputs back to centipawns.
    model.target_transform = metadata.get("target_transform", "linear")
    return model, metadata


@torch.no_grad()
def evaluate_fens(model, fens):
    """Evaluate a list of FEN strings in one batch. Returns a list of floats."""
    device = next(model.parameters()).device
    encoded = [encode_fen(fen) for fen in fens]
    board = torch.stack([e[0] for e in encoded]).to(device)            # (n, 8, 8)
    side_to_move = torch.stack([e[1] for e in encoded]).to(device)     # (n, 1)
    castling = torch.stack([e[2] for e in encoded]).to(device)         # (n, 4)
    en_passant = torch.stack([e[3] for e in encoded]).to(device)       # (n,)
    output = model(board, side_to_move, castling, en_passant)          # (n, 1), target units
    transform = getattr(model, "target_transform", "linear")
    return target_to_cp(output.squeeze(1).float(), transform).cpu().tolist()   # centipawns


def evaluate_fen(model, fen):
    """Evaluate a single FEN string. Returns a float."""
    return evaluate_fens(model, [fen])[0]


SAMPLE_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",   # start position
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq -",  # 1.e4 e5 2.Nf3
    "8/8/2B2k2/p4p2/5P1p/Pb6/1P3KP1/8 w - -",                 # endgame from the setup page
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq -",   # fool's mate, Black mates
    "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - -",                       # White up a rook
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fens", nargs="*", help="FEN strings to evaluate (default: a few samples)")
    parser.add_argument("--model", default="models/chess_eval_10m_tanh_best.pt", help="checkpoint path")
    args = parser.parse_args()

    model, metadata = load_model(args.model)
    device = next(model.parameters()).device
    print(f"loaded {args.model} ({type(model).__name__}) on {device}")
    if metadata:
        shown = {k: v for k, v in metadata.items() if k != "epoch_losses"}
        print("trained:", ", ".join(f"{k}={v}" for k, v in shown.items()))

    fens = args.fens or SAMPLE_FENS
    scores = evaluate_fens(model, fens)
    print()
    for fen, score in zip(fens, scores):
        print(f"{score:+8.1f}   {fen}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
