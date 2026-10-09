"""Evaluate a saved checkpoint on a held-out test CSV and report errors in
centipawns.

The test set must be positions the model never trained or validated on
(see get_training_data.py --skip-per-group). The network runs in float32
with no autocast, its output is inverted from the checkpoint's target
transform back to centipawns, and the metrics are:

    MAE          mean |prediction - engine|
    RMSE         sqrt(mean (prediction - engine)^2)
    median       median |prediction - engine|
    side agreement
                 both prediction and engine put into {Black better, equal,
                 White better} with a dead zone of +-DEAD_ZONE cp around 0;
                 reported as the share of positions where the classes match.
                 Also reported: sign accuracy on decisive positions only
                 (|engine| > DEAD_ZONE), ignoring the prediction's dead zone.

Usage:
    python evaluate_model.py                                   # default model on data/test_data.csv
    python evaluate_model.py --model models/chess_eval_10m_best.pt --data data/test_data.csv
"""

import argparse
import sys

import numpy as np
import pandas as pd
import torch

from load_model import load_model
from train import encode_fens, target_to_cp

DEAD_ZONE = 50.0     # centipawns; |eval| <= this counts as "equal"


def predict_cp(model, fens, batch_size=4096):
    """Network scores in centipawns (White's view) for a sequence of FENs."""
    board, side, castling, ep = encode_fens(fens)
    device = next(model.parameters()).device
    transform = getattr(model, "target_transform", "linear")
    out = []
    with torch.inference_mode():
        for i in range(0, len(fens), batch_size):
            sl = slice(i, i + batch_size)
            pred = model(torch.from_numpy(board[sl]).to(device), torch.from_numpy(side[sl]).to(device),
                         torch.from_numpy(castling[sl]).to(device), torch.from_numpy(ep[sl]).to(device))
            out.append(target_to_cp(pred.float().squeeze(1), transform).cpu().numpy())
    return np.concatenate(out)


def side_class(cp, dead_zone=DEAD_ZONE):
    """-1 = Black better, 0 = equal, +1 = White better."""
    return np.where(cp > dead_zone, 1, np.where(cp < -dead_zone, -1, 0))


def metrics(pred, truth, dead_zone=DEAD_ZONE):
    err = pred - truth
    decisive = np.abs(truth) > dead_zone
    return {
        "n": len(truth),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "median": float(np.median(np.abs(err))),
        "agreement": float((side_class(pred, dead_zone) == side_class(truth, dead_zone)).mean()),
        "sign_decisive": float((np.sign(pred[decisive]) == np.sign(truth[decisive])).mean()) if decisive.any() else float("nan"),
        "decisive_share": float(decisive.mean()),
    }


def print_metrics(name, m):
    print(f"{name:28s} n={m['n']:>7,}  MAE {m['mae']:6.1f} cp  RMSE {m['rmse']:6.1f} cp  "
          f"median {m['median']:5.1f} cp  side agreement {m['agreement']*100:5.1f}%  "
          f"sign on decisive {m['sign_decisive']*100:5.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/chess_eval_10m_tanh_best.pt")
    parser.add_argument("--data", default="data/test_data.csv")
    parser.add_argument("--dead-zone", type=float, default=DEAD_ZONE, help="cp band around 0 treated as equal")
    args = parser.parse_args()

    df = pd.read_csv(args.data, dtype={"fen": "string", "evaluation": "float32"})
    model, meta = load_model(args.model)
    truth = df["evaluation"].to_numpy(dtype=np.float64)
    pred = predict_cp(model, df["fen"]).astype(np.float64)

    print(f"model {args.model} ({type(model).__name__}, target {getattr(model, 'target_transform', 'linear')}) "
          f"on {args.data}; dead zone +-{args.dead_zone:.0f} cp")
    print_metrics("all positions", metrics(pred, truth, args.dead_zone))
    mate = np.abs(truth) >= 2500
    print_metrics("non-mate positions", metrics(pred[~mate], truth[~mate], args.dead_zone))
    print_metrics("forced-mate positions", metrics(pred[mate], truth[mate], args.dead_zone))
    bal = np.abs(truth) <= 100
    print_metrics("|engine| <= 100 cp", metrics(pred[bal], truth[bal], args.dead_zone))
    return 0


if __name__ == "__main__":
    sys.exit(main())
