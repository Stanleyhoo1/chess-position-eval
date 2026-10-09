"""FEN -> tensor encoding shared by training and inference, plus the mapping
between centipawns and the squashed target the network predicts.

Model inputs produced from a FEN:

    board            (8, 8) long tensor of piece IDs, row 0 = rank 8, col 0 = a-file.
                     0 = empty, 1-6 = white P N B R Q K, 7-12 = black p n b r q k.
                     Used directly as indices into the model's piece embedding.
    side_to_move     (1,)   1 = White to move, 0 = Black to move
    castling_rights  (4,)   [W kingside, W queenside, B kingside, B queenside]
    en_passant       ()     0 = none, 1-8 = a3..h3, 9-16 = a6..h6

``encode_fen`` builds these tensors for one position; ``encode_fens`` does the
same for a whole column of FENs as numpy arrays (used when loading a training
CSV). ``cp_to_target`` / ``target_to_cp`` convert between centipawns and the
network's output range.
"""

import math

import numpy as np
import torch


# ---------------------------------------------------------------------------
# FEN encoding
# ---------------------------------------------------------------------------

PIECE_IDS = {
    "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
    "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12,
}
FILES = "abcdefgh"

# Evaluations in the CSV are centipawns clipped to +-2500. The network never
# predicts raw centipawns; it predicts a *target* in [-1, 1] and load_model.py
# inverts the mapping so scores are reported in centipawns again.
#
#   "linear": target = cp / 2500. Equal weight per centipawn, so the loss is
#             dominated by blowouts and mates.
#   "tanh":   target = tanh(cp / 543), the curve Lichess uses to turn an
#             evaluation into a win probability. It stretches the range that
#             decides most games (|cp| <= 300 maps to |target| <= 0.5) and
#             compresses everything beyond a few pawns, so the model is
#             rewarded for telling +0.5 from +1.5 rather than +15 from +25.
EVAL_SCALE = 2500.0
TANH_SCALE = 543.0
TARGET_TRANSFORMS = ("tanh", "linear")
_TANH_MAX = math.tanh(EVAL_SCALE / TANH_SCALE)      # target of a +2500 position


def cp_to_target(cp, transform="tanh"):
    """Centipawn tensor -> training target tensor in [-1, 1]."""
    if transform == "linear":
        return cp / EVAL_SCALE
    if transform == "tanh":
        return torch.tanh(cp / TANH_SCALE)
    raise ValueError(f"unknown target transform {transform!r}")


def target_to_cp(target, transform="tanh"):
    """Inverse of cp_to_target. Outputs are clipped to +-EVAL_SCALE."""
    if transform == "linear":
        return (target * EVAL_SCALE).clamp(-EVAL_SCALE, EVAL_SCALE)
    if transform == "tanh":
        return TANH_SCALE * torch.atanh(target.clamp(-_TANH_MAX, _TANH_MAX))
    raise ValueError(f"unknown target transform {transform!r}")


def _board_squares(board_field):
    """FEN board field -> flat list of 64 piece IDs in FEN order (a8 .. h1)."""
    ranks = board_field.split("/")
    if len(ranks) != 8:
        raise ValueError(f"bad board field: {board_field!r}")
    squares = []
    for rank in ranks:
        n = len(squares)
        for ch in rank:
            if ch.isdigit():
                squares.extend((0,) * int(ch))
            else:
                squares.append(PIECE_IDS[ch])
        if len(squares) - n != 8:
            raise ValueError(f"bad board field: {board_field!r}")
    return squares


def fen_to_board(fen):
    """FEN (or just its board field) -> (8, 8) long tensor of piece IDs.

    Row 0 is rank 8 and column 0 is the a-file, i.e. the board as printed from
    White's side. The values are embedding indices, not magnitudes.
    """
    return torch.tensor(_board_squares(fen.split()[0]), dtype=torch.long).view(8, 8)


def encode_en_passant(ep_field):
    """'-' -> 0, a3..h3 -> 1..8, a6..h6 -> 9..16."""
    if ep_field == "-":
        return 0
    file_idx = FILES.index(ep_field[0])
    rank = ep_field[1]
    if rank == "3":
        return 1 + file_idx
    if rank == "6":
        return 9 + file_idx
    raise ValueError(f"bad en passant square: {ep_field!r}")


def _encode_fen_parts(fen):
    """FEN -> (64 piece IDs, side_to_move, [4 castling flags], en_passant) as plain ints."""
    parts = fen.split()
    squares = _board_squares(parts[0])
    side_to_move = 1 if parts[1] == "w" else 0
    castling_field = parts[2] if len(parts) > 2 else "-"
    castling = [int(c in castling_field) for c in "KQkq"]
    en_passant = encode_en_passant(parts[3]) if len(parts) > 3 else 0
    return squares, side_to_move, castling, en_passant


def encode_fen(fen):
    """FEN -> (board, side_to_move, castling_rights, en_passant) tensors with
    the per-sample shapes documented at the top of this file."""
    squares, side_to_move, castling, en_passant = _encode_fen_parts(fen)
    return (
        torch.tensor(squares, dtype=torch.long).view(8, 8),
        torch.tensor([side_to_move], dtype=torch.long),
        torch.tensor(castling, dtype=torch.long),
        torch.tensor(en_passant, dtype=torch.long),
    )


# Vectorized encoding of many FENs at once. Each FEN board field is expanded
# with str.translate so digits become runs of "0" and "/" disappears, giving
# exactly 64 characters per position; a 256-entry lookup table then maps the
# characters to piece IDs in one numpy operation.
_EXPAND_TABLE = str.maketrans({**{str(n): "0" * n for n in range(1, 9)}, "/": ""})
_PIECE_LUT = np.zeros(256, dtype=np.uint8)
for _ch, _pid in PIECE_IDS.items():
    _PIECE_LUT[ord(_ch)] = _pid
_EP_IDS = {f"{f}3": 1 + i for i, f in enumerate(FILES)}
_EP_IDS.update({f"{f}6": 9 + i for i, f in enumerate(FILES)})


def encode_fens(fens):
    """Encode a sequence of FEN strings into compact numpy arrays:
    board (n, 8, 8) uint8, side_to_move (n, 1) uint8, castling (n, 4) uint8,
    en_passant (n,) uint8. Same encoding as encode_fen, ~100x faster."""
    import pandas as pd   # only needed here; keeps the inference service free of pandas

    parts = pd.Series(fens, dtype="string").str.split(" ", expand=True)
    n = len(parts)

    expanded = parts[0].str.translate(_EXPAND_TABLE)
    if not (expanded.str.len() == 64).all():
        bad = parts[0][expanded.str.len() != 64].iloc[0]
        raise ValueError(f"bad board field: {bad!r}")
    board = _PIECE_LUT[np.frombuffer("".join(expanded).encode("ascii"), dtype=np.uint8)]
    board = board.reshape(n, 8, 8)

    side = (parts[1] == "w").to_numpy(dtype=np.uint8).reshape(n, 1)

    castling_field = parts[2].fillna("-") if parts.shape[1] > 2 else pd.Series(["-"] * n, dtype="string")
    castling = np.stack([castling_field.str.contains(c, regex=False).to_numpy(dtype=np.uint8)
                         for c in "KQkq"], axis=1)

    ep_field = parts[3].fillna("-") if parts.shape[1] > 3 else pd.Series(["-"] * n, dtype="string")
    en_passant = ep_field.map(_EP_IDS).fillna(0).to_numpy(dtype=np.uint8)

    return board, side, castling, en_passant
