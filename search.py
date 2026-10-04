"""Capture search (quiescence search) on top of the neural evaluator.

The network is a *static* evaluator: it looks at a position and guesses the
engine score without playing any moves, so it cannot see that a piece is
hanging. Real engines never ask a static evaluator to resolve that; they run
a quiescence search first, playing out captures (and check evasions) until
the position is quiet, and only then call the evaluator at the leaves.

This module does exactly that with python-chess for move generation and the
trained network for leaf evaluation. Scores are centipawns from White's point
of view, like the network's own output.

As a library:

    from load_model import load_model
    from search import Searcher
    model, _ = load_model("models/chess_eval_10m_best.pt")
    result = Searcher(model).evaluate("rnbqkb1r/pppppppp/8/8/4n3/3P4/PPP1PPPP/RNBQKBNR w KQkq -")
    result.score, result.static, result.line     # e.g. 310.0, 26.0, "1. dxe4"

From the command line:

    python search.py "<fen>" ["<fen>" ...]
    python search.py --model models/chess_eval_10m_best.pt --depth 8 "<fen>"
"""

import argparse
import sys
import time
from dataclasses import dataclass

import chess

from load_model import evaluate_fen, load_model

MATE_SCORE = 2500.0          # same cap the training targets used for forced mates
INF = float("inf")

# Piece values for move ordering only (most valuable victim first).
_ORDER_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


@dataclass
class SearchResult:
    score: float        # centipawns from White's point of view after the capture search
    static: float       # the network's score of the root position alone
    line: str           # principal variation in SAN, "" if standing pat
    nodes: int
    seconds: float


class Searcher:
    """Alpha-beta quiescence search using a ChessEvaluationModel/ResNet for leaves."""

    def __init__(self, model, max_depth=6, max_nodes=4000):
        self.model = model
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self._cache = {}
        self.nodes = 0

    # ---- static evaluation -------------------------------------------------

    def static_eval(self, board):
        """Network score of `board`, White's perspective, cached per position."""
        key = board._transposition_key()
        score = self._cache.get(key)
        if score is None:
            fen = " ".join(board.fen().split()[:4])      # drop the move counters
            score = evaluate_fen(self.model, fen)
            self._cache[key] = score
        return score

    # ---- search -----------------------------------------------------------

    def _ordered_moves(self, board, in_check):
        if in_check:
            moves = list(board.legal_moves)             # all evasions
        else:
            moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]

        def priority(move):
            victim = board.piece_type_at(move.to_square)
            victim_value = _ORDER_VALUE[victim] if victim else (1 if board.is_en_passant(move) else 0)
            attacker_value = _ORDER_VALUE[board.piece_type_at(move.from_square)]
            promo = _ORDER_VALUE[move.promotion] if move.promotion else 0
            return -(victim_value * 10 - attacker_value + promo * 10)

        moves.sort(key=priority)
        return moves

    def _qsearch(self, board, alpha, beta, depth):
        """Negamax quiescence search. Returns (score for the side to move, pv moves)."""
        self.nodes += 1

        if board.is_checkmate():
            return -MATE_SCORE, []
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0, []

        side = 1.0 if board.turn == chess.WHITE else -1.0
        in_check = board.is_check()

        if not in_check:
            stand_pat = side * self.static_eval(board)
            if depth <= 0 or self.nodes >= self.max_nodes:
                return stand_pat, []
            if stand_pat >= beta:
                return stand_pat, []
            alpha = max(alpha, stand_pat)
            best, best_line = stand_pat, []
        else:
            if depth <= 0 or self.nodes >= self.max_nodes:
                return side * self.static_eval(board), []
            best, best_line = -INF, []

        for move in self._ordered_moves(board, in_check):
            board.push(move)
            score, line = self._qsearch(board, -beta, -alpha, depth - 1)
            board.pop()
            score = -score
            if score > best:
                best, best_line = score, [move] + line
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best, best_line

    def evaluate(self, fen_or_board):
        """Run the capture search from a FEN string or chess.Board."""
        board = chess.Board(fen_or_board) if isinstance(fen_or_board, str) else fen_or_board.copy()
        started = time.time()
        self.nodes = 0
        static = self.static_eval(board)
        side = 1.0 if board.turn == chess.WHITE else -1.0
        score, line = self._qsearch(board, -INF, INF, self.max_depth)
        return SearchResult(
            score=side * score,
            static=static,
            line=board.variation_san(line) if line else "",
            nodes=self.nodes,
            seconds=time.time() - started,
        )


SAMPLE_FENS = [
    "r1bq1rk1/ppp2ppp/2np4/4p3/2B1n3/2NP1N2/PPP2PPP/R1BQ1RK1 w - -",  # middlegame: black knight hangs on e4
    "r1bq1rk1/ppp2ppp/2np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - -", # same position, knight safe on f6
    "rnbqkb1r/pppppppp/8/8/4n3/3P4/PPP1PPPP/RNBQKBNR w KQkq -",      # opening: black knight hangs to dxe4
    "rnbqkbnr/pppp1ppp/4p3/3Q4/8/8/PPPPPPPP/RNB1KBNR b KQkq -",      # white queen hangs to exd5
    "rnbqkbnr/ppp1pppp/8/3p4/2B5/8/PPPPPPPP/RNBQK1NR b KQkq -",      # white bishop hangs to dxc4
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq -",     # fool's mate: White is mated
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",          # start position (quiet)
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fens", nargs="*", help="FEN strings to evaluate (default: a few tactical samples)")
    parser.add_argument("--model", default="models/chess_eval_10m_tanh_best.pt", help="checkpoint path")
    parser.add_argument("--depth", type=int, default=6, help="maximum capture-sequence length in plies")
    parser.add_argument("--max-nodes", type=int, default=4000, help="node budget per position")
    args = parser.parse_args()

    model, _ = load_model(args.model)
    searcher = Searcher(model, max_depth=args.depth, max_nodes=args.max_nodes)
    print(f"loaded {args.model} ({type(model).__name__})")
    print(f"{'static':>8}  {'search':>8}  {'nodes':>5}  {'time':>6}  line / fen")
    for fen in args.fens or SAMPLE_FENS:
        r = searcher.evaluate(fen)
        print(f"{r.static:+8.0f}  {r.score:+8.0f}  {r.nodes:>5}  {r.seconds:5.2f}s  {r.line or '(stand pat)'}")
        print(f"{'':35}{fen}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
