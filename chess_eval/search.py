"""Search on top of the neural evaluator.

The network is a *static* evaluator: it looks at a position and guesses the
engine score without playing any moves, so it cannot see that a piece is
hanging or that a fork is coming. Real engines never ask a static evaluator
to do that; they search.

Two searches live here, both using python-chess for move generation and the
trained network for leaf evaluation:

  * quiescence search (Searcher.evaluate): plays out captures, promotions
    and check evasions until the position is quiet, then calls the network.
    Fixes hanging pieces; cheap (a few network calls).
  * full-width alpha-beta search (Searcher.best_move): tries *every* legal
    move for both sides to a given depth, with the quiescence search at the
    leaves, iterative deepening and a time budget. This is what sees forks,
    pins, mate threats and other quiet tactics, and what a bot should use to
    pick moves.

Network calls are batched: whenever a node is expanded, the static scores of
all its children are fetched in one forward pass, so the cost is roughly one
network call per expanded node rather than one per leaf.

Scores are centipawns from White's point of view, like the network's output.
Mates are reported as +-2500 (the training cap) plus a "mate_in" count.

Two endgame aids keep the bot from drawing won games:
  * any position that already occurred in the game (or earlier in the line
    being searched) scores as a draw, so a winning bot never repeats and a
    losing one is happy to;
  * a "mop-up" bonus when one side has a bare king rewards driving that king
    to the edge and bringing the other king closer, giving the search a
    gradient toward mate while mate itself is still beyond its horizon.

As a library:

    from chess_eval.load_model import load_model
    from chess_eval.search import Searcher
    model, _ = load_model("models/chess_eval_10m_tanh_best.pt")
    s = Searcher(model)
    s.evaluate(fen).score                      # quiescence only
    r = s.best_move(fen, depth=4, max_seconds=3)
    r.move, r.san, r.score, r.mate_in, r.line, r.depth

From the command line (run from the repository root):

    python -m chess_eval.search "<fen>" ...          # quiescence evaluation
    python -m chess_eval.search --best "<fen>" ...   # best move (full search)
"""

import argparse
import sys
import time
from dataclasses import dataclass

import chess

from chess_eval.load_model import evaluate_fens, load_model

MATE_SCORE = 2500.0          # reported score for a forced mate (training cap)
MATE_VALUE = 100000          # internal mate score; minus ply so faster mates win
INF = float("inf")

# Manhattan distance of each square from the board centre (0 at d4/e4/d5/e5, 6 in the corners).
_CENTER_DISTANCE = [abs(chess.square_file(sq) * 2 - 7) // 2 + abs(chess.square_rank(sq) * 2 - 7) // 2
                    for sq in chess.SQUARES]
MOPUP_SCALE = 8.0            # cp per unit of the classic 4.7*centre + 1.6*(14 - king distance) term

# Piece values for move ordering only (most valuable victim first).
_ORDER_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


class _TimeUp(Exception):
    pass


@dataclass
class SearchResult:
    score: float        # centipawns from White's point of view after the capture search
    static: float       # the network's score of the root position alone
    line: str           # principal variation in SAN, "" if standing pat
    nodes: int
    seconds: float


@dataclass
class MoveResult:
    move: str           # best move in UCI ("e2e4"), "" if no legal move
    san: str            # same move in SAN
    score: float        # centipawns, White's point of view, clipped to +-MATE_SCORE
    mate_in: int        # +n = White mates in n moves, -n = Black mates in n, 0 = no forced mate found
    line: str           # principal variation in SAN
    depth: int          # deepest fully completed iteration
    nodes: int
    seconds: float


def _mate_in_from_score(score, ply=0):
    """Internal side-to-move-independent score -> signed mate-in-N (0 if none)."""
    if abs(score) < MATE_VALUE - 1000:
        return 0
    plies = int(MATE_VALUE - abs(score) - ply)
    moves = (plies + 1) // 2
    return moves if score > 0 else -moves


def _clip(score):
    if score >= MATE_VALUE - 1000:
        return MATE_SCORE
    if score <= -MATE_VALUE + 1000:
        return -MATE_SCORE
    return max(-MATE_SCORE, min(MATE_SCORE, score))


class Searcher:
    """Alpha-beta search using a ChessEvaluationModel/ResNet for leaf scores."""

    def __init__(self, model, max_depth=6, max_nodes=4000):
        self.model = model
        self.max_qdepth = max_depth        # quiescence depth (plies of captures)
        self.max_nodes = max_nodes         # quiescence node budget (evaluate() only)
        self._cache = {}
        self.nodes = 0
        self._deadline = None
        self._seen = {}            # transposition key -> times seen in the game + current line

    # ---- static evaluation -------------------------------------------------

    @staticmethod
    def _fen4(board):
        return " ".join(board.fen().split()[:4])      # drop the move counters

    @staticmethod
    def mopup(board):
        """Bonus (White's perspective) for cornering a bare king, 0 otherwise."""
        white_rest = board.occupied_co[chess.WHITE] & ~board.kings
        black_rest = board.occupied_co[chess.BLACK] & ~board.kings
        if white_rest and black_rest:
            return 0.0
        if not white_rest and not black_rest:
            return 0.0
        strong = chess.WHITE if white_rest else chess.BLACK
        weak_king = board.king(not strong)
        strong_king = board.king(strong)
        if weak_king is None or strong_king is None:
            return 0.0
        king_distance = chess.square_distance(weak_king, strong_king)
        term = 4.7 * _CENTER_DISTANCE[weak_king] + 1.6 * (14 - king_distance)
        bonus = MOPUP_SCALE * term
        return bonus if strong == chess.WHITE else -bonus

    def static_eval(self, board):
        """Network score of `board` (plus the mop-up bonus), White's perspective, cached."""
        key = board._transposition_key()
        score = self._cache.get(key)
        if score is None:
            score = evaluate_fens(self.model, [self._fen4(board)])[0] + self.mopup(board)
            self._cache[key] = score
        return score

    def prefetch(self, board, moves):
        """Batch-evaluate the positions after each of `moves` into the cache.
        Returns {move: child transposition key} so callers can order moves by
        the cached child scores without pushing the moves again."""
        child_keys = {}
        keys, fens = [], []
        for move in moves:
            board.push(move)
            key = board._transposition_key()
            child_keys[move] = key
            if key not in self._cache and not board.is_game_over(claim_draw=False):
                keys.append(key)
                fens.append(self._fen4(board))
            board.pop()
        if fens:
            scores = evaluate_fens(self.model, fens)
            for move, key in child_keys.items():
                if key in keys:
                    board.push(move)
                    self._cache[key] = scores[keys.index(key)] + self.mopup(board)
                    board.pop()
        return child_keys

    # ---- move ordering -----------------------------------------------------

    def _order(self, board, moves, first=None, child_keys=None):
        """Captures by victim value first (MVV-LVA), promotions, then quiet moves
        ranked by the child's cached static score when `child_keys` is given."""
        side = 1.0 if board.turn == chess.WHITE else -1.0
        cache = self._cache

        def priority(move):
            if move == first:
                return -1e9
            victim = board.piece_type_at(move.to_square)
            victim_value = _ORDER_VALUE[victim] if victim else (1 if board.is_en_passant(move) else 0)
            attacker_value = _ORDER_VALUE[board.piece_type_at(move.from_square)]
            promo = _ORDER_VALUE[move.promotion] if move.promotion else 0
            if victim_value or promo:
                return -(1e6 + victim_value * 10 - attacker_value + promo * 10)
            if child_keys is None:
                return 0.0
            child = cache.get(child_keys.get(move))
            return -(side * child) if child is not None else 0.0

        return sorted(moves, key=priority)

    # ---- quiescence ----------------------------------------------------------

    def _qsearch(self, board, alpha, beta, depth, ply=0, budget=None):
        """Negamax capture search. Returns (score for side to move, pv moves)."""
        self.nodes += 1
        self._check_time()

        if board.is_checkmate():
            return -(MATE_VALUE - ply), []
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0, []

        side = 1.0 if board.turn == chess.WHITE else -1.0
        in_check = board.is_check()
        out_of_budget = depth <= 0 or (budget is not None and self.nodes >= budget)

        if not in_check:
            stand_pat = side * self.static_eval(board)
            if out_of_budget or stand_pat >= beta:
                return stand_pat, []
            alpha = max(alpha, stand_pat)
            best, best_line = stand_pat, []
            moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        else:
            if out_of_budget:
                return side * self.static_eval(board), []
            best, best_line = -INF, []
            moves = list(board.legal_moves)

        if not moves:
            return best, best_line
        child_keys = self.prefetch(board, moves) if len(moves) > 1 else None
        for move in self._order(board, moves, child_keys=child_keys):
            board.push(move)
            score, line = self._qsearch(board, -beta, -alpha, depth - 1, ply + 1, budget)
            board.pop()
            score = -score
            if score > best:
                best, best_line = score, [move] + line
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best, best_line

    # ---- full-width alpha-beta -------------------------------------------------

    def _check_time(self):
        if self._deadline is not None and (self.nodes & 63) == 0 and time.time() > self._deadline:
            raise _TimeUp()

    def _search(self, board, depth, alpha, beta, ply, first=None):
        """Negamax alpha-beta over all legal moves; quiescence at depth 0."""
        if depth <= 0:
            return self._qsearch(board, alpha, beta, self.max_qdepth, ply)
        self.nodes += 1
        self._check_time()

        key = board._transposition_key()
        if ply > 0 and self._seen.get(key, 0) > 0:
            return 0.0, []                 # repetition of a game or line position: a draw
        if board.halfmove_clock >= 100:
            return 0.0, []                 # fifty-move rule

        moves = list(board.legal_moves)
        if not moves:
            return (-(MATE_VALUE - ply) if board.is_check() else 0.0), []
        if board.is_insufficient_material():
            return 0.0, []

        # Children of a depth-1 node are quiescence roots and need stand-pat
        # scores anyway, so fetch them in one batch (also gives move ordering).
        child_keys = self.prefetch(board, moves) if depth == 1 and len(moves) > 1 else None

        best, best_line = -INF, []
        self._seen[key] = self._seen.get(key, 0) + 1      # this position is now part of the line
        try:
            for move in self._order(board, moves, first, child_keys):
                board.push(move)
                score, line = self._search(board, depth - 1, -beta, -alpha, ply + 1)
                board.pop()
                score = -score
                if score > best:
                    best, best_line = score, [move] + line
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
        finally:
            self._seen[key] -= 1
        return best, best_line

    # ---- public API ------------------------------------------------------------

    def evaluate(self, fen_or_board):
        """Quiescence-only evaluation from a FEN string or chess.Board."""
        board = chess.Board(fen_or_board) if isinstance(fen_or_board, str) else fen_or_board.copy()
        started = time.time()
        self.nodes = 0
        self._deadline = None
        static = self.static_eval(board)
        side = 1.0 if board.turn == chess.WHITE else -1.0
        score, line = self._qsearch(board, -INF, INF, self.max_qdepth, 0, budget=self.max_nodes)
        return SearchResult(
            score=_clip(side * score),
            static=static,
            line=board.variation_san(line) if line else "",
            nodes=self.nodes,
            seconds=time.time() - started,
        )

    def best_move(self, fen_or_board, depth=4, max_seconds=3.0, history=None):
        """Full-width search with iterative deepening up to `depth` plies or
        `max_seconds`, whichever comes first. The deepest *completed* iteration
        decides the move.

        Positions already seen in the game are treated as draws. They are taken
        from the board's move stack (pass a chess.Board built from the game's
        moves) and/or from `history`, an iterable of FEN strings or boards."""
        board = chess.Board(fen_or_board) if isinstance(fen_or_board, str) else fen_or_board.copy()
        started = time.time()
        self.nodes = 0
        side = 1.0 if board.turn == chess.WHITE else -1.0

        self._seen = {}
        replay = board.root()
        for move in board.move_stack:          # every earlier position of the game
            self._seen[replay._transposition_key()] = self._seen.get(replay._transposition_key(), 0) + 1
            replay.push(move)
        for item in history or ():
            b = chess.Board(item) if isinstance(item, str) else item
            self._seen[b._transposition_key()] = self._seen.get(b._transposition_key(), 0) + 1
        self._seen.pop(board._transposition_key(), None)   # the root itself is searched normally

        if not any(board.legal_moves):
            score = -(MATE_VALUE) if board.is_check() else 0.0
            return MoveResult("", "", _clip(side * score), _mate_in_from_score(side * score),
                              "", 0, 0, time.time() - started)

        self._deadline = started + max_seconds if max_seconds else None
        best_score, best_line, completed = None, [], 0
        for d in range(1, depth + 1):
            work = board.copy()      # a time-out unwinds mid-line; never leave the root board mutated
            try:
                score, line = self._search(work, d, -INF, INF, 0, first=best_line[0] if best_line else None)
            except _TimeUp:
                break
            best_score, best_line, completed = score, line, d
            if abs(score) >= MATE_VALUE - 1000:
                break                          # forced mate found; deeper search can't improve it
            if self._deadline and time.time() > self._deadline:
                break
        self._deadline = None

        white_score = side * best_score
        move = best_line[0]
        return MoveResult(
            move=move.uci(),
            san=board.san(move),
            score=_clip(white_score),
            mate_in=_mate_in_from_score(white_score),
            line=board.variation_san(best_line),
            depth=completed,
            nodes=self.nodes,
            seconds=time.time() - started,
        )


SAMPLE_FENS = [
    "r1bq1rk1/ppp2ppp/2np4/4p3/2B1n3/2NP1N2/PPP2PPP/R1BQ1RK1 w - -",  # middlegame: black knight hangs on e4
    "r1bq1rk1/ppp2ppp/2np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - -", # same position, knight safe on f6
    "2r1k3/8/8/8/4N3/8/8/4K3 w - -",                                  # Nd6+ forks king and rook
    "k7/8/2K5/8/8/8/8/7R w - -",                                      # mate in 2: Kb6 then Rh8#
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq -",     # fool's mate: White is mated
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fens", nargs="*", help="FEN strings (default: a few tactical samples)")
    parser.add_argument("--model", default="models/chess_eval_10m_tanh_best.pt", help="checkpoint path")
    parser.add_argument("--best", action="store_true", help="find the best move (full search) instead of evaluating")
    parser.add_argument("--depth", type=int, default=4, help="full-search depth in plies (with --best)")
    parser.add_argument("--seconds", type=float, default=3.0, help="time budget per position (with --best)")
    parser.add_argument("--qdepth", type=int, default=6, help="quiescence depth in plies")
    parser.add_argument("--max-nodes", type=int, default=4000, help="quiescence node budget (evaluate only)")
    args = parser.parse_args()

    model, _ = load_model(args.model)
    searcher = Searcher(model, max_depth=args.qdepth, max_nodes=args.max_nodes)
    print(f"loaded {args.model} ({type(model).__name__})")
    fens = args.fens or SAMPLE_FENS
    if args.best:
        print(f"{'move':>6} {'score':>8} {'depth':>5} {'nodes':>7} {'time':>6}  line / fen")
        for fen in fens:
            r = searcher.best_move(fen, depth=args.depth, max_seconds=args.seconds)
            mate = f"  (mate in {abs(r.mate_in)})" if r.mate_in else ""
            print(f"{r.san or '-':>6} {r.score:+8.0f} {r.depth:>5} {r.nodes:>7} {r.seconds:5.2f}s  {r.line or '(no legal moves)'}{mate}")
            print(f"{'':37}{fen}")
    else:
        print(f"{'static':>8}  {'search':>8}  {'nodes':>5}  {'time':>6}  line / fen")
        for fen in fens:
            r = searcher.evaluate(fen)
            print(f"{r.static:+8.0f}  {r.score:+8.0f}  {r.nodes:>5}  {r.seconds:5.2f}s  {r.line or '(stand pat)'}")
            print(f"{'':35}{fen}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
