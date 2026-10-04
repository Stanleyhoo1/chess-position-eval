"""HTTP server that evaluates chess positions with the trained model.

Endpoints:
    POST /api/evaluate   body {"fen": "<fen>"} ->
                         {"fen": ..., "cp": <centipawns after capture search>,
                          "pawns": <cp / 100>, "static_cp": <network alone>,
                          "line": "<captures played, SAN>", "nodes": n, "seconds": s}
                         or body {"fens": ["<fen>", ...]} (up to MAX_BATCH) ->
                         {"results": [{same fields}, ...]} in the same order. The
                         search node budget is shared across the batch (at least
                         MIN_BATCH_NODES each), so scoring every legal move for the
                         in-app bot costs about as much as one single evaluation.
    GET  /api/health     -> {"ok": true, "model": "<path>", "device": "cpu", ...}

Each evaluation runs the quiescence (capture) search from search.py with the
network scoring the quiet leaf positions, so hanging pieces and mates are
resolved rather than guessed. Scores are from White's point of view and
clipped to +-EVAL_SCALE centipawns (the range the model was trained on).
``pawns`` is the same number divided by 100, which is what the eval bar
displays (e.g. 125 cp -> +1.25).

Unless --no-static is given, every other path is served from web/, so

    python server.py

is enough to run the whole app at http://localhost:8000 without Docker.
Inside Docker the nginx container serves web/ and proxies /api/ here.
"""

import argparse
import json
import os
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import chess
import torch

from load_model import load_model
from search import Searcher
from train import EVAL_SCALE

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
MAX_BATCH = 256
MIN_BATCH_NODES = 250
MAX_BODY = 128 * MAX_BATCH


class Handler(SimpleHTTPRequestHandler):
    """Serves /api/* from the model and everything else from web/."""

    model = None
    model_path = None
    static = True
    search_depth = 6
    search_nodes = 4000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    # --- helpers -----------------------------------------------------------

    def send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            raise ValueError("request body missing or too large")
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON: {e}") from None

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # --- routes ------------------------------------------------------------

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/api/health":
            device = str(next(self.model.parameters()).device)
            return self.send_json(HTTPStatus.OK, {
                "ok": True, "model": self.model_path, "device": device,
                "eval_scale": EVAL_SCALE,
            })
        if self.path.startswith("/api/"):
            return self.send_json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
        if not self.static:
            return self.send_json(HTTPStatus.NOT_FOUND, {"error": "static files not served here"})
        return super().do_GET()

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/evaluate":
            return self.send_json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
        try:
            data = self.read_json()
            if not isinstance(data, dict):
                raise ValueError("body must be a JSON object")
            if "fens" in data:
                fens = data["fens"]
                if not isinstance(fens, list) or not fens:
                    raise ValueError('"fens" must be a non-empty list')
                if len(fens) > MAX_BATCH:
                    raise ValueError(f'"fens" may hold at most {MAX_BATCH} positions')
                fens = [clean_fen(f) for f in fens]
                nodes = max(MIN_BATCH_NODES, self.search_nodes // len(fens))
                results = [self.result_json(f, evaluate_position(self.model, f, self.search_depth, nodes))
                           for f in fens]
                return self.send_json(HTTPStatus.OK, {"results": results})
            fen = clean_fen(data.get("fen"))
            result = evaluate_position(self.model, fen, self.search_depth, self.search_nodes)
        except (ValueError, KeyError, IndexError) as e:
            return self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
        return self.send_json(HTTPStatus.OK, self.result_json(fen, result))

    @staticmethod
    def result_json(fen, result):
        cp = result.score
        return {
            "fen": fen, "cp": round(cp, 1), "pawns": round(cp / 100, 2),
            "static_cp": round(result.static, 1), "line": result.line,
            "nodes": result.nodes, "seconds": round(result.seconds, 3),
        }


def clean_fen(fen):
    if not isinstance(fen, str) or not fen.strip():
        raise ValueError('"fen" must be a non-empty string')
    return fen.strip()


_STATUS_MESSAGES = {
    chess.STATUS_NO_WHITE_KING: "White has no king",
    chess.STATUS_NO_BLACK_KING: "Black has no king",
    chess.STATUS_TOO_MANY_KINGS: "a side has more than one king",
    chess.STATUS_PAWNS_ON_BACKRANK: "pawns cannot stand on the first or last rank",
    chess.STATUS_OPPOSITE_CHECK: "the side not to move is in check",
    chess.STATUS_TOO_MANY_WHITE_PAWNS: "White has too many pawns",
    chess.STATUS_TOO_MANY_BLACK_PAWNS: "Black has too many pawns",
    chess.STATUS_TOO_MANY_WHITE_PIECES: "White has too many pieces",
    chess.STATUS_TOO_MANY_BLACK_PIECES: "Black has too many pieces",
    chess.STATUS_BAD_CASTLING_RIGHTS: "castling rights do not match the king and rook squares",
    chess.STATUS_INVALID_EP_SQUARE: "the en-passant square is not valid",
    chess.STATUS_TOO_MANY_CHECKERS: "the king is in check from too many pieces",
}


def validate_position(fen):
    """Return a chess.Board for the FEN, raising ValueError with a readable reason."""
    try:
        board = chess.Board(fen)
    except ValueError as e:
        raise ValueError(f"invalid FEN: {e}") from None
    if not board.is_valid():
        status = board.status()
        reasons = [msg for flag, msg in _STATUS_MESSAGES.items() if status & flag]
        raise ValueError("position is not legal: " + ("; ".join(reasons) or "unknown problem"))
    return board


def evaluate_position(model, fen, depth=6, max_nodes=4000):
    """FEN -> search.SearchResult (scores in centipawns from White's point of
    view, clipped to +-EVAL_SCALE). Raises ValueError for an invalid position."""
    board = validate_position(fen)
    with torch.inference_mode():
        result = Searcher(model, max_depth=depth, max_nodes=max_nodes).evaluate(board)
    result.score = max(-EVAL_SCALE, min(EVAL_SCALE, result.score))
    result.static = max(-EVAL_SCALE, min(EVAL_SCALE, result.static))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", "models/chess_eval_10m_tanh_best.pt"),
                        help="checkpoint path (default: $MODEL_PATH or models/chess_eval_10m_tanh_best.pt)")
    parser.add_argument("--depth", type=int, default=int(os.environ.get("SEARCH_DEPTH", 6)),
                        help="capture-search depth in plies")
    parser.add_argument("--max-nodes", type=int, default=int(os.environ.get("SEARCH_NODES", 4000)),
                        help="capture-search node budget per request")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-static", action="store_true", help="only serve /api/, not web/")
    parser.add_argument("--cpu", action="store_true", help="run the model on CPU even if CUDA is available")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else None
    model, metadata = load_model(args.model, device=device)
    print(f"loaded {args.model} on {next(model.parameters()).device}"
          + (f" (trained on {metadata.get('rows', '?'):,} rows)" if metadata.get("rows") else ""))

    Handler.model = model
    Handler.model_path = args.model
    Handler.static = not args.no_static
    Handler.search_depth = args.depth
    Handler.search_nodes = args.max_nodes

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    what = "api + web/" if Handler.static else "api only"
    print(f"serving {what} at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
