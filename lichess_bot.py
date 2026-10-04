"""Play on Lichess as a BOT account, choosing moves with the trained
evaluator and the alpha-beta search from search.py.

One-time setup
--------------
1. Create a *new* Lichess account for the bot. It must not have played any
   games yet, because step 4 is only allowed for fresh accounts.
2. Create an API token at https://lichess.org/account/oauth/token/create
   with the scopes "Play games with the bot API" (bot:play) and
   "Create, accept, decline challenges" (challenge:write).
3. Put the token in the environment:   export LICHESS_TOKEN=lip_xxxxxxxx
4. python lichess_bot.py upgrade        # irreversible: marks the account as a BOT

Playing
-------
    python lichess_bot.py play --ai 3                     # one game vs Lichess Stockfish level 3
    python lichess_bot.py play --ai 5 --color black --clock 5+3 --games 3
    python lichess_bot.py play --user maia1               # challenge another bot or user (casual)
    python lichess_bot.py resume                          # play all games already in progress, then exit
    python lichess_bot.py listen                          # accept incoming standard challenges

Options such as --depth / --seconds control the search per move; the move
clock is also respected so the bot does not flag in fast games. Add
--chat-eval to have the bot post its evaluation to the game chat after every
move, since Lichess shows no eval bar for games in progress. Watch the
games live at https://lichess.org/@/<bot name>/all.
"""

import argparse
import json
import os
import sys
import threading
import time

import chess
import requests

from load_model import load_model
from search import Searcher

LICHESS = "https://lichess.org"
DEFAULT_MODEL = "models/chess_eval_10m_tanh_best.pt"


# ---------------------------------------------------------------------------
# Lichess API
# ---------------------------------------------------------------------------

class LichessAPI:
    def __init__(self, token):
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {token}"

    def get(self, path, **kw):
        r = self.s.get(LICHESS + path, timeout=30, **kw)
        r.raise_for_status()
        return r.json()

    def post(self, path, data=None):
        r = self.s.post(LICHESS + path, data=data, timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text.strip()[:300]}")
        return r.json() if r.content else {}

    def stream(self, path):
        """Yield parsed ndjson objects; Lichess sends blank keep-alive lines."""
        with self.s.get(LICHESS + path, stream=True, timeout=(30, 90)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    yield json.loads(line)

    # convenience wrappers
    def account(self):
        return self.get("/api/account")

    def now_playing(self):
        return self.get("/api/account/playing").get("nowPlaying", [])

    def upgrade_to_bot(self):
        return self.post("/api/bot/account/upgrade")

    def challenge_ai(self, level, clock, color):
        data = {"level": level, "color": color}
        data.update(_clock_fields(clock))
        return self.post("/api/challenge/ai", data)

    def challenge_user(self, user, clock, color, rated=False):
        data = {"rated": "true" if rated else "false", "color": color}
        data.update(_clock_fields(clock))
        return self.post(f"/api/challenge/{user}", data)

    def accept(self, challenge_id):
        return self.post(f"/api/challenge/{challenge_id}/accept")

    def decline(self, challenge_id, reason="generic"):
        return self.post(f"/api/challenge/{challenge_id}/decline", {"reason": reason})

    def move(self, game_id, uci):
        return self.post(f"/api/bot/game/{game_id}/move/{uci}")

    def chat(self, game_id, text):
        try:
            self.post(f"/api/bot/game/{game_id}/chat", {"room": "player", "text": text})
        except RuntimeError:
            pass                        # chat is cosmetic; never let it break a game


def _clock_fields(clock):
    """'5+3' -> {'clock.limit': 300, 'clock.increment': 3}; None -> correspondence (days=1)."""
    if not clock:
        return {"days": 1}
    minutes, _, inc = clock.partition("+")
    return {"clock.limit": int(float(minutes) * 60), "clock.increment": int(inc or 0)}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Engine:
    """Thin wrapper around Searcher with clock-aware time management."""

    def __init__(self, model_path, depth, seconds, qdepth, max_nodes):
        model, _ = load_model(model_path)
        self.searcher = Searcher(model, max_depth=qdepth, max_nodes=max_nodes)
        self.depth = depth
        self.seconds = seconds
        self.lock = threading.Lock()    # Searcher is not thread-safe
        print(f"engine: {model_path} ({type(model).__name__}) depth {depth}, "
              f"{seconds:.1f}s/move max", flush=True)

    def budget(self, time_left_ms, inc_ms):
        """Seconds to spend on this move given the clock."""
        if time_left_ms is None:
            return self.seconds
        left = time_left_ms / 1000
        inc = (inc_ms or 0) / 1000
        return max(0.1, min(self.seconds, left / 40 + inc * 0.8, left * 0.5))

    def choose(self, board, time_left_ms=None, inc_ms=None):
        with self.lock:
            return self.searcher.best_move(board, depth=self.depth,
                                           max_seconds=self.budget(time_left_ms, inc_ms))


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------

def board_from(initial_fen, moves):
    board = chess.Board() if initial_fen in ("startpos", "", None) else chess.Board(initial_fen)
    for uci in moves.split():
        board.push_uci(uci)
    return board


def play_game(api, engine, game_id, my_id, greet=True, chat_eval=False):
    """Follow one game's stream and reply with a move whenever it is our turn."""
    color = None
    initial_fen = "startpos"
    print(f"[{game_id}] joined", flush=True)
    for event in api.stream(f"/api/bot/game/stream/{game_id}"):
        kind = event.get("type")
        if kind == "gameFull":
            initial_fen = event.get("initialFen", "startpos")
            white_id = (event.get("white") or {}).get("id")
            color = chess.WHITE if white_id == my_id else chess.BLACK
            opp = (event.get("black") if color == chess.WHITE else event.get("white")) or {}
            print(f"[{game_id}] playing {'White' if color == chess.WHITE else 'Black'} vs "
                  f"{opp.get('name') or opp.get('id') or 'AI'}  https://lichess.org/{game_id}", flush=True)
            if greet:
                api.chat(game_id, f"gl hf! I'm a neural-network evaluator + alpha-beta search bot (depth {engine.depth}).")
            state = event["state"]
        elif kind == "gameState":
            state = event
        else:
            continue                    # chatLine, opponentGone

        status = state.get("status")
        if status != "started":
            winner = state.get("winner")
            outcome = "draw" if not winner else f"{winner} wins"
            print(f"[{game_id}] over: {status} ({outcome})", flush=True)
            return status

        board = board_from(initial_fen, state.get("moves", ""))
        if board.turn != color or board.is_game_over():
            continue
        my_time = state.get("wtime") if color == chess.WHITE else state.get("btime")
        my_inc = state.get("winc") if color == chess.WHITE else state.get("binc")
        result = engine.choose(board, my_time, my_inc)
        if not result.move:
            continue
        mate = f" mate in {abs(result.mate_in)}" if result.mate_in else ""
        print(f"[{game_id}] {board.fullmove_number}{'.' if color == chess.WHITE else '...'} {result.san:<7} "
              f"{result.score:+6.0f}{mate}  d{result.depth} {result.nodes}n {result.seconds:.1f}s  {result.line}", flush=True)
        try:
            api.move(game_id, result.move)
        except RuntimeError as e:
            print(f"[{game_id}] move rejected: {e}", flush=True)
            continue
        if chat_eval:
            # Lichess has no eval bar for live games, so post our score to the
            # game chat; it shows on the game page for everyone watching.
            if result.mate_in:
                text = f"{result.san}: mate in {abs(result.mate_in)}"
            else:
                text = f"{result.san}: eval {result.score / 100:+.2f} (depth {result.depth})"
            api.chat(game_id, text)
    return "stream-ended"


# ---------------------------------------------------------------------------
# Event loop
# ---------------------------------------------------------------------------

def acceptable(challenge):
    """Accept standard-chess challenges only; return a decline reason otherwise."""
    if challenge.get("variant", {}).get("key") != "standard":
        return "variant"
    if challenge.get("speed") == "correspondence" and not challenge.get("days"):
        return "timeControl"
    return None


def run_events(api, engine, my_id, accept_challenges, wait_for=None, chat_eval=False):
    """Follow the account event stream and play games as they start.

    With `wait_for` (a game id) return once that game has finished; otherwise
    run forever.  Lichess re-sends gameStart for every game the account is
    already in, so an old game left over from a previous run can finish before
    ours does -- it must not count as "our" game.
    """
    threads = {}
    for event in api.stream("/api/stream/event"):
        kind = event.get("type")
        if kind == "challenge":
            ch = event["challenge"]
            if ch.get("challenger", {}).get("id") == my_id:
                continue                # our own outgoing challenge
            reason = acceptable(ch) if accept_challenges else "generic"
            who = ch.get("challenger", {}).get("name", "?")
            if reason:
                print(f"declining challenge from {who} ({reason})", flush=True)
                api.decline(ch["id"], reason)
            else:
                print(f"accepting challenge from {who}", flush=True)
                api.accept(ch["id"])
        elif kind == "gameStart":
            game_id = event["game"]["gameId"]
            if game_id in threads:
                continue
            t = threading.Thread(target=play_game, args=(api, engine, game_id, my_id, True, chat_eval), daemon=True)
            threads[game_id] = t
            t.start()
        elif kind == "gameFinish":
            game_id = event["game"]["gameId"]
            if wait_for is not None and game_id == wait_for:
                if game_id in threads:
                    threads[game_id].join()     # let it print the result line
                wait_for_threads(threads)
                return


def wait_for_threads(threads):
    """Block until every game thread has exited.

    The game threads run the model inside PyTorch; if the interpreter starts
    shutting down while one of them is still searching, libtorch's teardown
    aborts the process with "terminate called without an active exception".
    """
    live = {gid: t for gid, t in threads.items() if t.is_alive()}
    if not live:
        return
    ids = ", ".join(f"https://lichess.org/{gid}" for gid in live)
    print(f"waiting for {len(live)} other game(s) still in progress: {ids}  (Ctrl+C to abandon)", flush=True)
    for t in live.values():
        t.join()


def resume_games(api, engine, my_id, chat_eval=False):
    """Play every game the account currently has in progress and return when
    they are all finished. Returns the number of games played."""
    games = api.now_playing()
    if not games:
        print("no games in progress", flush=True)
        return 0
    for g in games:
        opp = g.get("opponent", {})
        who = opp.get("username") or (f"Stockfish level {opp.get('ai')}" if opp.get("ai") else "?")
        turn = "our move" if g.get("isMyTurn") else "waiting for opponent"
        print(f"resuming {g['gameId']}: {g.get('color')} vs {who} ({g.get('speed')}, {turn})  https://lichess.org/{g['gameId']}", flush=True)
    threads = [threading.Thread(target=play_game, args=(api, engine, g["gameId"], my_id, False, chat_eval), daemon=True)
               for g in games]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return len(games)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token", default=os.environ.get("LICHESS_TOKEN"), help="API token (default: $LICHESS_TOKEN)")
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL), help="checkpoint path")
    parser.add_argument("--depth", type=int, default=4, help="full-width search depth in plies")
    parser.add_argument("--seconds", type=float, default=3.0, help="max seconds per move (less when short on time)")
    parser.add_argument("--qdepth", type=int, default=6, help="quiescence depth")
    parser.add_argument("--max-nodes", type=int, default=4000, help="quiescence node budget")
    parser.add_argument("--chat-eval", action="store_true",
                        help="post the bot's evaluation to the game chat after each move (visible on lichess.org)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("upgrade", help="turn this (fresh) account into a BOT account; irreversible")

    p = sub.add_parser("play", help="start a game and play it")
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--ai", type=int, choices=range(1, 9), metavar="LEVEL", help="Lichess Stockfish level 1-8")
    who.add_argument("--user", help="challenge this user or bot (casual)")
    p.add_argument("--color", default="random", choices=["white", "black", "random"])
    p.add_argument("--clock", default="10+5", help="minutes+increment, e.g. 5+3 (default 10+5)")
    p.add_argument("--games", type=int, default=1, help="how many games to play in a row")

    sub.add_parser("resume", help="play all games currently in progress, then exit")
    sub.add_parser("listen", help="accept incoming challenges and play them until stopped")
    args = parser.parse_args()

    if not args.token:
        sys.exit("no token: pass --token or set LICHESS_TOKEN (see the docstring for setup)")
    api = LichessAPI(args.token)
    me = api.account()
    my_id = me["id"]
    is_bot = me.get("title") == "BOT"
    print(f"logged in as {me.get('username')}{' (BOT)' if is_bot else ''}", flush=True)

    if args.cmd == "upgrade":
        if is_bot:
            print("already a BOT account")
            return 0
        print("This permanently converts the account into a BOT account (no rated games vs humans).")
        if input("Type 'yes' to continue: ").strip().lower() != "yes":
            return 1
        print(api.upgrade_to_bot())
        return 0

    if not is_bot:
        sys.exit("this account is not a BOT account yet; run: python lichess_bot.py upgrade")

    engine = Engine(args.model, args.depth, args.seconds, args.qdepth, args.max_nodes)

    if args.cmd == "resume":
        try:
            resume_games(api, engine, my_id, chat_eval=args.chat_eval)
        except KeyboardInterrupt:
            pass
        return 0

    if args.cmd == "listen":
        print("waiting for challenges (Ctrl+C to stop)", flush=True)
        try:
            run_events(api, engine, my_id, accept_challenges=True, chat_eval=args.chat_eval)
        except KeyboardInterrupt:
            pass
        return 0

    for n in range(args.games):
        if args.ai:
            game = api.challenge_ai(args.ai, args.clock, args.color)
            game_id = game.get("id")
            print(f"game {n + 1}/{args.games}: challenged Stockfish level {args.ai}  https://lichess.org/{game_id}", flush=True)
        else:
            resp = api.challenge_user(args.user, args.clock, args.color)
            game_id = (resp.get("challenge") or resp).get("id")   # the game keeps the challenge id once accepted
            print(f"game {n + 1}/{args.games}: challenged {args.user}, waiting for them to accept  https://lichess.org/{game_id}", flush=True)
        try:
            run_events(api, engine, my_id, accept_challenges=False, wait_for=game_id, chat_eval=args.chat_eval)
        except KeyboardInterrupt:
            return 0
    return 0


def _exit(code):
    """Exit without tearing libtorch down under a game thread that is still searching."""
    sys.stdout.flush()
    sys.stderr.flush()
    if any(t.is_alive() for t in threading.enumerate() if t is not threading.main_thread()):
        os._exit(code or 0)
    sys.exit(code)


if __name__ == "__main__":
    _exit(main())
