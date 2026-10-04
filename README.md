# Chess Position Setup

A chess.com-style position editor: set up a position by dragging or clicking pieces
onto the board, or by pasting a FEN. Choose the side to move and castling rights, then
copy the resulting FEN, or press **Analyze position** to score it with the trained
neural network. The result is shown on an eval bar beside the board, in pawns from
White's point of view (+1.00 = White is a pawn better).

## Run

With Docker (nginx serves `web/`, a second container runs the model behind `/api/`):

```sh
docker compose up --build
```

Open <http://localhost:8080>.

Without Docker, the Python server serves both the page and the model:

```sh
python server.py            # http://localhost:8000
```

Both need a trained checkpoint at `models/chess_eval.pt` (produced by `train.py`).

## API

`POST /api/evaluate` with `{"fen": "<fen>"}` returns
`{"fen": ..., "cp": <centipawns>, "pawns": <cp / 100>}`, clipped to ±2500 cp.
`GET /api/health` reports the loaded checkpoint.

## Usage

- Click a piece in the palette, then click squares to place copies of it. Click the
  same palette piece again (or press Esc) to deselect.
- Drag pieces from the palette or around the board. Drag a piece off the board, or
  right-click it, to remove it.
- Click a piece on the board, then click another square to move it.
- Castling checkboxes are only enabled when the king and the relevant rook are on
  their home squares.
- Paste a FEN into the text box and press Load (or Enter). Only the board field is
  required; side to move and castling are optional and the move counters are ignored.
- The text box always shows the FEN for the current position, without the half-move
  and full-move counters: `<board> <side> <castling> -`.

## Layout

- `web/` — static site (`index.html`, `style.css`, `app.js`, `fen.js`, `pieces/`)
- `web/fen.js` — pure FEN parse/serialize logic, loadable under node for testing
- `Dockerfile`, `nginx.conf`, `docker-compose.yml` — nginx container serving `web/` and proxying `/api/`
- `server.py`, `Dockerfile.api` — model inference service (`load_model.py` + `model.py`)
- `get_training_data.py`, `train.py` — build the `fen,evaluation` dataset and train the model

Piece images are the "cburnett" set from Wikimedia Commons, licensed CC BY-SA 3.0.
