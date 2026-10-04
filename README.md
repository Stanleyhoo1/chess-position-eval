# Chess Position Setup

A chess.com-style position editor: set up a position by dragging or clicking pieces
onto the board, or by pasting a FEN. Choose the side to move and castling rights, then
copy the resulting FEN.

## Run

```sh
docker compose up --build
```

Open <http://localhost:8080>.

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
- `Dockerfile`, `nginx.conf`, `docker-compose.yml` — nginx container serving `web/`

Piece images are the "cburnett" set from Wikimedia Commons, licensed CC BY-SA 3.0.
