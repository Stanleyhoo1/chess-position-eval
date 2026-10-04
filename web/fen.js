// Pure FEN helpers. No DOM access, so this file also loads under node for tests.
(function (root) {
  'use strict';

  const PIECES = 'pnbrqkPNBRQK';
  const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -';
  const FILES = 'abcdefgh';

  // Squares are indexed 0..63 in FEN order: 0 = a8, 7 = h8, 56 = a1, 63 = h1.
  function sqIndex(name) {
    const file = FILES.indexOf(name[0]);
    const rank = Number(name[1]);
    return (8 - rank) * 8 + file;
  }
  function sqName(sq) {
    return FILES[sq & 7] + (8 - (sq >> 3));
  }

  function emptyBoard() {
    return new Array(64).fill(null);
  }

  function parseBoard(str) {
    const ranks = str.split('/');
    if (ranks.length !== 8) {
      throw new Error('Board must have 8 ranks separated by "/", got ' + ranks.length);
    }
    const board = emptyBoard();
    ranks.forEach((rankStr, r) => {
      let file = 0;
      let lastWasDigit = false;
      for (const ch of rankStr) {
        if (ch >= '1' && ch <= '8') {
          if (lastWasDigit) throw new Error('Consecutive digits in rank ' + (8 - r));
          file += Number(ch);
          lastWasDigit = true;
        } else if (PIECES.includes(ch)) {
          if (file > 7) throw new Error('Rank ' + (8 - r) + ' has more than 8 squares');
          board[r * 8 + file] = ch;
          file += 1;
          lastWasDigit = false;
        } else {
          throw new Error('Unexpected character "' + ch + '" in rank ' + (8 - r));
        }
      }
      if (file !== 8) {
        throw new Error('Rank ' + (8 - r) + ' has ' + file + ' squares, expected 8');
      }
    });
    return board;
  }

  // Accepts a full six-field FEN, or just the board, or board + side, or
  // board + side + castling. Missing fields take sensible defaults.
  function parseFen(fen) {
    const parts = String(fen || '').trim().split(/\s+/).filter(Boolean);
    if (parts.length === 0) throw new Error('FEN is empty');

    const board = parseBoard(parts[0]);

    let turn = 'w';
    if (parts.length > 1) {
      if (parts[1] !== 'w' && parts[1] !== 'b') {
        throw new Error('Side to move must be "w" or "b", got "' + parts[1] + '"');
      }
      turn = parts[1];
    }

    const castling = { K: false, Q: false, k: false, q: false };
    if (parts.length > 2 && parts[2] !== '-') {
      for (const ch of parts[2]) {
        if (!(ch in castling)) {
          throw new Error('Invalid castling field "' + parts[2] + '"');
        }
        castling[ch] = true;
      }
    }

    // En passant target square (index), or null. Only a3..h3 / a6..h6 are valid.
    let ep = null;
    if (parts.length > 3 && parts[3] !== '-') {
      if (!/^[a-h][36]$/.test(parts[3])) {
        throw new Error('Invalid en passant square "' + parts[3] + '"');
      }
      ep = sqIndex(parts[3]);
    }

    return { board, turn, castling, ep };
  }

  function boardToFen(board) {
    const out = [];
    for (let r = 0; r < 8; r++) {
      let row = '';
      let empty = 0;
      for (let f = 0; f < 8; f++) {
        const p = board[r * 8 + f];
        if (p) {
          if (empty) { row += empty; empty = 0; }
          row += p;
        } else {
          empty += 1;
        }
      }
      if (empty) row += empty;
      out.push(row);
    }
    return out.join('/');
  }

  function castlingToFen(castling) {
    const s = ['K', 'Q', 'k', 'q'].filter((k) => castling[k]).join('');
    return s || '-';
  }

  // Produces "<board> <side> <castling> <en passant>". The half-move and
  // full-move counters are deliberately omitted.
  function toFen(state) {
    const ep = state.ep === null || state.ep === undefined ? '-' : sqName(state.ep);
    return boardToFen(state.board) + ' ' + state.turn + ' ' + castlingToFen(state.castling) + ' ' + ep;
  }

  // Which castling rights are physically possible given king/rook placement.
  function castlingAvailability(board) {
    const at = (name) => board[sqIndex(name)];
    return {
      K: at('e1') === 'K' && at('h1') === 'R',
      Q: at('e1') === 'K' && at('a1') === 'R',
      k: at('e8') === 'k' && at('h8') === 'r',
      q: at('e8') === 'k' && at('a8') === 'r',
    };
  }

  // ---------- Legality checks ----------

  const isWhite = (p) => p === p.toUpperCase();
  const onBoard = (r, f) => r >= 0 && r < 8 && f >= 0 && f < 8;

  const KNIGHT_STEPS = [[-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1]];
  const KING_STEPS = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];
  const BISHOP_DIRS = [[-1, -1], [-1, 1], [1, -1], [1, 1]];
  const ROOK_DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]];

  // Is square `sq` attacked by any piece of colour `by` ('w' or 'b')?
  // Occupancy of `sq` itself does not matter; blockers in between do.
  function isAttacked(board, sq, by) {
    const r = sq >> 3;
    const f = sq & 7;
    const white = by === 'w';
    const mine = (p) => p && isWhite(p) === white;
    const at = (rr, ff) => (onBoard(rr, ff) ? board[rr * 8 + ff] : null);

    // Pawns: a white pawn attacks the two squares diagonally above it, i.e.
    // towards rank 8 (row 0), so it sits one row *below* the target.
    const pawnRow = white ? r + 1 : r - 1;
    for (const df of [-1, 1]) {
      const p = at(pawnRow, f + df);
      if (p === (white ? 'P' : 'p')) return true;
    }
    for (const [dr, df] of KNIGHT_STEPS) {
      const p = at(r + dr, f + df);
      if (mine(p) && p.toLowerCase() === 'n') return true;
    }
    for (const [dr, df] of KING_STEPS) {
      const p = at(r + dr, f + df);
      if (mine(p) && p.toLowerCase() === 'k') return true;
    }
    const slide = (dirs, types) => {
      for (const [dr, df] of dirs) {
        let rr = r + dr;
        let ff = f + df;
        while (onBoard(rr, ff)) {
          const p = board[rr * 8 + ff];
          if (p) {
            if (mine(p) && types.includes(p.toLowerCase())) return true;
            break;
          }
          rr += dr;
          ff += df;
        }
      }
      return false;
    };
    return slide(BISHOP_DIRS, ['b', 'q']) || slide(ROOK_DIRS, ['r', 'q']);
  }

  function findKing(board, color) {
    const king = color === 'w' ? 'K' : 'k';
    const idx = board.indexOf(king);
    return idx === -1 ? null : idx;
  }

  // Reasons the position cannot be analysed. Only basic legality is checked:
  // king counts, pawn ranks, kings not touching, and the side *not* to move
  // not being in check (its king could otherwise be captured immediately).
  // Piece counts and reachability from the start position are not checked.
  function positionWarnings(board, turn) {
    const warnings = [];
    let whiteKings = 0;
    let blackKings = 0;
    let backRankPawn = false;
    board.forEach((p, sq) => {
      if (p === 'K') whiteKings += 1;
      if (p === 'k') blackKings += 1;
      const rank = sq >> 3;
      if ((p === 'P' || p === 'p') && (rank === 0 || rank === 7)) backRankPawn = true;
    });
    if (whiteKings !== 1) warnings.push('White needs exactly one king');
    if (blackKings !== 1) warnings.push('Black needs exactly one king');
    if (backRankPawn) warnings.push('Pawns cannot stand on the first or last rank');
    if (whiteKings !== 1 || blackKings !== 1) return warnings;

    const wk = findKing(board, 'w');
    const bk = findKing(board, 'b');
    if (Math.abs((wk >> 3) - (bk >> 3)) <= 1 && Math.abs((wk & 7) - (bk & 7)) <= 1) {
      warnings.push('The kings cannot stand next to each other');
      return warnings;   // the check test below would only repeat this
    }
    if (turn === 'w' || turn === 'b') {
      const idle = turn === 'w' ? 'b' : 'w';
      const idleKing = idle === 'w' ? wk : bk;
      if (isAttacked(board, idleKing, turn)) {
        warnings.push((idle === 'w' ? 'White' : 'Black') + ' is in check but it is ' +
          (turn === 'w' ? "White's" : "Black's") + ' move');
      }
    }
    return warnings;
  }

  // ---------- Move generation ----------
  //
  // A position is { board, turn, castling, ep } exactly as parseFen returns it.
  // A move is { from, to, promo?, castle?, ep?, double? } with square indices.

  const pieceColor = (p) => (isWhite(p) ? 'w' : 'b');

  function generateMoves(pos) {
    const { board, turn, castling, ep } = pos;
    const white = turn === 'w';
    const own = (p) => !!p && isWhite(p) === white;
    const enemy = (p) => !!p && isWhite(p) !== white;
    const moves = [];
    const add = (from, to, extra) => moves.push(Object.assign({ from, to }, extra || {}));
    const addPawn = (from, to) => {
      if ((to >> 3) === (white ? 0 : 7)) {
        for (const promo of 'qrbn') add(from, to, { promo });
      } else {
        add(from, to);
      }
    };
    const slide = (sq, r, f, dirs) => {
      for (const [dr, df] of dirs) {
        let rr = r + dr;
        let ff = f + df;
        while (onBoard(rr, ff)) {
          const to = rr * 8 + ff;
          if (board[to]) { if (enemy(board[to])) add(sq, to); break; }
          add(sq, to);
          rr += dr;
          ff += df;
        }
      }
    };

    for (let sq = 0; sq < 64; sq++) {
      const p = board[sq];
      if (!own(p)) continue;
      const r = sq >> 3;
      const f = sq & 7;
      const t = p.toLowerCase();
      if (t === 'p') {
        const dir = white ? -1 : 1;
        const r1 = r + dir;
        if (onBoard(r1, f) && !board[r1 * 8 + f]) {
          addPawn(sq, r1 * 8 + f);
          const startRow = white ? 6 : 1;
          if (r === startRow && !board[(r + 2 * dir) * 8 + f]) add(sq, (r + 2 * dir) * 8 + f, { double: true });
        }
        for (const df of [-1, 1]) {
          const ff = f + df;
          if (!onBoard(r1, ff)) continue;
          const to = r1 * 8 + ff;
          if (enemy(board[to])) addPawn(sq, to);
          else if (ep !== null && ep !== undefined && to === ep) add(sq, to, { ep: true });
        }
      } else if (t === 'n' || t === 'k') {
        for (const [dr, df] of (t === 'n' ? KNIGHT_STEPS : KING_STEPS)) {
          const rr = r + dr;
          const ff = f + df;
          if (onBoard(rr, ff) && !own(board[rr * 8 + ff])) add(sq, rr * 8 + ff);
        }
        if (t === 'k') {
          const them = white ? 'b' : 'w';
          const safe = (...sqs) => sqs.every((x) => !isAttacked(board, x, them));
          const empty = (...sqs) => sqs.every((x) => !board[x]);
          if (white && sq === 60) {
            if (castling.K && board[63] === 'R' && empty(61, 62) && safe(60, 61, 62)) add(60, 62, { castle: 'K' });
            if (castling.Q && board[56] === 'R' && empty(57, 58, 59) && safe(60, 59, 58)) add(60, 58, { castle: 'Q' });
          } else if (!white && sq === 4) {
            if (castling.k && board[7] === 'r' && empty(5, 6) && safe(4, 5, 6)) add(4, 6, { castle: 'k' });
            if (castling.q && board[0] === 'r' && empty(1, 2, 3) && safe(4, 3, 2)) add(4, 2, { castle: 'q' });
          }
        }
      } else if (t === 'b') {
        slide(sq, r, f, BISHOP_DIRS);
      } else if (t === 'r') {
        slide(sq, r, f, ROOK_DIRS);
      } else if (t === 'q') {
        slide(sq, r, f, BISHOP_DIRS);
        slide(sq, r, f, ROOK_DIRS);
      }
    }
    return moves;
  }

  // Apply a move and return the new position (the input is not modified).
  function makeMove(pos, m) {
    const board = pos.board.slice();
    const castling = Object.assign({}, pos.castling);
    const p = board[m.from];
    const white = isWhite(p);
    board[m.to] = m.promo ? (white ? m.promo.toUpperCase() : m.promo) : p;
    board[m.from] = null;
    if (m.ep) board[m.to + (white ? 8 : -8)] = null;
    if (m.castle) {
      const rook = { K: [63, 61], Q: [56, 59], k: [7, 5], q: [0, 3] }[m.castle];
      board[rook[1]] = board[rook[0]];
      board[rook[0]] = null;
    }
    if (p.toLowerCase() === 'k') {
      if (white) { castling.K = false; castling.Q = false; } else { castling.k = false; castling.q = false; }
    }
    // A rook leaving or being captured on its home square loses that right.
    for (const [sq, right] of [[63, 'K'], [56, 'Q'], [7, 'k'], [0, 'q']]) {
      if (m.from === sq || m.to === sq) castling[right] = false;
    }
    const ep = m.double ? (m.from + m.to) / 2 : null;
    return { board, turn: white ? 'b' : 'w', castling, ep };
  }

  function isInCheck(pos) {
    const k = findKing(pos.board, pos.turn);
    return k !== null && isAttacked(pos.board, k, pos.turn === 'w' ? 'b' : 'w');
  }

  // Moves that do not leave the mover's own king attacked.
  function legalMoves(pos) {
    return generateMoves(pos).filter((m) => {
      const next = makeMove(pos, m);
      const k = findKing(next.board, pos.turn);
      return k !== null && !isAttacked(next.board, k, next.turn);
    });
  }

  const api = {
    PIECES, START_FEN, FILES,
    sqIndex, sqName, emptyBoard,
    parseBoard, parseFen, boardToFen, castlingToFen, toFen,
    castlingAvailability, positionWarnings, isAttacked, findKing,
    pieceColor, generateMoves, makeMove, legalMoves, isInCheck,
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  } else {
    root.Fen = api;
  }
})(typeof window !== 'undefined' ? window : this);
