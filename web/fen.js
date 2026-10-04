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

    return { board, turn, castling };
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

  // Produces "<board> <side> <castling> -". The half-move and full-move
  // counters are deliberately omitted.
  function toFen(state) {
    return boardToFen(state.board) + ' ' + state.turn + ' ' + castlingToFen(state.castling) + ' -';
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

  function positionWarnings(board) {
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
    return warnings;
  }

  const api = {
    PIECES, START_FEN, FILES,
    sqIndex, sqName, emptyBoard,
    parseBoard, parseFen, boardToFen, castlingToFen, toFen,
    castlingAvailability, positionWarnings,
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  } else {
    root.Fen = api;
  }
})(typeof window !== 'undefined' ? window : this);
