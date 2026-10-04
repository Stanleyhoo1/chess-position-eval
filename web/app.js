(() => {
  'use strict';

  const F = window.Fen;
  const $ = (id) => document.getElementById(id);

  const boardEl = $('board');
  const paletteEl = $('palette');
  const turnEl = $('turn');
  const fenEl = $('fen');
  const msgEl = $('msg');
  const warningsEl = $('warnings');
  const eraseBtn = $('erase');
  const castleEls = { K: $('cK'), Q: $('cQ'), k: $('ck'), q: $('cq') };

  const start = F.parseFen(F.START_FEN);
  const state = {
    board: start.board,
    turn: start.turn,
    castling: start.castling,
    flipped: false,
    tool: null,      // piece code ('K', 'p', ...), 'erase', or null
    selected: null,  // square index awaiting a destination click
  };

  const pieceClass = (p) => (p === p.toUpperCase() ? 'w' : 'b') + p.toUpperCase();

  // ---------- Board DOM ----------

  const squareEls = new Array(64);

  function buildBoard() {
    boardEl.innerHTML = '';
    for (let visual = 0; visual < 64; visual++) {
      const sq = state.flipped ? 63 - visual : visual;
      const rank = sq >> 3;
      const file = sq & 7;
      const el = document.createElement('div');
      el.className = 'square ' + (((rank + file) & 1) ? 'dark' : 'light');
      el.dataset.sq = sq;

      const vcol = visual & 7;
      const vrow = visual >> 3;
      if (vcol === 0) {
        const c = document.createElement('span');
        c.className = 'coord rank';
        c.textContent = 8 - rank;
        el.appendChild(c);
      }
      if (vrow === 7) {
        const c = document.createElement('span');
        c.className = 'coord file';
        c.textContent = F.FILES[file];
        el.appendChild(c);
      }
      const holder = document.createElement('div');
      holder.className = 'holder';
      el.appendChild(holder);

      squareEls[sq] = el;
      boardEl.appendChild(el);
    }
    renderPieces();
  }

  function renderPieces() {
    for (let sq = 0; sq < 64; sq++) {
      const el = squareEls[sq];
      const p = state.board[sq];
      el.lastElementChild.className = 'holder' + (p ? ' piece ' + pieceClass(p) : '');
      el.classList.toggle('has-piece', !!p);
      el.classList.toggle('selected', state.selected === sq);
    }
    boardEl.classList.toggle('tool-active', !!state.tool && state.tool !== 'erase');
    boardEl.classList.toggle('tool-erase', state.tool === 'erase');
  }

  // ---------- Palette ----------

  function buildPalette() {
    paletteEl.innerHTML = '';
    for (const color of ['b', 'w']) {
      for (const p of 'pbnrqk') {
        const code = color === 'w' ? p.toUpperCase() : p;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'piece-btn';
        btn.dataset.piece = code;
        btn.title = (color === 'w' ? 'White ' : 'Black ') + pieceName(p);
        const img = document.createElement('div');
        img.className = 'piece ' + pieceClass(code);
        btn.appendChild(img);
        paletteEl.appendChild(btn);
      }
    }
    renderPalette();
  }

  function pieceName(p) {
    return { p: 'pawn', b: 'bishop', n: 'knight', r: 'rook', q: 'queen', k: 'king' }[p.toLowerCase()];
  }

  function renderPalette() {
    for (const btn of paletteEl.children) {
      btn.classList.toggle('selected', btn.dataset.piece === state.tool);
    }
    eraseBtn.classList.toggle('active', state.tool === 'erase');
  }

  // ---------- Sync panel from state ----------

  function syncCastling() {
    const avail = F.castlingAvailability(state.board);
    for (const r of ['K', 'Q', 'k', 'q']) {
      if (!avail[r]) state.castling[r] = false;
      const el = castleEls[r];
      el.disabled = !avail[r];
      el.checked = state.castling[r];
      el.parentElement.classList.toggle('disabled', !avail[r]);
    }
  }

  function syncFen() {
    fenEl.value = F.toFen(state);
  }

  function syncWarnings() {
    warningsEl.innerHTML = '';
    for (const w of F.positionWarnings(state.board)) {
      const li = document.createElement('li');
      li.textContent = w;
      warningsEl.appendChild(li);
    }
  }

  function setMsg(text, kind) {
    msgEl.textContent = text || '';
    msgEl.className = 'msg' + (kind ? ' ' + kind : '');
  }

  function update() {
    turnEl.value = state.turn;
    syncCastling();
    renderPieces();
    renderPalette();
    syncFen();
    syncWarnings();
  }

  // ---------- Pointer handling (click + drag) ----------

  let ptr = null;

  function startPointer(e, piece, fromSq) {
    if (e.button !== 0 || ptr) return;
    e.preventDefault();
    ptr = { piece, fromSq, x: e.clientX, y: e.clientY, moved: false, ghost: null };
  }

  function squareAt(x, y) {
    const el = document.elementFromPoint(x, y);
    const sqEl = el && el.closest('.square');
    return sqEl ? Number(sqEl.dataset.sq) : null;
  }

  function onPointerMove(e) {
    if (!ptr) return;
    if (!ptr.moved) {
      if (Math.hypot(e.clientX - ptr.x, e.clientY - ptr.y) < 5) return;
      ptr.moved = true;
      const size = squareEls[0].getBoundingClientRect().width;
      const ghost = document.createElement('div');
      ghost.className = 'ghost piece ' + pieceClass(ptr.piece);
      ghost.style.width = ghost.style.height = size + 'px';
      document.body.appendChild(ghost);
      ptr.ghost = ghost;
      ptr.size = size;
      if (ptr.fromSq !== null) squareEls[ptr.fromSq].classList.add('dragging');
      if (state.selected !== null) { state.selected = null; renderPieces(); }
    }
    ptr.ghost.style.left = (e.clientX - ptr.size / 2) + 'px';
    ptr.ghost.style.top = (e.clientY - ptr.size / 2) + 'px';
  }

  function endPointer(e, cancelled) {
    if (!ptr) return;
    const p = ptr;
    ptr = null;
    if (p.ghost) p.ghost.remove();
    if (p.fromSq !== null) squareEls[p.fromSq].classList.remove('dragging');
    if (cancelled) return;

    if (p.moved) {
      const target = squareAt(e.clientX, e.clientY);
      if (p.fromSq !== null) {
        if (target === null) {
          state.board[p.fromSq] = null;              // dragged off the board
        } else if (target !== p.fromSq) {
          state.board[target] = p.piece;
          state.board[p.fromSq] = null;
        }
      } else if (target !== null) {
        state.board[target] = p.piece;               // from palette
      }
      update();
    } else if (p.fromSq !== null) {
      clickSquare(p.fromSq);
    } else {
      clickPalette(p.piece);
    }
  }

  function clickSquare(sq) {
    if (state.tool === 'erase') {
      state.board[sq] = null;
    } else if (state.tool) {
      state.board[sq] = state.tool;
    } else if (state.selected !== null) {
      if (state.selected !== sq) {
        state.board[sq] = state.board[state.selected];
        state.board[state.selected] = null;
      }
      state.selected = null;
    } else if (state.board[sq]) {
      state.selected = sq;
    }
    update();
  }

  function clickPalette(piece) {
    state.tool = state.tool === piece ? null : piece;
    state.selected = null;
    update();
  }

  boardEl.addEventListener('pointerdown', (e) => {
    const sqEl = e.target.closest('.square');
    if (!sqEl) return;
    const sq = Number(sqEl.dataset.sq);
    const piece = state.board[sq];
    if (piece && !state.tool) {
      startPointer(e, piece, sq);
    } else if (piece && state.tool) {
      // A tool is active, but allow dragging existing pieces too; a plain
      // click still applies the tool.
      startPointer(e, piece, sq);
    } else if (e.button === 0) {
      e.preventDefault();
      clickSquare(sq);
    }
  });

  boardEl.addEventListener('contextmenu', (e) => {
    const sqEl = e.target.closest('.square');
    if (!sqEl) return;
    e.preventDefault();
    const sq = Number(sqEl.dataset.sq);
    if (state.board[sq]) {
      state.board[sq] = null;
      if (state.selected === sq) state.selected = null;
      update();
    }
  });

  paletteEl.addEventListener('pointerdown', (e) => {
    const btn = e.target.closest('.piece-btn');
    if (!btn) return;
    startPointer(e, btn.dataset.piece, null);
  });

  window.addEventListener('pointermove', onPointerMove);
  window.addEventListener('pointerup', (e) => endPointer(e, false));
  window.addEventListener('pointercancel', (e) => endPointer(e, true));

  // ---------- Controls ----------

  eraseBtn.addEventListener('click', () => {
    state.tool = state.tool === 'erase' ? null : 'erase';
    state.selected = null;
    update();
  });

  $('flip').addEventListener('click', () => {
    state.flipped = !state.flipped;
    buildBoard();
  });

  $('reset').addEventListener('click', () => {
    const s = F.parseFen(F.START_FEN);
    state.board = s.board;
    state.turn = s.turn;
    state.castling = s.castling;
    state.selected = null;
    setMsg('');
    update();
  });

  $('clear').addEventListener('click', () => {
    state.board = F.emptyBoard();
    state.turn = 'w';
    state.castling = { K: false, Q: false, k: false, q: false };
    state.selected = null;
    setMsg('');
    update();
  });

  turnEl.addEventListener('change', () => {
    state.turn = turnEl.value;
    syncFen();
  });

  for (const r of Object.keys(castleEls)) {
    castleEls[r].addEventListener('change', () => {
      state.castling[r] = castleEls[r].checked;
      syncFen();
    });
  }

  function loadFen() {
    let parsed;
    try {
      parsed = F.parseFen(fenEl.value);
    } catch (err) {
      setMsg(err.message, 'error');
      return;
    }
    const requested = { ...parsed.castling };
    state.board = parsed.board;
    state.turn = parsed.turn;
    state.castling = parsed.castling;
    state.selected = null;
    update();
    const dropped = Object.keys(requested).filter((k) => requested[k] && !state.castling[k]);
    if (dropped.length) {
      setMsg('Loaded. Castling ' + dropped.join('') + ' dropped: king or rook not on its home square.', 'ok');
    } else {
      setMsg('Position loaded.', 'ok');
    }
  }

  $('load').addEventListener('click', loadFen);
  fenEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); loadFen(); }
  });
  fenEl.addEventListener('focus', () => fenEl.select());

  $('copy').addEventListener('click', async () => {
    const text = fenEl.value;
    try {
      await navigator.clipboard.writeText(text);
    } catch (_) {
      fenEl.select();
      document.execCommand('copy');
    }
    setMsg('FEN copied.', 'ok');
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      state.tool = null;
      state.selected = null;
      update();
      fenEl.blur();
    }
  });

  // ---------- Init ----------

  buildPalette();
  buildBoard();
  update();
})();
