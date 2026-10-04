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
  const loadBtn = $('load');
  const analyzeBtn = $('analyze');
  const newAnalysisBtn = $('newanalysis');
  const undoBtn = $('undo');
  const redoBtn = $('redo');
  const botColorEl = $('botcolor');
  const newGameBtn = $('newgame');
  const tabs = { analysis: $('tab-analysis'), bot: $('tab-bot') };
  const panels = { analysis: $('analysis-panel'), bot: $('bot-panel') };
  const analysisHint = $('analysis-hint');
  const panelEl = document.querySelector('.panel');
  const evalbarEl = $('evalbar');
  const evaltextEl = $('evaltext');
  const castleEls = { K: $('cK'), Q: $('cQ'), k: $('ck'), q: $('cq') };

  const start = F.parseFen(F.START_FEN);
  const state = {
    board: start.board,
    turn: start.turn,
    castling: start.castling,
    ep: start.ep,    // en passant target square index, or null
    flipped: false,
    tool: null,      // piece code ('K', 'p', ...), 'erase', or null
    selected: null,  // square index awaiting a destination click
    evalCp: null,    // last model evaluation in centipawns (White's view), or null
    evalFen: null,   // FEN that evaluation was computed for
    locked: false,   // true once Load has finalized the position for analysis
    legal: [],       // legal moves in the locked position (side to move)
    lastMove: null,  // { from, to } of the last move played while locked
    history: [],     // snapshots for undo while locked
    redo: [],        // snapshots undone and available for redo
    result: null,    // '1-0' | '0-1' | '1/2-1/2' once the locked game is over
    note: '',        // prefix for the next evaluation message, e.g. "Bot played e7-e5. "
    mode: 'analysis', // 'analysis' (set up + analyze) or 'bot' (play a game against the bot)
    bot: '',         // '' (off), 'w' or 'b': which side the bot plays (bot mode only)
    botThinking: false,
  };
  let typingFen = false;   // true while applying a FEN the user is typing (don't rewrite the field)

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
      el.classList.toggle('last-move', !!state.lastMove && (state.lastMove.from === sq || state.lastMove.to === sq));
      el.classList.toggle('movable', state.locked && !!p && F.pieceColor(p) === state.turn && !state.result);
    }
    const dests = state.locked && state.selected !== null ? movesFrom(state.selected) : [];
    for (let sq = 0; sq < 64; sq++) {
      squareEls[sq].classList.toggle('dest', dests.some((m) => m.to === sq));
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

  // ---------- Eval bar ----------

  // Centipawns -> percentage of the bar that is white, using lichess's
  // win-probability curve so the bar saturates smoothly instead of linearly.
  function whitePercent(cp) {
    const winChance = 2 / (1 + Math.exp(-0.00368208 * cp)) - 1;   // -1 .. 1
    return 50 + 50 * winChance;
  }

  // Centipawns -> pawns with a sign, e.g. 125 -> "+1.25". Two decimals, or
  // one once the number reaches 10 so it still fits in the bar.
  function formatEval(cp) {
    const pawns = Math.abs(cp) / 100;
    const sign = cp > 0 ? '+' : cp < 0 ? '\u2212' : '';
    return sign + pawns.toFixed(pawns >= 10 ? 1 : 2);
  }

  function renderEval() {
    const cp = state.evalCp;
    const stale = cp !== null && state.evalFen !== F.toFen(state) && !state.result;
    evalbarEl.classList.toggle('empty', cp === null);
    evalbarEl.classList.toggle('stale', stale);
    evalbarEl.classList.toggle('flipped', state.flipped);
    if (cp === null) {
      evalbarEl.classList.remove('white-ahead');
      evalbarEl.style.setProperty('--white-pct', '50');
      evalbarEl.removeAttribute('aria-valuenow');
      evaltextEl.textContent = '';
      return;
    }
    evalbarEl.style.setProperty('--white-pct', String(whitePercent(cp)));
    evalbarEl.setAttribute('aria-valuenow', (cp / 100).toFixed(2));
    evalbarEl.classList.toggle('white-ahead', cp >= 0);
    evaltextEl.textContent = state.result ? { '1-0': '1-0', '0-1': '0-1' }[state.result] || '\u00bd' : formatEval(cp);
  }

  let analyzeSeq = 0;

  async function analyzePosition() {
    if (state.result) return;
    const fen = F.toFen(state);
    const seq = ++analyzeSeq;
    analyzeBtn.disabled = true;
    setMsg('Analyzing\u2026');
    try {
      const res = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || ('Server returned ' + res.status));
      if (seq !== analyzeSeq || !state.locked) return;   // a newer request superseded this one
      state.evalCp = data.cp;
      state.evalFen = fen;
      renderEval();
      const who = data.cp > 30 ? 'White is better' : data.cp < -30 ? 'Black is better' : 'Roughly equal';
      let text = 'Evaluation ' + formatEval(data.cp) + ' \u2014 ' + who + '.';
      if (data.line) {
        // The capture search changed the picture: say which captures it played.
        text += ' After ' + data.line + ' (static ' + formatEval(data.static_cp) + ').';
      }
      setMsg(state.note + text, 'ok');
    } catch (err) {
      if (seq === analyzeSeq) setMsg('Analysis failed: ' + err.message, 'error');
    } finally {
      if (seq === analyzeSeq) analyzeBtn.disabled = false;
    }
  }

  // ---------- Sync panel from state ----------

  function syncCastling() {
    const avail = F.castlingAvailability(state.board);
    for (const r of ['K', 'Q', 'k', 'q']) {
      if (!avail[r]) state.castling[r] = false;
      const el = castleEls[r];
      el.disabled = !avail[r] || state.locked;
      el.checked = state.castling[r];
      el.parentElement.classList.toggle('disabled', !avail[r]);
    }
  }

  function syncFen() {
    if (!typingFen) fenEl.value = F.toFen(state);
  }

  function syncWarnings() {
    warningsEl.innerHTML = '';
    for (const w of F.positionWarnings(state.board, state.turn)) {
      const li = document.createElement('li');
      li.textContent = w;
      warningsEl.appendChild(li);
    }
  }

  function setMsg(text, kind) {
    msgEl.textContent = text || '';
    msgEl.className = 'msg' + (kind ? ' ' + kind : '');
  }

  const position = () => ({ board: state.board, turn: state.turn, castling: state.castling, ep: state.ep });
  const movesFrom = (sq) => state.legal.filter((m) => m.from === sq);

  function update() {
    turnEl.value = state.turn;
    state.legal = state.locked && !state.result ? F.legalMoves(position()) : [];
    syncCastling();
    renderPieces();
    renderPalette();
    syncFen();
    syncWarnings();
    renderEval();
  }

  // ---------- Finalize / lock ----------

  function setLocked(locked) {
    state.locked = locked;
    state.tool = null;
    state.selected = null;
    state.lastMove = null;
    state.history = [];
    state.redo = [];
    state.result = null;
    state.botThinking = false;
    undoBtn.disabled = true;
    redoBtn.disabled = true;
    panelEl.classList.toggle('locked', locked);
    boardEl.classList.toggle('locked', locked);
    fenEl.readOnly = locked;
    turnEl.disabled = locked;
    for (const id of ['erase', 'reset', 'clear']) $(id).disabled = locked;
    for (const r of Object.keys(castleEls)) castleEls[r].disabled = locked || castleEls[r].disabled;
    loadBtn.classList.toggle('hidden', locked);
    analyzeBtn.classList.toggle('hidden', !locked);
    newAnalysisBtn.classList.toggle('hidden', !locked);
    for (const el of [undoBtn, redoBtn]) el.classList.toggle('hidden', !locked);
    update();
    syncHistoryButtons();
  }

  // ---------- Modes ----------

  function setMode(mode) {
    if (state.mode === mode) return;
    state.mode = mode;
    for (const m of Object.keys(tabs)) {
      tabs[m].classList.toggle('active', m === mode);
      tabs[m].setAttribute('aria-selected', m === mode ? 'true' : 'false');
      panels[m].classList.toggle('hidden', m !== mode);
    }
    analysisHint.classList.toggle('hidden', mode !== 'analysis');
    if (mode === 'bot') {
      startBotGame();
    } else {
      // Back to the editor, keeping whatever is on the board.
      state.bot = '';
      state.evalCp = null;
      state.evalFen = null;
      setLocked(false);
      setMsg('Edit the position, then press Load.');
    }
  }

  function startBotGame() {
    const choice = botColorEl.value;
    const user = choice === 'random' ? (Math.random() < 0.5 ? 'w' : 'b') : choice;
    const start = F.parseFen(F.START_FEN);
    state.board = start.board;
    state.turn = start.turn;
    state.castling = start.castling;
    state.ep = start.ep;
    state.evalCp = null;
    state.evalFen = null;
    state.bot = user === 'w' ? 'b' : 'w';
    if (state.flipped !== (user === 'b')) {       // your pieces at the bottom
      state.flipped = user === 'b';
      buildBoard();
    }
    setLocked(true);
    setMsg('New game \u2014 you play ' + (user === 'w' ? 'White' : 'Black') +
      (choice === 'random' ? ' (chosen at random)' : '') + '. ' +
      (user === 'w' ? 'Your move.' : 'Bot moves first.'), 'ok');
    maybeBotMove();
  }

  // ---------- Playing moves in the locked position ----------

  const snapshot = () => ({
    board: state.board, turn: state.turn, castling: state.castling, ep: state.ep,
    lastMove: state.lastMove, evalCp: state.evalCp, evalFen: state.evalFen, result: state.result,
  });

  function syncHistoryButtons() {
    undoBtn.disabled = state.history.length === 0 || state.botThinking;
    redoBtn.disabled = state.redo.length === 0 || state.botThinking;
  }

  // Play from -> to if it is legal. Promotions by hand always choose a queen.
  function tryMove(from, to) {
    if (state.botThinking) return false;
    const options = movesFrom(from).filter((m) => m.to === to);
    if (!options.length) return false;
    playMove(options.find((m) => !m.promo || m.promo === 'q'));
    return true;
  }

  function playMove(move) {
    state.history.push(snapshot());
    state.redo = [];
    state.note = '';
    const next = F.makeMove(position(), move);
    state.board = next.board;
    state.turn = next.turn;
    state.castling = next.castling;
    state.ep = next.ep;
    state.lastMove = { from: move.from, to: move.to };
    state.selected = null;
    update();
    syncHistoryButtons();
    afterMove();
  }

  const moveName = (m) => F.sqName(m.from) + (state.board[m.to] ? 'x' : '-') + F.sqName(m.to) + (m.promo ? '=' + m.promo.toUpperCase() : '');

  function afterMove() {
    if (state.legal.length === 0) {
      const inCheck = F.isInCheck(position());
      if (inCheck) {
        state.result = state.turn === 'w' ? '0-1' : '1-0';
        state.evalCp = state.turn === 'w' ? -2500 : 2500;
        setMsg('Checkmate \u2014 ' + (state.turn === 'w' ? 'Black' : 'White') + ' wins.', 'ok');
      } else {
        state.result = '1/2-1/2';
        state.evalCp = 0;
        setMsg('Stalemate \u2014 draw.', 'ok');
      }
      state.evalFen = F.toFen(state);
      update();
      syncHistoryButtons();
      return;
    }
    analyzePosition();
    maybeBotMove();
  }

  function restore(snap) {
    analyzeSeq++;                       // drop any in-flight analysis of the position we are leaving
    analyzeBtn.disabled = false;
    state.note = '';
    Object.assign(state, snap);
    state.selected = null;
    update();
    syncHistoryButtons();
    setMsg(state.evalCp !== null && !state.result ? 'Evaluation ' + formatEval(state.evalCp) + '.' : '', 'ok');
  }

  // Undo one ply; with a bot active, step back to the user's turn (at most two plies).
  function undoMove() {
    if (state.botThinking) return;
    let steps = state.bot && state.turn !== state.bot && state.history.length >= 2 ? 2 : 1;
    while (steps-- > 0 && state.history.length) {
      state.redo.push(snapshot());
      restore(state.history.pop());
    }
  }

  function redoMove() {
    if (state.botThinking) return;
    let steps = 1;
    while (steps-- > 0 && state.redo.length) {
      state.history.push(snapshot());
      restore(state.redo.pop());
      // With a bot active, keep going until it is the user's turn again.
      if (state.bot && state.turn === state.bot && state.redo.length) steps = 1;
    }
  }

  // ---------- Bot ----------

  // Plays the legal move whose resulting position the model scores best for
  // the side to move (highest cp for White, lowest for Black). Mates are
  // taken immediately; stalemate counts as 0.
  async function botMove() {
    if (!state.locked || state.result || state.botThinking || state.legal.length === 0) return;
    const pos = position();
    const mover = state.turn;
    const fenBefore = F.toFen(state);
    const candidates = state.legal.map((m) => {
      const next = F.makeMove(pos, m);
      let fixed = null;                               // decided without the model
      if (F.legalMoves(next).length === 0) fixed = F.isInCheck(next) ? (mover === 'w' ? 2500 : -2500) : 0;
      return { move: m, fen: F.toFen(next), fixed };
    });
    state.botThinking = true;
    syncHistoryButtons();
    setMsg('Bot is thinking\u2026');
    try {
      const need = candidates.filter((c) => c.fixed === null);
      let scores = new Map();
      if (need.length) {
        const res = await fetch('/api/evaluate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ fens: need.map((c) => c.fen) }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.error || ('Server returned ' + res.status));
        need.forEach((c, i) => scores.set(c, data.results[i].cp));
      }
      // The position may have changed while we waited (undo, new analysis).
      if (!state.locked || F.toFen(state) !== fenBefore) return;
      let best = null;
      let bestScore = null;
      for (const c of candidates) {
        const cp = c.fixed !== null ? c.fixed : scores.get(c);
        const score = mover === 'w' ? cp : -cp;         // higher is better for the mover
        if (best === null || score > bestScore) { best = c; bestScore = score; }
      }
      const cp = best.fixed !== null ? best.fixed : scores.get(best);
      const name = moveName(best.move);
      state.botThinking = false;
      playMove(best.move);
      state.note = 'Bot played ' + name + '. ';
      if (state.result) setMsg(state.note + msgEl.textContent, 'ok');
      else setMsg(state.note + 'Analyzing\u2026');
    } catch (err) {
      setMsg('Bot failed: ' + err.message, 'error');
    } finally {
      state.botThinking = false;
      syncHistoryButtons();
    }
  }

  function maybeBotMove() {
    if (state.bot && state.bot === state.turn) botMove();
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
      if (state.locked) {
        if (target !== null && target !== p.fromSq && !tryMove(p.fromSq, target)) {
          state.selected = null;
          update();
        }
        return;
      }
      state.ep = null;                                   // position edited by hand
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
    if (state.locked) {
      const p = state.board[sq];
      if (state.selected !== null && state.selected !== sq && tryMove(state.selected, sq)) return;
      state.selected = p && F.pieceColor(p) === state.turn && !state.result && state.selected !== sq ? sq : null;
      update();
      return;
    }
    state.ep = null;                                     // position edited by hand
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
    if (state.locked) {
      if (state.result || state.botThinking) return;
      if (piece && F.pieceColor(piece) === state.turn) {
        startPointer(e, piece, sq);                      // drag or click-select own piece
      } else if (e.button === 0) {
        e.preventDefault();
        clickSquare(sq);                                 // destination click (or deselect)
      }
      return;
    }
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
    if (state.locked) return;
    const sq = Number(sqEl.dataset.sq);
    if (state.board[sq]) {
      state.ep = null;
      state.board[sq] = null;
      if (state.selected === sq) state.selected = null;
      update();
    }
  });

  paletteEl.addEventListener('pointerdown', (e) => {
    if (state.locked) return;
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
    renderEval();
  });

  analyzeBtn.addEventListener('click', analyzePosition);

  $('reset').addEventListener('click', () => {
    const s = F.parseFen(F.START_FEN);
    state.board = s.board;
    state.turn = s.turn;
    state.castling = s.castling;
    state.ep = s.ep;
    state.selected = null;
    setMsg('');
    update();
  });

  $('clear').addEventListener('click', () => {
    state.board = F.emptyBoard();
    state.turn = 'w';
    state.castling = { K: false, Q: false, k: false, q: false };
    state.ep = null;
    state.selected = null;
    setMsg('');
    update();
  });

  turnEl.addEventListener('change', () => {
    state.turn = turnEl.value;
    state.ep = null;
    syncFen();
    syncWarnings();
    renderEval();
  });

  for (const r of Object.keys(castleEls)) {
    castleEls[r].addEventListener('change', () => {
      state.castling[r] = castleEls[r].checked;
      syncFen();
      renderEval();
    });
  }

  // Parse the FEN field and apply it to the board. Returns the parsed
  // castling the user asked for, or null (with an error shown) if invalid.
  function applyFenField() {
    let parsed;
    try {
      parsed = F.parseFen(fenEl.value);
    } catch (err) {
      setMsg(err.message, 'error');
      return null;
    }
    const requested = { ...parsed.castling };
    state.board = parsed.board;
    state.turn = parsed.turn;
    state.castling = parsed.castling;
    state.ep = parsed.ep;
    state.selected = null;
    update();
    return requested;
  }

  // Typing in the FEN field updates the board immediately.
  fenEl.addEventListener('input', () => {
    if (state.locked) return;
    typingFen = true;
    try {
      if (applyFenField() !== null) setMsg('');
    } finally {
      typingFen = false;
    }
  });
  // Once the user leaves the field, show the normalized FEN.
  fenEl.addEventListener('blur', () => { if (!state.locked) syncFen(); });

  // Load finalizes the position: editing is locked and analysis becomes available.
  function loadFen() {
    const requested = applyFenField();
    if (requested === null) return;
    const problems = F.positionWarnings(state.board, state.turn);
    if (problems.length) {
      setMsg('Fix the position before loading: ' + problems.join('; ') + '.', 'error');
      return;
    }
    setLocked(true);
    const dropped = Object.keys(requested).filter((k) => requested[k] && !state.castling[k]);
    if (dropped.length) {
      setMsg('Position locked in. Castling ' + dropped.join('') + ' dropped: king or rook not on its home square.', 'ok');
    } else {
      setMsg('Position locked in. Click Analyze position, or play a move.', 'ok');
    }
  }

  function newAnalysis() {
    state.bot = '';
    state.evalCp = null;
    state.evalFen = null;
    setLocked(false);
    setMsg('Edit the position, then press Load.');
  }

  loadBtn.addEventListener('click', loadFen);
  newAnalysisBtn.addEventListener('click', newAnalysis);
  undoBtn.addEventListener('click', undoMove);
  redoBtn.addEventListener('click', redoMove);
  tabs.analysis.addEventListener('click', () => setMode('analysis'));
  tabs.bot.addEventListener('click', () => setMode('bot'));
  newGameBtn.addEventListener('click', startBotGame);
  fenEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); if (!state.locked) loadFen(); }
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
    const typing = ['INPUT', 'SELECT', 'TEXTAREA'].includes((e.target.tagName || '').toUpperCase());
    if (state.locked && !typing) {
      if (e.key === 'ArrowLeft') { e.preventDefault(); undoMove(); return; }
      if (e.key === 'ArrowRight') { e.preventDefault(); redoMove(); return; }
    }
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
