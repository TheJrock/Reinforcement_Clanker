# UI Build Plan — Family Reunion Demo

**Deadline:** ~1 week  
**Target games:** Connect Four, Mancala, Checkers  
**Stack:** Flask (backend) + vanilla JS/CSS (frontend)  
**Architecture principle:** build the full stack once around Connect Four, then each new game is just a new board renderer.

---

## Day 1 — Backend Foundation

- [ ] Create `web/` directory and `web/app.py` with Flask skeleton
- [ ] Create `web/agents.py` — load all recommended `.pth` models at startup into a dict
- [ ] Wire `GET /` route → placeholder `index.html`
- [ ] Wire `GET /game/<name>` route → placeholder `game.html`
- [ ] Wire `POST /api/new_game` — init game env in session, return initial state as JSON
- [ ] Wire `POST /api/move` — accept human move, step env, call agent, return new state + AI move + game status as JSON
- [ ] Test backend routes manually (curl or Postman) with Connect Four before touching the frontend

---

## Day 2 — Connect Four: JS + Board Rendering

- [ ] Create `static/js/game.js` — core game loop (new_game on load, post move, handle response)
- [ ] Create `static/js/boards/connect4.js` — `render(state)` draws a 6×7 CSS grid
- [ ] Wire click handlers on columns → triggers `POST /api/move`
- [ ] Disable input while waiting for AI response
- [ ] Display game status (whose turn, win/loss message)
- [ ] Full human vs AI game playable end-to-end

---

## Day 3 — Connect Four: CSS Polish

- [ ] Layout: centered board, clean header, status bar
- [ ] Piece colors (red/yellow), empty cell styling
- [ ] Drop animation (CSS transition on piece appearing)
- [ ] Win highlight (flash or color the winning 4)
- [ ] New Game button resets board without page reload
- [ ] Looks good — you'd be happy showing this to someone

---

## Day 4 — Mancala

- [ ] Add Mancala env + agent to `agents.py`
- [ ] Create `static/js/boards/mancala.js` — SVG oval layout (6 pits per side + 2 stores)
- [ ] Render pit counts as numbers inside pit circles
- [ ] Click handler on your 6 pits → legal move validation → POST move
- [ ] Highlight whose turn it is; handle bonus-turn logic (AI may move multiple times)
- [ ] Display stores prominently (score at a glance)
- [ ] Style with `board.css` — distinct look from Connect Four

---

## Day 5 — Checkers

- [ ] Add Checkers env + agent to `agents.py`
- [ ] Create `static/js/boards/checkers.js` — 8×8 SVG or CSS grid, dark/light squares
- [ ] Render pieces (circles), kings (crowned or double-circle)
- [ ] Two-step move input: click piece to select → click destination to move
- [ ] Highlight legal destinations after piece selection
- [ ] Handle multi-jump sequences if env requires them
- [ ] Style consistently with the rest of the UI

---

## Day 6 — Landing Page + Overall Polish

- [ ] `index.html` — game picker with a card per game (name, brief description, "Play" button)
- [ ] Shared header/nav so all pages feel cohesive
- [ ] Consistent color palette and font across all three games
- [ ] Mobile-friendly basics (board scales on small screens, buttons are tappable)
- [ ] Error handling in JS (network failure, illegal move response)
- [ ] Test all three games start-to-finish, including win/loss/draw states

---

## Day 7 — Buffer / Stretch Goals

- [ ] AI vs AI watch mode (auto-play with a delay between moves)
- [ ] Move history sidebar
- [ ] Difficulty selector (random agent vs trained agent)
- [ ] Deploy to Render or Streamlit Cloud so it's accessible without running locally
- [ ] Any remaining polish from Days 1–6

---

## Deferred (Post-Reunion)

- User accounts + opponent profiling
- Remaining games (Gomoku, Othello, Backgammon, UTTT)
- Phase 2 (MCTS/AlphaZero) agent integration
