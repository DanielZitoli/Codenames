# Codenames — project overview

A clone of the board game Codenames with two goals that share one codebase:

1. **An ML project:** learn ML concepts by building an AI codemaster (spymaster) from
   contextual word embeddings. It must stand on its own as a research project with its own
   results and write-up.
2. **A party-game website:** a responsive site for playing with friends in the style of
   Jackbox.tv. A shared screen shows the board, players join from their phones with a room
   code, and the AI can fill a spymaster seat.

The website **uses** the ML project; the ML project never depends on the website.

The owner is an experienced solo developer who is newer to multi-package repos and ML
research tooling. Prefer simple, readable code, and briefly explain structural or ML
decisions when you introduce them.

Each package has its own `CLAUDE.md` with details. Read the one for the package you're
working in.

---

## Repository layout

```
codenames/
├── CLAUDE.md                  # this file: structure, goals, boundaries
├── pyproject.toml             # uv workspace root (members: packages/*, apps/server)
├── packages/
│   ├── core/                  # codenames_core: rules, board, clue legality, shared schemas
│   └── codemaster/            # the ML project (see packages/codemaster/CLAUDE.md)
├── apps/
│   ├── server/                # game server: rooms, players, turns, websockets (Python)
│   └── web/                   # frontend: host screen + phone controller (TypeScript)
├── services/
│   └── ai/                    # LATER, optional: HTTP wrapper around codemaster agents
└── .gitignore                 # data downloads, artifacts/, node_modules, .venv, etc.
```

| Package | Language | Depends on | Purpose |
|---|---|---|---|
| `packages/core` | Python | nothing ML (pydantic only) | Single source of truth for game rules and data types |
| `packages/codemaster` | Python | `core`; torch etc. only in its `pipeline` extra | WSI pipeline, sense artifacts, AI agents, evaluation |
| `apps/server` | Python (FastAPI + websockets) | `core`, `codemaster` (base install only) | Authoritative real-time game server |
| `apps/web` | TypeScript | server's API/websocket schema | Host screen and phone UI |

---

## Boundaries (the rules that keep the pieces independent)

1. **Dependencies point one way:** `web → server → codemaster → core`. Nothing imports
   "upward". `core` imports nothing from the other packages, and `codemaster` never
   imports from `apps/`.
2. **The website never runs a neural network.** `codemaster` has two halves:
   - `pipeline/`: offline and heavy (torch, transformers). Runs on the owner's MacBook and
     produces a versioned **sense artifact** (`senses-{model}-vN.npz` + `manifest.json`).
   - `engine/`, `agents/`, `eval/`: online and light (numpy + `core`). They load the
     artifact and pick clues with matrix maths.

   The server installs `codemaster` **without** the `pipeline` extra. A test in
   `codemaster` enforces that the online modules never import torch or `pipeline/`.
3. **Rules live in one place.** Board generation, turn order, clue legality and win
   conditions are implemented only in `core`. The server, the self-play simulator and the
   agents all call `core`; none re-implement rules.
4. **Agents only see what their role may see.** `core` defines role-specific views:
   ```python
   class SpymasterView(BaseModel): ...   # all cards + the key (team colours, assassin)
   class OperativeView(BaseModel): ...   # all cards + revealed colours only

   class Codemaster(Protocol):
       def give_clue(self, view: SpymasterView) -> Clue: ...
   class Guesser(Protocol):
       def guess(self, view: OperativeView, clue: Clue) -> list[str]: ...
   ```
   Human players and AI agents receive exactly the same views.
5. **The server is authoritative.** It holds the full game state, validates every clue and
   guess with `core`, and sends each client only the view for its role. **The key is never
   sent to the host screen or to operative phones**, only to spymaster phones.
6. **Artifacts are versioned files, not code.** They're published outside git (GitHub
   Releases or the Hugging Face Hub). The server pins an artifact version in config and
   downloads it at startup. Swapping models means changing that pin.

---

## Website design (Jackbox-style)

- **Host screen** (laptop or TV): creates a room, shows a short room code and the 5×5
  board, and shows the game state. It's a public display, so it gets the operative view.
- **Phones:** players open the site, enter the room code and a nickname (no accounts),
  then pick a team and role. Spymaster phones show the key and a clue entry box; operative
  phones let the team discuss and tap guesses.
- **Real-time:** websockets between server and every client. Server events are typed with
  the `core` schemas; generate TypeScript types from the server's OpenAPI / JSON Schema
  rather than hand-writing them.
- **Game modes:**
  - friends only (all human);
  - AI spymaster for one or both teams, with humans guessing;
  - later: AI guessers, and model choice as a difficulty level (e.g. GloVe = easy,
    DeBERTa = hard).
- **AI integration:** the server calls agents through a small `AIClient` interface.
  Start with an in-process implementation (import `codemaster.agents` directly). Move to
  `services/ai` over HTTP only if the AI needs to scale or deploy separately.
- Room state lives in server memory to start. Rooms expire after inactivity. A database
  is out of scope until there's a reason for one.

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1. ML foundation | WSI sense vectors for BERT-base, BERT-large, DeBERTa-v3 + GloVe baseline; SemEval evaluation; artifact export | **Current** (see `packages/codemaster/CLAUDE.md`) |
| 2. Rules + agents | `core` rules and views; codemaster clue search and scoring; guesser bot built on a *different* model; self-play win-rate evaluation; sweep clue-vocabulary size N | Next |
| 3. Playable site | Server rooms and websockets; host screen and phone UI; friends-only mode, then AI spymaster mode | After 2 (the `core` part of phase 2 can start in parallel) |
| 4. Polish | Difficulty levels by model; optional LLM clue reranker experiment; deployment | Later |

Don't start work from a later phase unless the owner asks.

---

## Open decisions (ask before assuming otherwise)

- **Server language:** Python (FastAPI + websockets) is the current plan because it keeps
  one rules implementation. Switching to a Node/TypeScript server would mean duplicating
  the rules and moving the AI behind `services/ai`.
- **Frontend framework:** not chosen yet. Must be mobile-first and responsive.
- **Hosting/deployment:** not chosen yet.

---

## Environment and conventions

- Python 3.11+ with a `uv` workspace at the repo root. The ML pipeline needs
  `uv sync --extra pipeline` inside `packages/codemaster`; everything else uses the base
  install.
- ML work runs on macOS Apple Silicon (PyTorch `mps` device).
- Tests: `pytest` per Python package; add frontend tests once a framework is chosen.
- Never commit data downloads, model weights, embeddings or artifacts.
- Keep the shared types in `core` as the contract. Changing a view or event schema means
  updating the server, the frontend types and any agents in the same change.
- **Ask before:** downloads over ~1 GB, runs expected to take more than ~30 minutes, new
  heavy dependencies, or changing a boundary listed above.
