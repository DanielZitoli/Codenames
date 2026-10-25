# services/ai

LATER, optional: HTTP wrapper around `codemaster` agents, for use only if the AI needs
to scale or deploy separately from `apps/server`. Until then, the server calls agents
in-process (see root `CLAUDE.md` → "AI integration").
