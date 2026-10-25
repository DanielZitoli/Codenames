"""codemaster: WSI pipeline, sense artifacts, AI agents, and evaluation.

Structure (see packages/codemaster/CLAUDE.md for the full design):
- pipeline/: offline, heavy (torch, transformers) — produces versioned sense artifacts.
- engine/, agents/, eval/: online, light (numpy + codenames_core) — load an artifact and
  pick clues/guesses with matrix maths. Must never import torch or pipeline/.
"""
