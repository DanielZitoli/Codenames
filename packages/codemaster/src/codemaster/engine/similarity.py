"""Shared similarity API over an exported sense artifact (CLAUDE.md, Stage 7 / "Shared
similarity API").

`SenseIndex` is the only interface the (future) codemaster/guesser agents depend on. It
loads the `.npz` + manifest that `codemaster.pipeline.export` (Stage 7) produces and never
touches raw embeddings or models — numpy only.

Not implemented yet: the artifact format is fixed by this signature, but the loader and
`sim`/`sim_matrix` bodies are built once Stage 7 exists to produce a real artifact to load.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class SenseIndex:
    """Looks up L2-normalized sense vectors for a word and scores clue/card similarity.

    One instance per model output (e.g. a `bert-base` index and a separate `glove` index).
    """

    def __init__(self, artifact_path: str | Path) -> None:
        self.artifact_path = Path(artifact_path)
        raise NotImplementedError(
            "SenseIndex loading depends on the Stage 7 export format; "
            "not built yet (see packages/codemaster/CLAUDE.md)."
        )

    def senses(self, word: str) -> np.ndarray:
        """Return this word's sense centroids, shape (k, d), L2-normalized."""
        raise NotImplementedError

    def sim(self, clue: str, card: str) -> float:
        """max over (s_clue in senses[clue], s_card in senses[card]) of cos(s_clue, s_card)."""
        raise NotImplementedError

    def sim_matrix(self, clues: list[str], cards: list[str]) -> np.ndarray:
        """Vectorized `sim` over all (clue, card) pairs, shape (len(clues), len(cards))."""
        raise NotImplementedError
