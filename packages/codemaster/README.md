# codemaster: an AI Codenames codemaster from induced word senses

**Status:** ML foundation phase — building the sense-vector pipeline. No results yet.

## Goal

Codenames clues exploit bare-word ambiguity (BANK → "river" or "money"). This project
represents every card word and candidate clue word as a small set of **sense vectors**,
induced without supervision from contextual embeddings (Word Sense Induction), rather than
disambiguated against a fixed sense inventory like WordNet. The eventual codemaster scores
a clue against a board by taking the best-matching sense pair between clue and card.

## Method

1. Harvest real usages of each word from Wikipedia + FineWeb-Edu.
2. Embed each usage with a contextual encoder (BERT-base, BERT-large, DeBERTa-v3) and pool
   to one vector per occurrence.
3. Normalize to counter embedding anisotropy, then cluster occurrences per word (k-means /
   agglomerative) to induce that word's senses.
4. Validate the clustering against the SemEval-2010/2013 WSI benchmarks before committing
   to a full-vocabulary run, and compare against a static GloVe baseline.
5. Export a versioned sense artifact (vectors + manifest) that the online game — numpy
   only, no torch — loads to score clues at play time.

See `CLAUDE.md` in this directory for the full design and `pyproject.toml` for how the
heavy (`pipeline`) and light (`engine`/`agents`/`eval`) dependencies are split.

## Results

Not yet — pipeline stages are being built one at a time. This section will carry the
layer-sweep and SemEval comparison tables from `results/`, plus sense-inspection highlights
from `reports/`, once they exist.
