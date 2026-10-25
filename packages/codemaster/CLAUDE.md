# codemaster — ML package (WSI embedding engine)

> This file covers `packages/codemaster/`. The repo-wide structure, goals and boundaries are
> in the root `CLAUDE.md`. This package must work as a **standalone ML project**: it runs,
> tests and evaluates without the web app. The web app depends on it, never the reverse.

## Project context

We're building a Codenames clone with an AI codemaster. This package's **current phase** is
the ML foundation: a pipeline that represents every card word and every candidate clue word
as a **set of sense vectors**, induced without supervision (Word Sense Induction, WSI) from
contextual embeddings. We then compare encoders against each other and against a static
GloVe baseline.

The package has two halves with very different dependencies:
- **`pipeline/` (offline, heavy):** torch, transformers, datasets. Runs on the owner's
  MacBook and produces a versioned **sense artifact** (a file of vectors plus a manifest).
- **`engine/`, `agents/`, `eval/` (online, light):** numpy and `codenames_core` only. They
  load the artifact and do all game-time work with matrix maths. This is the code the live
  website runs, so it must never import from `pipeline/` or from torch.

The owner is an experienced solo developer who is newer to ML research tooling. Prefer
readable, well-structured code over clever code, and briefly explain non-obvious ML design
choices when you introduce a new module.

### Why WSI, not WSD

Codenames cards are bare words with no context, and their ambiguity is what clues exploit:
BANK can be reached through "river" or through "money." We are not disambiguating a usage;
we are **enumerating the senses** a word has. WSD needs a fixed sense inventory (WordNet),
which is too fine-grained ("bank" has 10 noun senses) and misses senses players actually use
(AMAZON the company, MERCURY the planet). WSI clusters real usages from a corpus, so it
finds the senses that occur in practice.

### How the senses will be used (next phase; shape the outputs for this)

```
sim(clue, card) = max over (s_clue in senses[clue], s_card in senses[card]) of cos(s_clue, s_card)
```

The codemaster will choose a clue and a target subset of team cards, maximizing the margin
between the weakest target similarity and the strongest similarity to any opponent, neutral
or assassin card (with the assassin weighted heavily). Using max over senses on the danger
cards is deliberately pessimistic: if *any* sense of a clue lands near the assassin, a human
might go there.

### Out of scope for now

Codemaster search and scoring (`agents/`), the guesser bot, self-play evaluation, and
LLM-based approaches. These are the next phase; define clean interfaces for them, but don't
build them yet. The game rules engine and board-view schemas belong to `packages/core`
(`codenames_core`), not to this package.

---

## Models under comparison

| ID | Hugging Face checkpoint | Layers | Notes |
|---|---|---|---|
| `bert-base` | `bert-base-uncased` | 12 | Most-used model in the WSI literature; reference point |
| `bert-large` | `bert-large-uncased` | 24 | About 3× slower than base |
| `deberta-v3` | `microsoft/deberta-v3-large` | 24 | Strongest published WSI results among encoders. Use `microsoft/deberta-v3-base` for fast dev iteration |
| `glove` | GloVe `glove.840B.300d` (Common Crawl, cased) | n/a | Static baseline: one vector per word; `glove.6B.300d` is fine for dev |

Model-specific gotchas:

- **DeBERTa-v3 has no masked-LM head.** It was pretrained with replaced-token detection,
  so it can't do fill-mask. Any MLM-based step (cluster labelling) must use `bert-base-uncased`.
- **DeBERTa-v3 needs extra packages.** Its fast tokenizer requires `sentencepiece` and
  `protobuf`. Verify that `return_offsets_mapping=True` works with it, and cover it in tests.
- **Casing differs.** The BERT models are uncased and DeBERTa-v3 is cased. Feed the original
  sentence text to every model; target matching is case-insensitive.

---

## Data

### 1. Card words: `data/cards.txt`

These are the ~400 Codenames card words, one per line, supplied by the owner. Don't generate
or download this list. Some entries may be multi-word (e.g. "ICE CREAM"), and the matcher and
pooling code must handle multi-word spans.

### 2. Clue vocabulary: `data/clue_vocab.txt` (generated)

- Source: the `wordfreq` library, `top_n_list("en", 50_000)`.
- Filters: alphabetic only, length ≥ 3, drop stopwords, collapse inflections to lemmas with
  `lemminflect` (keep the most frequent surface form), and drop a small profanity blocklist.
  Do **not** filter by part of speech; verbs and adjectives are legal clues.
- Do **not** remove card words globally. The Codenames rule only bans clues related to
  *visible* cards, so board-specific filtering (same lemma, shared stem, substring) happens
  at game time, not here.
- Keep the top **N** after filtering, with **N = 10,000 by default**, configurable. We'll
  later sweep N (5k, 10k, 20k, 40k) against win rate.
- Store log-frequency alongside each word; the codemaster will use it as a familiarity penalty.

### 3. Context corpora (for harvesting usages)

Stream these; never download a full dump.

- **English Wikipedia:** `wikimedia/wikipedia`, config `20231101.en` (check for a newer
  snapshot). Streaming mode via `datasets`.
- **FineWeb-Edu sample:** `HuggingFaceFW/fineweb-edu`, config `sample-10BT`. Web text adds
  everyday and colloquial senses that encyclopedic text underrepresents.
- Interleave the two sources (default 50/50) so sense coverage isn't biased toward one register.
- Verify dataset IDs and configs on the Hub before implementing.

### 4. GloVe vectors

These come from the Stanford GloVe site. Downloads are large: ask the owner before downloading.
After download, convert once to a compact `.npy` plus vocab file restricted to the card and
clue vocabulary, so experiments load in seconds.

### 5. WSI evaluation benchmarks (SemEval)

We use these to validate that our clustering actually separates senses, and to choose the
layer config and clustering hyperparameters **before** the full vocabulary run.

**SemEval-2010 Task 14 (WSI & Disambiguation)**
- 100 target lemmas (50 nouns, 50 verbs), 8,915 test instances. Each instance has a
  single gold sense.
- The official task also provided a large unannotated training set per lemma. We follow the
  common BERT-era protocol instead: cluster the test instances of each lemma directly.
- Unsupervised metrics are **paired F-score (F-S)** and **V-measure (V-M)**. Report their
  geometric mean as **AVG**.
- The two metrics pull in opposite directions. Putting everything in one cluster scores
  well on F-S, and giving each instance its own cluster scores well on V-M. Always report
  both, plus both trivial baselines.

**SemEval-2013 Task 13 (WSI for Graded and Non-Graded Senses)**
- 50 lemmas (20 nouns, 20 verbs, 10 adjectives), 4,664 test instances drawn from the Open
  American National Corpus, with 22–100 instances per lemma.
- Instances can carry **multiple weighted WordNet senses** (graded annotation).
- Clustering metrics are **Fuzzy B-Cubed (FBC)** and **Fuzzy NMI (FNMI)**. Report their
  geometric mean as **AVG**. FBC favours one cluster, and FNMI favours one-per-instance, so
  report the trivial baselines here too.
- Ignore the 8-lemma trial set; it was annotated differently from the test set.

**Getting the data and scorers**
- Official scorers for both tasks are Java JARs; call them via `subprocess`. Java must be
  installed; check with `java -version` and tell the owner if it's missing.
- Amrami & Goldberg's public WSI repository (`asafamr/bertwsi` on GitHub) has scripts for
  fetching both datasets and scorers. Use it as a reference for paths and formats.
- If you write a Python reimplementation of any metric, it must reproduce the official
  scorer's output on our predictions before we trust it.

---

## Pipeline

Each stage is a CLI subcommand (`python -m codemaster.pipeline <stage> --config configs/<name>.yaml`),
run from `packages/codemaster/`. Every stage supports `--limit` for smoke tests and writes
its outputs to `artifacts/` (gitignored).

### Stage 0: Vocabulary
Build `clue_vocab.txt` (see Data §2). Load `cards.txt`. Precompute a map from surface form to
lemma for every target word (including inflections from `lemminflect`) and a set of
multi-word phrases.

### Stage 1: Harvest contexts (shared across all models)
- Stream the interleaved corpora, split into sentences (spaCy `sentencizer` only, no full
  pipeline), lowercase-tokenize with a simple regex, and look up targets by set membership.
  Don't run a lemmatizer per sentence; it's too slow.
- Use **reservoir sampling** per target so early documents don't dominate:
  `max_contexts = 500` for card words and 200 for clue words.
- Stop when every card word is full, or when a document budget (configurable) runs out.
- Store a record per occurrence: `word, lemma, sentence, char_start, char_end, source`.
  Write to Parquet, sharded by word.
- **All models embed exactly the same contexts**, so comparisons are fair. Fix the random seed.
- Log per-word context counts. Words with fewer than `min_contexts = 50` get a single sense
  later.

### Stage 2: Embed occurrences
For each occurrence and each model:
1. Tokenize with `return_offsets_mapping=True`, then truncate to 128 subword tokens
   **centred on the target span**.
2. Map the target's character span to subword indices using the offsets (multi-word spans
   included).
3. Run the model with `output_hidden_states=True`, on the `mps` device, batched.
4. Pool by **averaging the target's subword pieces**, then averaging over the configured
   layer set.

Run fp32 first. Only try fp16 after confirming it produces no NaNs and matches fp32 within
tolerance.

**Layer selection happens on SemEval first.** For the ~13.5k SemEval instances, save all
hidden layers and sweep single layers plus common combinations (last 4, middle 4, all).
Only the winning layer config per model is used for the full vocabulary run; we never store
every layer for the full vocabulary.

### Stage 3: Normalize
Contextual vectors are anisotropic: a few "rogue" dimensions dominate cosine similarity.
- For each model and layer config, compute per-dimension mean and std from a random sample
  of about 50k occurrence vectors. Save to `artifacts/norm/{model}_{layers}.npz`.
- Transform: standardize each dimension, then L2-normalize.
- Report mean pairwise cosine between random words before and after; it should drop
  sharply. This is a required sanity check.

### Stage 4: Cluster (induce senses)
Implement two methods, both choosing the number of clusters automatically:
- **k-means** for k = 2…6, picking the k with the best silhouette score. Silhouette can't
  score k = 1, so accept k ≥ 2 only if the best silhouette exceeds a threshold τ; otherwise
  the word has one sense.
- **Agglomerative clustering**, average linkage, cosine distance, with a distance threshold.

τ and the distance threshold are tuned on SemEval (AVG), then frozen for the full run.

Post-processing: drop clusters smaller than 5% of the word's occurrences and reassign their
members to the nearest surviving centroid. Words with fewer than `min_contexts` occurrences
get one sense (the mean).

Outputs per model, in `artifacts/senses/{model}.parquet` (or `.npz` + index):
`word, sense_id, centroid, count, share`, plus a **type vector** (mean of all occurrences),
which serves as a "BERT-distilled static embedding" baseline.

Storage: keep fp16 occurrence vectors for **card words only**, for later analysis. For clue
words, keep centroids and counts and discard raw vectors after clustering.

### Stage 5: GloVe baselines
- **Game side:** each word has a single "sense", its GloVe vector, normalized the same way.
- **SemEval side:** GloVe can't embed an occurrence by itself, so represent each instance as
  the mean GloVe vector of the content words in a ±10-word window around the target. That's
  the classic pre-BERT WSI baseline. Then cluster with the same Stage 4 code.

### Stage 6: Inspection report
Write `reports/{model}_senses.md` for a configurable word list, defaulting to all card words.
For each word, list each sense with:
- its share,
- 3 exemplar sentences (the occurrences nearest the centroid),
- the top 8 MLM substitutes from `bert-base-uncased`: mask the target in ~20 member
  sentences and aggregate the fill-ins,
- the top 10 nearest clue-vocabulary senses.

Spot-check list to always include: BANK, PITCH, SPRING, MERCURY, AMAZON, BAT, CRANE, SCALE,
PLATE, ROUND (where present in `cards.txt`).

### Stage 7: Export the sense artifact
This stage produces the only thing the website consumes. For each model, write:
- `senses-{model}-v{N}.npz` containing: `vectors` (all sense centroids stacked, fp16,
  L2-normalized, shape `(total_senses, d)`), `word_ids` (a sense-row → word-index map),
  `words`, `log_freq`, `is_card`, and `offsets` (each word's slice of rows).
- `senses-{model}-v{N}.manifest.json` containing: model checkpoint, layer config,
  clustering method and thresholds, N, corpus budget, git commit, creation date, SemEval
  AVG scores, and a SHA-256 of the `.npz`.

The artifact version `vN` increments whenever the format or contents change. The loader in
`engine/` checks the manifest's format version and refuses incompatible files with a clear
error. Artifacts are published outside git (GitHub Releases or the Hugging Face Hub); never
commit them.

### Shared similarity API: `src/codemaster/engine/similarity.py`
numpy only. It loads a Stage 7 artifact; it never touches raw embeddings or models.
```python
class SenseIndex:
    def senses(self, word: str) -> np.ndarray: ...          # (k, d), L2-normalized
    def sim(self, clue: str, card: str) -> float: ...       # max over sense pairs
    def sim_matrix(self, clues: list[str], cards: list[str]) -> np.ndarray: ...  # vectorized
```
Implement one `SenseIndex` per model output (BERT models, DeBERTa, GloVe, type-vector
variants). The future codemaster depends only on this interface.

---

## Experiments to produce in this phase

1. **Layer sweep:** model × layer config on SemEval 2010 and 2013, using k-means with
   default τ. Output to `results/layer_sweep.csv`.
2. **Clustering comparison:** k-means vs. agglomerative, with tuned thresholds, at each
   model's best layer config. Output to `results/semeval.csv`, with columns
   `model, layers, method, F-S, V-M, AVG10, FBC, FNMI, AVG13`, plus rows for the trivial
   baselines and the GloVe context-average baseline.
3. **Full vocabulary run** for each model with its frozen best config, plus the Stage 6
   reports.
4. **Summary:** `results/README.md` with tables, average number of senses per card word per
   model, runtime per model on this machine, and observations from the spot-check words.

---

## Package layout

```
packages/codemaster/
├── CLAUDE.md                # this file
├── README.md                # standalone write-up: goal, method, results (portfolio-facing)
├── pyproject.toml           # base deps light; heavy deps in the `pipeline` extra
├── configs/                 # YAML: model, layers, clustering, corpus budget, N, seeds
├── data/                    # cards.txt (owner-supplied, committed), clue_vocab.txt,
│                            # external downloads (gitignored)
├── artifacts/               # contexts/, embeddings/, norm/, senses/, export/ (gitignored)
├── results/                 # CSVs and findings README (committed)
├── reports/                 # per-model sense inspection markdown (committed)
├── src/codemaster/
│   ├── pipeline/            # OFFLINE: vocab, harvest, embed, normalize, cluster, glove,
│   │                        # report, semeval, export  (torch/transformers allowed here only)
│   ├── engine/              # ONLINE: artifact loader, SenseIndex (numpy only)
│   ├── agents/              # ONLINE: Codemaster / Guesser implementations (next phase)
│   └── eval/                # self-play simulator (next phase; numpy + codenames_core)
└── tests/
```

## Environment

- macOS on Apple Silicon (MacBook Pro). Use the PyTorch `mps` device, falling back to CPU.
- Python 3.11+, managed with `uv`. This package is a member of the repo's `uv` workspace.
- Dependencies are split in `pyproject.toml`:
  - base: `numpy`, `codenames-core` (workspace dependency), `pyyaml`
  - extra `pipeline`: `torch transformers datasets sentencepiece protobuf wordfreq
    lemminflect spacy scikit-learn pyarrow tqdm`
  - dev: `pytest`
- Offline work: `uv sync --extra pipeline`. The website installs only the base package.
- Expect BERT-large and DeBERTa-v3-large to be roughly 3× slower than base models. Report
  measured throughput after the first smoke run so we can size the corpus budget.

## Working agreement

- Build one stage at a time. Get each stage passing a `--limit` smoke test before moving on.
- **Ask before** any download larger than ~1 GB, before running anything expected to take
  more than ~30 minutes, and before adding new heavy dependencies.
- Never commit `data/` downloads or `artifacts/`; keep `.gitignore` up to date.
- All randomness is seeded from config. Every run writes a `run.json` with the config, git
  commit, model, timings and counts.
- **Required tests:**
  - offset-to-subword alignment for all three tokenizers, covering punctuation, casing,
    inflections and multi-word targets;
  - centred truncation keeps the target span;
  - normalization stats round-trip;
  - clustering on a synthetic two-blob dataset returns k = 2, and on one blob returns k = 1;
  - **dependency boundary:** importing `codemaster.engine`, `codemaster.agents` and
    `codemaster.eval` must not import `torch`, `transformers` or anything in
    `codemaster.pipeline` (check `sys.modules` after import);
  - an exported artifact round-trips through the `engine/` loader, and the manifest hash
    matches.
- When results look surprising (for example, GloVe beating BERT), report it plainly. Don't
  tune until it looks "right".

## Key references

- Amrami & Goldberg (2018, 2019): substitute-based WSI with BERT.
- Eyal et al. (2022), *Large Scale Substitution-based Word Sense Induction*: full-vocabulary
  WSI over Wikipedia.
- *Word Sense Induction with Hierarchical Clustering and Mutual Information Maximization*
  (2022): layer and model sweeps on SemEval 2010 and 2013.
- Manandhar et al. (2010), SemEval-2010 Task 14; Jurgens & Klapaftis (2013), SemEval-2013
  Task 13.
- Ethayarajh (2019), anisotropy; Timkey & van Schijndel (2021), rogue dimensions.
- Bommasani, Davis & Cardie (2020): pooling contextual vectors into static ones.
- Koyyalagunta et al. (2021), *Playing Codenames with Language Graphs and Word Embeddings*;
  Kim et al. (2019), *Cooperation and Codenames*.
