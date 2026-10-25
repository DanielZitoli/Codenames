# results/

Committed experiment outputs (CSVs + findings), per CLAUDE.md "Experiments to produce in
this phase":

- `layer_sweep.csv` — model × layer config on SemEval 2010/2013.
- `semeval.csv` — k-means vs. agglomerative at each model's best layer config, columns
  `model, layers, method, F-S, V-M, AVG10, FBC, FNMI, AVG13`.
- `README.md` (this file, to be replaced) — summary tables, senses-per-word, runtime per
  model, and spot-check observations once the experiments run.

Nothing here yet — the pipeline stages haven't been built.
