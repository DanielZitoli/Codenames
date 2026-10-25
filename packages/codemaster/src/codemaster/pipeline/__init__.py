"""OFFLINE pipeline: vocab, harvest, embed, normalize, cluster, glove, report, semeval, export.

Heavy dependencies (torch, transformers, datasets, ...) are allowed only here, behind the
`pipeline` extra. Nothing in `engine/`, `agents/` or `eval/` may import from this package —
see the dependency-boundary test in tests/.

Stages are added one at a time (see CLAUDE.md "Working agreement"); none exist yet.
"""
