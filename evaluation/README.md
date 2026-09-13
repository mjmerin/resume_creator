# Model evaluation set

`cases.json` contains 20 small, human-labeled postings evaluated against the fictional example profile and general variant. Bracketed IDs make requirement recall measurable without relying on fuzzy text matching.

Run two or more repetitions to measure score consistency:

```sh
.venv/bin/python scripts/benchmark_match.py gemma2:9b-instruct-q8_0 --runs 3
```

Compare multiple locally installed models by listing them as positional arguments. The benchmark reports invented-evidence rate, required-requirement recall, JSON success rate, score consistency, mean runtime, and Ollama-reported loaded VRAM. Treat invented evidence as the primary rejection criterion. Review and extend the human labels whenever the résumé schema or representative target roles change.
