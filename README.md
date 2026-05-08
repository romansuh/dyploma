# dyploma

Practical implementation for the BSc thesis "Розробка та аналіз гібридного методу для семантичної фільтрації неінформативних записів у системних логах" (Odesa I. I. Mechnikov National University, Applied Mathematics, 2026).

The pipeline: regex preprocessing → Drain template parsing → BERT [CLS] vectorization → KMeans clustering → supervised classification (GaussianNB / Logistic Regression / LinearSVC) → evaluation with MCC, F1-macro, ROC-AUC. Dataset: ZooKeeper logs from [LogHub](https://github.com/logpai/loghub/tree/master/Zookeeper).

## Setup

```sh
uv sync
uv run jupyter lab
```

Then open the notebooks under `notebooks/` and run them in order, `01_*` through `05_*`. Each writes its outputs to `data/processed/` for the next one to read; the dataflow table is in [CLAUDE.md](CLAUDE.md#7-data-flow-between-notebooks).

## Layout

See [CLAUDE.md](CLAUDE.md) for the full project context, layout, code style rules and Python version fallback procedure.
