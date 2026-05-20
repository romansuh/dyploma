# dyploma

Practical implementation for the BSc thesis "Розробка та аналіз гібридного методу для семантичної фільтрації неінформативних записів у системних логах" (Odesa I. I. Mechnikov National University, Applied Mathematics, 2026).

The pipeline: regex preprocessing → Drain template parsing → BERT [CLS] vectorization → KMeans clustering → supervised classification (GaussianNB / Logistic Regression / LinearSVC) → evaluation with MCC, F1-macro, ROC-AUC. Dataset: ZooKeeper logs from [LogHub](https://github.com/logpai/loghub/tree/master/Zookeeper).

## Run in Google Colab

Each notebook has an "Open in Colab" badge at the top. Click it, then run the first code cell — it clones the repo, installs the two missing dependencies (`drain3`, `umap-learn`), and adds `src/` to `sys.path`. The annotated sample log (`data/raw/Zookeeper_2k.log`) is committed, so notebooks 01–05 run end-to-end on the free Colab runtime. The full `Zookeeper.log` is not committed — download it from [LogHub](https://github.com/logpai/loghub/tree/master/Zookeeper) if you need the full-domain experiments.

| # | Notebook |
|---|---|
| 01 | [Drain template parsing](https://colab.research.google.com/github/romansuh/dyploma/blob/master/notebooks/01_drain_zookeeper.ipynb) |
| 02 | [BERT vectorization](https://colab.research.google.com/github/romansuh/dyploma/blob/master/notebooks/02_bert_vectorization.ipynb) |
| 03 | [Template labeling + clustering](https://colab.research.google.com/github/romansuh/dyploma/blob/master/notebooks/03_clustering.ipynb) |
| 04 | [Binary classification](https://colab.research.google.com/github/romansuh/dyploma/blob/master/notebooks/04_classification.ipynb) |
| 05 | [Attention diagnostics](https://colab.research.google.com/github/romansuh/dyploma/blob/master/notebooks/05_attention.ipynb) |

For the BERT step, enable a GPU runtime: *Runtime → Change runtime type → T4 GPU*.

## Run locally

With [uv](https://docs.astral.sh/uv/) (recommended — uses the pinned `uv.lock`):

```sh
uv sync
uv run jupyter lab
```

With plain pip (uses the frozen `requirements.txt`):

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Then open the notebooks under `notebooks/` and run them in order, `01_*` through `05_*`. Each writes its outputs to `data/processed/` for the next one to read; the dataflow table is in [CLAUDE.md](CLAUDE.md#7-data-flow-between-notebooks).

## Layout

See [CLAUDE.md](CLAUDE.md) for the full project context, layout, code style rules and Python version fallback procedure.
