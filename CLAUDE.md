# CLAUDE.md

Context for future Claude Code sessions on this repository.

## 1. Project goal

Practical implementation accompanying the bachelor thesis "Розробка та аналіз гібридного методу для семантичної фільтрації неінформативних записів у системних логах" (Odesa I. I. Mechnikov National University, Applied Mathematics, defense 2026). The hybrid pipeline filters non-informative log records in six stages: (1) regex preprocessing of raw logs to insert placeholders for IPs, paths, exception classes, hex values and numbers; (2) the Drain online fixed-depth-tree parser to extract ~200 unique templates; (3) BERT [CLS] vectorization applied to the templates only (not to raw records), producing a 200×768 matrix; (4) KMeans clustering of those vectors into ~44 semantic clusters that serve as automatic labels; (5) supervised training of GaussianNB, Logistic Regression and LinearSVC on (template_vector, cluster_label) pairs; (6) evaluation with MCC as the primary metric and F1-macro plus ROC-AUC as secondary. The dataset is ZooKeeper logs from LogHub (https://github.com/logpai/loghub/tree/master/Zookeeper) — both the small annotated `Zookeeper_2k.log` and the full `Zookeeper.log`.

## 2. My background

I am a full-stack web developer (Next.js, React Native, Fastify, TypeScript), not an ML engineer; readability is the priority over cleverness or performance.

## 3. Code style rules

- Type hints (PEP 484) on every function — same readability target as TypeScript.
- Docstrings in English, PEP 257 format, short and concrete.
- Inline code comments in English, only when something non-obvious happens.
- Markdown cells in Jupyter notebooks in Ukrainian, explaining each step in plain language.
- Classes for stateful components (e.g. `DrainParser`, cluster manager); plain functions for pure transformations.
- Standard idiomatic ML code — no custom abstractions invented on top of scikit-learn / transformers.
- No metaprogramming and no decorators beyond the basics (`@staticmethod`, `@dataclass`, `@property`).
- Prefer explicit over clever; pipeline correctness and clarity matter more than performance optimization.

## 4. Tech stack

Pinned versions live in `pyproject.toml` and `uv.lock`. At scaffold time the resolved versions are:

- Python 3.13.5
- jupyterlab 4.5.7, ipykernel 7.2.0
- drain3 0.9.11
- transformers 5.8.0, torch 2.11.0, tokenizers 0.22.2
- scikit-learn 1.8.0, umap-learn 0.5.12
- matplotlib 3.10.9, seaborn 0.13.2
- pandas 3.0.2, numpy 2.4.4
- requests 2.33.1, tqdm 4.67.3

The `drain3` library is the baseline parser; it is wrapped in our own `DrainParser` class in `src/drain/parser.py` that adds structured logging of every parse decision, an online-mode demonstration helper, and a method to export the current template tree as JSON. BERT uses `bert-base-uncased`; the tokenizer is extended with custom special tokens `<IP>`, `<NUM>`, `<PATH>`, `<HEX>`, `<EXC>`, `<SESSION>`, `<UUID>`. Plotting is matplotlib + seaborn only — no plotly. Notebooks are classic Jupyter (jupyterlab) — no marimo, no Quarto.

## 5. Layout

ROOT/
    CLAUDE.md
    README.md
    pyproject.toml                  (managed by uv)
    uv.lock
    .gitignore
    .python-version
    data/
        raw/                        (downloaded ZooKeeper logs, gitignored)
        processed/
            templates/              (JSON exports from Drain)
            embeddings/             (.npy arrays of BERT vectors)
            clusters/               (JSON cluster assignments)
            results/                (JSON metric results, classifier predictions)
    notebooks/
        01_drain_zookeeper.ipynb
        02_bert_vectorization.ipynb
        03_clustering.ipynb
        04_classification.ipynb
        05_attention_visualization.ipynb
    src/
        __init__.py
        preprocessing/
            __init__.py
            regex_normalizer.py
        drain/
            __init__.py
            parser.py
        bert/
            __init__.py
            vectorizer.py
        clustering/
            __init__.py
            kmeans_pipeline.py
        classification/
            __init__.py
            trainer.py
        metrics/
            __init__.py
            evaluator.py
        io/
            __init__.py
            persistence.py          (JSON and .npy save/load helpers)
    scripts/
        download_zookeeper.py

Each `src/` module is consumed by exactly one notebook. Notebooks orchestrate; `src/` holds reusable logic. No business logic in notebooks beyond glue and visualization.

## 6. How to run

1. `uv sync`
2. `uv run jupyter lab`

Run in this order. `uv sync` installs the locked dependencies into `.venv/`; `uv run jupyter lab` starts JupyterLab using that environment.

## 7. Data flow between notebooks

| Notebook | Reads from data/processed | Writes to data/processed |
|----------|---------------------------|--------------------------|
| 01_drain_zookeeper | (reads `data/raw/Zookeeper.log`, `Zookeeper_2k.log`) | `templates/templates.json` (template tree + `template_id → template_string`), `templates/record_to_template.json` (raw record id → template id) |
| 02_bert_vectorization | `templates/templates.json` | `embeddings/template_vectors.npy` (shape `(N_templates, 768)`), `embeddings/template_index.json` (row index → template id) |
| 03_clustering | `embeddings/template_vectors.npy`, `embeddings/template_index.json` | `clusters/template_to_cluster.json` (template id → cluster label), `clusters/projection_tsne.npy`, `clusters/projection_umap.npy` |
| 04_classification | `embeddings/template_vectors.npy`, `clusters/template_to_cluster.json` | `results/metrics.json` (MCC, F1-macro, ROC-AUC per classifier), `results/predictions_<classifier>.json` |
| 05_attention_visualization | `templates/templates.json` (selected templates only) | (writes plots next to the notebook, no JSON/npy outputs) |

## 8. Inline checks instead of tests

Instead of pytest test files, every notebook must contain `assert` statements and print-based sanity checks at key steps — for example, asserting expected array shapes after BERT vectorization, asserting cluster count after KMeans, printing class distribution before classification. No `tests/` directory and no test files exist yet; do not create them.

## 9. Git

Local repository only for now, no remote configured. The `.gitignore` covers Python build artifacts, virtual environments, Jupyter checkpoints, OS files, IDE metadata, the `data/raw/` directory and the `data/processed/` outputs (regenerable from notebooks).

## 10. Python version fallback

The current environment is Python 3.13.5 and all listed dependencies installed cleanly with prebuilt wheels — no fallback was needed at scaffold time. If a future dependency fails to provide a 3.13 wheel, do not attempt to compile from source. Instead:

1. `pyenv install 3.12` (or the latest 3.12.x).
2. `pyenv local 3.12.x` in the repo root — this updates `.python-version`.
3. Update `requires-python` in `pyproject.toml` to `>=3.12`.
4. Re-run `uv sync`.
5. Document the offending package and its missing wheel here, in this section, so future sessions know why the fallback happened.
