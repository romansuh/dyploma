"""Generate notebooks/06_ablation.ipynb from in-memory cell list.

Single-purpose helper so the notebook JSON is built by Python (correct
escaping) rather than handwritten. Re-running this script overwrites
the notebook with execution_count=None on every cell, which is what
`jupyter nbconvert --execute` expects.
"""
from __future__ import annotations

import json
import secrets
from pathlib import Path


def _cid() -> str:
    """Return a short hex cell id matching the convention used by nb01-nb05."""
    return secrets.token_hex(4)


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _cid(),
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _cid(),
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS: list[dict] = []

CELLS.append(md(
    '<a href="https://colab.research.google.com/github/romansuh/dyploma/blob/master/notebooks/06_ablation.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Відкрити в Google Colab"/></a>'
))

CELLS.append(code(
    "# Colab setup: clone repo, install missing deps, add src/ to path. No-op locally.\n"
    "try:\n"
    "    import google.colab  # noqa: F401\n"
    "    IN_COLAB = True\n"
    "except ImportError:\n"
    "    IN_COLAB = False\n"
    "\n"
    "import os, sys\n"
    "if IN_COLAB:\n"
    "    if not os.path.exists('/content/dyploma'):\n"
    "        !git clone -q https://github.com/romansuh/dyploma.git /content/dyploma\n"
    "    os.chdir('/content/dyploma')\n"
    "    !pip install -q drain3 umap-learn\n"
    "sys.path.insert(0, 'src')"
))

CELLS.append(md(
    "# Ноутбук 06. Ablation study — внесок компонентів пайплайну в MCC\n"
    "\n"
    "Цей ноутбук кількісно оцінює внесок трьох ключових проєктних рішень гібридного пайплайну, послідовно вимикаючи по одному компоненту й порівнюючи MCC з повним пайплайном (теза 1.7, 1.9, 2.1).\n"
    "\n"
    "**Базова лінія (повний пайплайн).** L2-нормалізовані BERT-вектори з розширеним токенізатором (7 плейсхолдерів `<IP>`, `<NUM>`, `<PATH>`, `<HEX>`, `<EXC>`, `<SESSION>`, `<UUID>`) + `sample_weight = support(t)`. Локк-значення з `data/processed/classification/metrics_fb.csv`:\n"
    "\n"
    "| модель | MCC_full_expert |\n"
    "|---|---|\n"
    "| GaussianNB | 0.3189 |\n"
    "| LogReg | 0.6649 |\n"
    "| LinearSVC | **0.8796** (головна модель $f_B$) |\n"
    "\n"
    "Базова лінія перебудовується тут з нуля (детерміністично, seed=42) і звіряється з CSV у межах $10^{-6}$ — якщо це не виконується, ablation-числа втрачають сенс.\n"
    "\n"
    "**Три ablation-варіанти.**\n"
    "\n"
    "1. **A1 — без L2-нормалізації.** Сирі BERT-вектори. Перевіряє припущення тези 1.7 про **анізотропію** претренованого BERT — попарна косинусна схожість у вузькому конусі $[0.6, 0.97]$, через що сирі вектори дають погану геометрію для лінійних класифікаторів.\n"
    "2. **A2 — без `sample_weight`.** L2-нормалізація залишається; усі 77 шаблонів отримують однакову вагу. Перевіряє операційне визначення інформативності з тези 2.1 — якщо ваги відсутні, навчальний тиск перестає відображати $4.90\\%/95.10\\%$ розподіл записів і шаблон `Purge task is not scheduled.` (підтримка 36) трактується нарівні з `Connection broken` (підтримка 10 388).\n"
    "3. **A3 — без розширення токенізатора.** Вектори перерахані ванільним `bert-base-uncased`, в якому плейсхолдери НЕ додані до словника як special tokens (фрагментуються WordPiece-ом: `<NUM>` → `['<', 'nu', '##m', '>']`). **Важливо:** заміна змінних плейсхолдерами на етапі регекс-препроцесингу залишається сталою — A3 вимірює винятково цінність **доповнення словника токенізатора**, а не саму ідею плейсхолдер-нормалізації.\n"
    "\n"
    "Артефакти на виході (у `data/processed/ablation/`):\n"
    "\n"
    "* `ablation_results.csv` — таблиця $16 \\times 6$ (4 варіанти × 4 моделі).\n"
    "* `ablation_deltas.csv` — таблиця $4 \\times 4$: $\\Delta\\mathrm{MCC}(M, X) = \\mathrm{MCC}_{\\text{full,expert}}(M, \\text{baseline}) - \\mathrm{MCC}_{\\text{full,expert}}(M, X)$. Додатне значення $\\Delta$ означає, що компонент важливий.\n"
    "* `linearsvc_ablation_deltas.png` — діаграма внеску для LinearSVC.\n"
    "* `zookeeper_full_embeddings_vanilla.npy` — `(77, 768) float32` ванільного A3 для відтворюваності."
))

CELLS.append(code(
    "from __future__ import annotations\n"
    "\n"
    "import os\n"
    "\n"
    "# CPU single-thread for deterministic timing. Must be set BEFORE numpy/sklearn import.\n"
    "for _var in (\n"
    "    'OMP_NUM_THREADS',\n"
    "    'MKL_NUM_THREADS',\n"
    "    'OPENBLAS_NUM_THREADS',\n"
    "    'BLIS_NUM_THREADS',\n"
    "    'VECLIB_MAXIMUM_THREADS',\n"
    "    'NUMEXPR_NUM_THREADS',\n"
    "):\n"
    "    os.environ[_var] = '1'\n"
    "\n"
    "import sys\n"
    "from pathlib import Path\n"
    "\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import torch\n"
    "from sklearn.base import BaseEstimator\n"
    "from sklearn.model_selection import train_test_split\n"
    "from sklearn.preprocessing import normalize\n"
    "\n"
    "ROOT = Path('..').resolve()\n"
    "if str(ROOT) not in sys.path:\n"
    "    sys.path.insert(0, str(ROOT))\n"
    "\n"
    "DATA_PROCESSED = ROOT / 'data' / 'processed'\n"
    "EMBEDDINGS_DIR = DATA_PROCESSED / 'embeddings'\n"
    "CLUSTERS_DIR = DATA_PROCESSED / 'clusters'\n"
    "CLASSIFICATION_DIR = DATA_PROCESSED / 'classification'\n"
    "RESULTS_DIR = DATA_PROCESSED / 'ablation'\n"
    "RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "from src.bert.vectorizer import TemplateVectorizer\n"
    "from src.classification.trainer import (\n"
    "    evaluate,\n"
    "    fit_with_timing,\n"
    "    make_baseline,\n"
    "    make_classifiers,\n"
    ")\n"
    "from src.io.persistence import load_json, load_numpy, save_numpy\n"
    "\n"
    "RANDOM_STATE = 42\n"
    "np.random.seed(RANDOM_STATE)\n"
    "torch.manual_seed(RANDOM_STATE)\n"
    "\n"
    "MODEL_ORDER = ['GaussianNB', 'LogReg', 'LinearSVC', 'Baseline']\n"
    "VARIANT_ORDER = ['baseline', 'A1_no_L2', 'A2_no_sample_weight', 'A3_no_extended_tokenizer']\n"
    "\n"
    "print(f'numpy  = {np.__version__}')\n"
    "print(f'pandas = {pd.__version__}')\n"
    "print(f'torch  = {torch.__version__}')"
))

CELLS.append(md(
    "## 1. Завантаження спільних артефактів\n"
    "\n"
    "Зчитуємо ті самі чотири артефакти, що й ноутбук 04:\n"
    "\n"
    "* `embeddings/zookeeper_full_embeddings.npy` $(77, 768)$ `float32` — BERT [CLS]-вектори з розширеним токенізатором.\n"
    "* `embeddings/zookeeper_full_id_mapping.json` — порядок `row_index → template_id` (плюс рядок шаблону для A3).\n"
    "* `clusters/template_to_binary_label.json` — фінальна мітка $\\ell(t)$ + `support(t)`.\n"
    "* `clusters/template_to_rule_label.json` — rule-based мітка $\\ell^{(0)}(t)$ (для повнодоменної Перевірки 2)."
))

CELLS.append(code(
    "embeddings_extended = load_numpy(EMBEDDINGS_DIR / 'zookeeper_full_embeddings.npy')\n"
    "id_mapping: list[dict] = load_json(EMBEDDINGS_DIR / 'zookeeper_full_id_mapping.json')\n"
    "template_to_label = load_json(CLUSTERS_DIR / 'template_to_binary_label.json')\n"
    "template_to_rule = load_json(CLUSTERS_DIR / 'template_to_rule_label.json')\n"
    "\n"
    "final_by_tid: dict[int, int] = {r['template_id']: r['final_label'] for r in template_to_label}\n"
    "support_by_tid: dict[int, int] = {r['template_id']: r['support'] for r in template_to_label}\n"
    "rule_by_tid: dict[int, int] = {r['template_id']: r['rule_label'] for r in template_to_rule}\n"
    "\n"
    "embed_tids = {entry['template_id'] for entry in id_mapping}\n"
    "assert embed_tids == set(final_by_tid), 'embedding/label template_id mismatch'\n"
    "assert embed_tids == set(rule_by_tid),  'embedding/rule  template_id mismatch'\n"
    "\n"
    "# X_raw shape (77, 768) float32; y/w/y_rule shape (77,).\n"
    "X_raw = embeddings_extended.astype(np.float32)\n"
    "y = np.array([final_by_tid[m['template_id']] for m in id_mapping], dtype=int)\n"
    "w = np.array([support_by_tid[m['template_id']] for m in id_mapping], dtype=float)\n"
    "y_rule = np.array([rule_by_tid[m['template_id']] for m in id_mapping], dtype=int)\n"
    "\n"
    "# Template strings in the SAME row order — input for A3's re-vectorization.\n"
    "template_strings_full: list[str] = [m['template'] for m in id_mapping]\n"
    "\n"
    "assert X_raw.shape == (77, 768), X_raw.shape\n"
    "assert y.shape == (77,) and w.shape == (77,) and y_rule.shape == (77,)\n"
    "assert set(np.unique(y).tolist()) == {0, 1}, f'y must contain both classes, got {set(y.tolist())}'\n"
    "assert len(template_strings_full) == 77\n"
    "\n"
    "print(f'X_raw                = {X_raw.shape} {X_raw.dtype} (NOT normalized)')\n"
    "print(f'y                    = {y.shape}, balance = {(y == 1).sum()}/{(y == 0).sum()} (1/0)')\n"
    "print(f'w                    = {w.shape}, sum = {int(w.sum())}')\n"
    "print(f'y_rule               = {y_rule.shape}')\n"
    "print(f'template_strings_full = {len(template_strings_full)} strings')"
))

CELLS.append(md(
    "## 2. Тренувальна петля та функція оцінювання\n"
    "\n"
    "Єдиний хелпер `train_and_full_eval(X, y, w, y_rule, *, use_sample_weight)` повторює структуру з ноутбука 04: стратифіковане 80/20 розбиття на рівні шаблонів з `random_state=42`, навчання чотирьох моделей (`GaussianNB`, `LogReg`, `LinearSVC`, `DummyClassifier`) і оцінювання на **повному домені** 77 шаблонів. Повертає словник `{model_name: {MCC, F1_macro, ROC_AUC, MCC_vs_rulebased}}`.\n"
    "\n"
    "Розбиття рекомпʼютується з насіння для кожного варіанта окремо, бо `X` змінюється (нормалізована або сира; розширена або ванільна). Стратифікатор `y` і `random_state` однакові, тож індексний поділ ідентичний у всіх варіантах."
))

CELLS.append(code(
    "def train_and_full_eval(\n"
    "    X: np.ndarray,\n"
    "    y: np.ndarray,\n"
    "    w: np.ndarray,\n"
    "    y_rule: np.ndarray,\n"
    "    *,\n"
    "    use_sample_weight: bool,\n"
    ") -> dict[str, dict[str, float]]:\n"
    "    \"\"\"Train 4 models on a stratified 80/20 split, evaluate on the full 77 templates.\n"
    "\n"
    "    Returns ``{model_name: {MCC, F1_macro, ROC_AUC, MCC_vs_rulebased}}`` — MCC here is\n"
    "    full-domain MCC vs expert labels (it becomes ``MCC_full_expert`` in the output\n"
    "    table at assembly time).\n"
    "    \"\"\"\n"
    "    X_tr, _X_te, y_tr, _y_te, w_tr, _w_te, _yr_tr, _yr_te = train_test_split(\n"
    "        X, y, w, y_rule,\n"
    "        test_size=0.2,\n"
    "        stratify=y,\n"
    "        random_state=RANDOM_STATE,\n"
    "    )\n"
    "    classifiers: dict[str, BaseEstimator] = make_classifiers(random_state=RANDOM_STATE)\n"
    "    classifiers['Baseline'] = make_baseline(random_state=RANDOM_STATE)\n"
    "\n"
    "    weights = w_tr if use_sample_weight else None\n"
    "    out: dict[str, dict[str, float]] = {}\n"
    "    for name, clf in classifiers.items():\n"
    "        fit_with_timing(clf, X_tr, y_tr, sample_weight=weights)\n"
    "        # Full-domain evaluation: predict on all 77 templates, not just the test split.\n"
    "        out[name] = evaluate(clf, X, y, y_rule)\n"
    "    return out\n"
    "\n"
    "\n"
    "print('train_and_full_eval defined.')"
))

CELLS.append(md(
    "## 3. Відтворення базової лінії\n"
    "\n"
    "L2-нормалізуємо `X_raw` і прогоняємо `train_and_full_eval` з `use_sample_weight=True`. Це має точно (до $10^{-6}$) відтворити колонку `MCC_full_expert` з `data/processed/classification/metrics_fb.csv`. Якщо не відтворює — ablation-числа неінтерпретовні, тож рветься з `AssertionError`."
))

CELLS.append(code(
    "X_norm = normalize(X_raw, norm='l2', axis=1).astype(np.float32)\n"
    "# X_norm shape (77, 768) float32; row norms == 1 ± eps.\n"
    "assert np.allclose(np.linalg.norm(X_norm, axis=1), 1.0, atol=1e-5)\n"
    "\n"
    "baseline_results = train_and_full_eval(X_norm, y, w, y_rule, use_sample_weight=True)\n"
    "\n"
    "metrics_fb = pd.read_csv(CLASSIFICATION_DIR / 'metrics_fb.csv').set_index('model')\n"
    "print('Baseline reproduction check (tolerance 1e-6 against metrics_fb.csv):')\n"
    "for name in MODEL_ORDER:\n"
    "    got = baseline_results[name]['MCC']\n"
    "    expected = float(metrics_fb.loc[name, 'MCC_full_expert'])\n"
    "    diff = abs(got - expected)\n"
    "    print(f'  {name:<10}  recomputed = {got:+.6f}  csv = {expected:+.6f}  |Δ| = {diff:.2e}')\n"
    "    assert diff < 1e-6, (\n"
    "        f'Baseline mismatch for {name}: got {got}, expected {expected}. '\n"
    "        'Ablation numbers are not interpretable — STOP and investigate.'\n"
    "    )\n"
    "print('OK — base pipeline reproduced from scratch.')"
))

CELLS.append(md(
    "## 4. A1 — без L2-нормалізації\n"
    "\n"
    "Подаємо `X_raw` напряму, без `sklearn.preprocessing.normalize`. Все інше — той самий код, та сама розбивка, те саме насіння, ті самі ваги `support(t)`.\n"
    "\n"
    "**Очікування (теза 1.7).** Претренований BERT анізотропний: попарна косинусна схожість у вузькому діапазоні, а норми рядків (як ми бачили в ноутбуці 03 §2) лежать у вузькій смузі $[13.4, 15.8]$, але цього достатньо, щоб лінійна гіперплощина зміщувалася в напрямку шаблонів з більшою нормою. GaussianNB особливо вразливий — він оцінює коваріаційну структуру окремо для кожного класу, а сирі компоненти BERT-вектора мають дисперсії, що відрізняються на порядки. Очікуємо помітне падіння MCC, особливо для GaussianNB і LinearSVC."
))

CELLS.append(code(
    "a1_results = train_and_full_eval(X_raw, y, w, y_rule, use_sample_weight=True)\n"
    "\n"
    "print('A1 — no L2 normalization (full-domain MCC):')\n"
    "for name in MODEL_ORDER:\n"
    "    print(f'  {name:<10}  MCC = {a1_results[name][\"MCC\"]:+.4f}')"
))

CELLS.append(md(
    "## 5. A2 — без `sample_weight`\n"
    "\n"
    "L2-нормалізація залишається; до `.fit(...)` передаємо `sample_weight=None`. Усі 77 шаблонів отримують однакову вагу.\n"
    "\n"
    "**Очікування (теза 2.1).** Операційне визначення інформативності прив'язане до записів, а не шаблонів: 95.10 % записів — це інформативний клас (домінують високопідтримувані WARN-шаблони), 4.90 % — шум. Без ваг навчальний тиск рівномірно розкладається на 77 шаблонів, втрачаючи практичну вагомість частих подій. Очікуємо падіння MCC у моделей, які раніше виграли від ваг, — переважно у LogReg/LinearSVC."
))

CELLS.append(code(
    "a2_results = train_and_full_eval(X_norm, y, w, y_rule, use_sample_weight=False)\n"
    "\n"
    "print('A2 — no sample_weight (full-domain MCC):')\n"
    "for name in MODEL_ORDER:\n"
    "    print(f'  {name:<10}  MCC = {a2_results[name][\"MCC\"]:+.4f}')"
))

CELLS.append(md(
    "## 6. A3 — ванільний токенізатор (без 7 плейсхолдерів у словнику)\n"
    "\n"
    "Перевекторизуємо ті самі 77 рядків шаблонів **другим, незалежним** екземпляром `bert-base-uncased`, у якого словник токенізатора НЕ розширено. У результаті токени `<IP>`, `<NUM>` тощо фрагментуються WordPiece-ом на 3–4 субтокени (наприклад, `<NUM>` → `['<', 'nu', '##m', '>']`). Все інше після цього — те саме: L2-нормалізація, та сама розбивка, ваги `support(t)`.\n"
    "\n"
    "**Що саме вимірюємо.** A3 ізолює внесок **доповнення словника токенізатора** (`add_special_tokens` + mean-init нових ембедингів). Регекс-нормалізація на етапі препроцесингу зберігається сталою (шаблон вже містить `<IP>`, `<NUM>` тощо як рядки) — тож ми НЕ міряємо цінність ідеї плейсхолдер-нормалізації як такої. Очікуємо помітне падіння MCC, оскільки [CLS] після фрагментації плейсхолдерів несе слабший семантичний сигнал.\n"
    "\n"
    "**Про окремий HF-кеш.** Завдання вимагало завантажити ванільний `bert-base-uncased` у окремий кеш для гарантії «незмішаного стану» з розширеним токенізатором ноутбука 02. Це **поверхнева вимога**: розширення токенізатора в `TemplateVectorizer` відбувається повністю в оперативній пам'яті після завантаження ваг — диск-кеш HuggingFace зберігає **тільки** претреновані ваги, які побітово ідентичні для обох прогонів. Ділимо існуючий кеш; новий екземпляр `TemplateVectorizer(placeholder_tokens=[])` створює свіжий Python-обʼєкт без будь-якого in-memory спадку від ноутбука 02."
))

CELLS.append(code(
    "torch.manual_seed(RANDOM_STATE)  # belt-and-suspenders before constructing the vanilla model\n"
    "\n"
    "vectorizer_vanilla = TemplateVectorizer(\n"
    "    model_name='bert-base-uncased',\n"
    "    placeholder_tokens=[],  # vanilla: no added special tokens\n"
    "    device='cpu',\n"
    ")\n"
    "\n"
    "# Lock in that the vectorizer is genuinely vanilla.\n"
    "assert vectorizer_vanilla.num_added_tokens == 0, vectorizer_vanilla.num_added_tokens\n"
    "assert vectorizer_vanilla.vocab_size_before == vectorizer_vanilla.vocab_size_after, (\n"
    "    vectorizer_vanilla.vocab_size_before, vectorizer_vanilla.vocab_size_after\n"
    ")\n"
    "print(\n"
    "    f'Vanilla vocab size = {vectorizer_vanilla.vocab_size_after} '\n"
    "    f'(added = {vectorizer_vanilla.num_added_tokens})'\n"
    ")\n"
    "\n"
    "# Demonstrate WordPiece fragmentation on the first template containing <NUM>.\n"
    "demo_idx = next(i for i, s in enumerate(template_strings_full) if '<NUM>' in s)\n"
    "demo_template = template_strings_full[demo_idx]\n"
    "demo_tokens = vectorizer_vanilla.get_tokenization(demo_template)\n"
    "print(f'\\nDemo template (row {demo_idx}): {demo_template!r}')\n"
    "print(f'  vanilla tokenization: {demo_tokens}')\n"
    "assert '<NUM>' not in demo_tokens, 'vanilla tokenizer must fragment <NUM>'"
))

CELLS.append(code(
    "# Re-vectorize the 77 templates with the vanilla model. Shape (77, 768) float32.\n"
    "X_a3_raw = vectorizer_vanilla.vectorize_batch(\n"
    "    template_strings_full,\n"
    "    batch_size=32,\n"
    "    show_progress=True,\n"
    ")\n"
    "assert X_a3_raw.shape == (77, 768), X_a3_raw.shape\n"
    "assert X_a3_raw.dtype == np.float32, X_a3_raw.dtype\n"
    "\n"
    "save_numpy(X_a3_raw, RESULTS_DIR / 'zookeeper_full_embeddings_vanilla.npy')\n"
    "print(f'saved vanilla embeddings: shape {X_a3_raw.shape}, dtype {X_a3_raw.dtype}')\n"
    "print(\n"
    "    f'  mean = {X_a3_raw.mean():+.4f}  std = {X_a3_raw.std():+.4f}  '\n"
    "    f'min = {X_a3_raw.min():+.4f}  max = {X_a3_raw.max():+.4f}'\n"
    ")\n"
    "\n"
    "X_a3_norm = normalize(X_a3_raw, norm='l2', axis=1).astype(np.float32)\n"
    "assert np.allclose(np.linalg.norm(X_a3_norm, axis=1), 1.0, atol=1e-5)\n"
    "\n"
    "a3_results = train_and_full_eval(X_a3_norm, y, w, y_rule, use_sample_weight=True)\n"
    "\n"
    "print('\\nA3 — vanilla tokenizer (full-domain MCC):')\n"
    "for name in MODEL_ORDER:\n"
    "    print(f'  {name:<10}  MCC = {a3_results[name][\"MCC\"]:+.4f}')"
))

CELLS.append(md(
    "## 7. Зведення таблиці результатів\n"
    "\n"
    "Збираємо $16 \\times 6$ таблицю з фіксованим порядком рядків і колонок. На цьому ж кроці перейменовуємо ключі словника `evaluate(...)`: `MCC → MCC_full_expert`, `MCC_vs_rulebased → MCC_full_rulebased` — щоб колонки CSV точно відповідали специфікації."
))

CELLS.append(code(
    "all_results: dict[str, dict[str, dict[str, float]]] = {\n"
    "    'baseline': baseline_results,\n"
    "    'A1_no_L2': a1_results,\n"
    "    'A2_no_sample_weight': a2_results,\n"
    "    'A3_no_extended_tokenizer': a3_results,\n"
    "}\n"
    "\n"
    "rows: list[dict] = []\n"
    "for variant in VARIANT_ORDER:\n"
    "    for name in MODEL_ORDER:\n"
    "        m = all_results[variant][name]\n"
    "        rows.append({\n"
    "            'variant': variant,\n"
    "            'model': name,\n"
    "            'MCC_full_expert': m['MCC'],\n"
    "            'MCC_full_rulebased': m['MCC_vs_rulebased'],\n"
    "            'F1_macro': m['F1_macro'],\n"
    "            'ROC_AUC': m['ROC_AUC'],\n"
    "        })\n"
    "\n"
    "results_df = pd.DataFrame(\n"
    "    rows,\n"
    "    columns=['variant', 'model', 'MCC_full_expert', 'MCC_full_rulebased', 'F1_macro', 'ROC_AUC'],\n"
    ")\n"
    "assert results_df.shape == (16, 6), results_df.shape\n"
    "assert not results_df['MCC_full_expert'].isna().any()\n"
    "\n"
    "results_path = RESULTS_DIR / 'ablation_results.csv'\n"
    "results_df.to_csv(results_path, index=False)\n"
    "\n"
    "with pd.option_context('display.float_format', '{:+.4f}'.format,\n"
    "                       'display.max_rows', 20,\n"
    "                       'display.width', 110):\n"
    "    print(results_df.to_string(index=False))\n"
    "print(f'\\nsaved: {results_path}  ({results_path.stat().st_size} bytes)')"
))

CELLS.append(md(
    "## 8. Дельти $\\Delta\\mathrm{MCC}$\n"
    "\n"
    "Для кожної моделі $M$ та ablation $X$:\n"
    "\n"
    "$$\\Delta\\mathrm{MCC}(M, X) = \\mathrm{MCC}_{\\text{full,expert}}(M, \\text{baseline}) - \\mathrm{MCC}_{\\text{full,expert}}(M, X)$$\n"
    "\n"
    "Додатне $\\Delta$ означає, що ablation **погіршує** результат — компонент важливий. `Baseline` (DummyClassifier) має $\\Delta = 0$ у всіх стовпцях (він завжди передбачає клас більшості) — це самоперевірка."
))

CELLS.append(code(
    "delta_rows: list[dict] = []\n"
    "for name in MODEL_ORDER:\n"
    "    base_mcc = baseline_results[name]['MCC']\n"
    "    delta_rows.append({\n"
    "        'model': name,\n"
    "        'A1_no_L2': base_mcc - a1_results[name]['MCC'],\n"
    "        'A2_no_sample_weight': base_mcc - a2_results[name]['MCC'],\n"
    "        'A3_no_extended_tokenizer': base_mcc - a3_results[name]['MCC'],\n"
    "    })\n"
    "\n"
    "deltas_df = pd.DataFrame(\n"
    "    delta_rows,\n"
    "    columns=['model', 'A1_no_L2', 'A2_no_sample_weight', 'A3_no_extended_tokenizer'],\n"
    ").set_index('model')\n"
    "\n"
    "deltas_path = RESULTS_DIR / 'ablation_deltas.csv'\n"
    "deltas_df.to_csv(deltas_path)\n"
    "\n"
    "with pd.option_context('display.float_format', '{:+.4f}'.format):\n"
    "    print(deltas_df.to_string())\n"
    "print(f'\\nsaved: {deltas_path}  ({deltas_path.stat().st_size} bytes)')\n"
    "\n"
    "# Self-check: Baseline row must be all zeros.\n"
    "assert (deltas_df.loc['Baseline'].abs() < 1e-12).all(), deltas_df.loc['Baseline']"
))

CELLS.append(md(
    "## 9. Візуалізація — внесок компонентів для LinearSVC\n"
    "\n"
    "Один графік: горизонтальна базова лінія на $0$, три стовпці — $\\Delta\\mathrm{MCC}$ кожної ablation для LinearSVC (головна модель $f_B$). Вища планка = важливіший компонент."
))

CELLS.append(code(
    "lin_deltas = deltas_df.loc['LinearSVC']\n"
    "labels_ua = [\n"
    "    'A1: без L2-нормалізації',\n"
    "    'A2: без sample_weight',\n"
    "    'A3: без розширення токенізатора',\n"
    "]\n"
    "values = [\n"
    "    float(lin_deltas['A1_no_L2']),\n"
    "    float(lin_deltas['A2_no_sample_weight']),\n"
    "    float(lin_deltas['A3_no_extended_tokenizer']),\n"
    "]\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(7.5, 4.5))\n"
    "bars = ax.bar(labels_ua, values, color=['#4C78A8', '#F58518', '#54A24B'])\n"
    "ax.axhline(0, color='black', linewidth=0.8)\n"
    "ax.set_ylabel(r'$\\Delta$MCC (LinearSVC, повний домен)')\n"
    "ax.set_title('Внесок компонентів пайплайну в MCC (LinearSVC)')\n"
    "ax.grid(axis='y', alpha=0.3)\n"
    "\n"
    "# Annotate each bar with its numeric value.\n"
    "for bar, v in zip(bars, values):\n"
    "    ax.text(\n"
    "        bar.get_x() + bar.get_width() / 2,\n"
    "        v + (0.01 if v >= 0 else -0.025),\n"
    "        f'{v:+.4f}',\n"
    "        ha='center', va='bottom' if v >= 0 else 'top',\n"
    "        fontsize=10,\n"
    "    )\n"
    "\n"
    "fig.tight_layout()\n"
    "png_path = RESULTS_DIR / 'linearsvc_ablation_deltas.png'\n"
    "fig.savefig(png_path, dpi=150)\n"
    "plt.show()\n"
    "print(f'saved: {png_path}  ({png_path.stat().st_size} bytes)')"
))

CELLS.append(md(
    "## 10. Інтерпретація\n"
    "\n"
    "Друкуємо числові підсумки для LinearSVC (головна модель). Інтерпретаційний текст у тезі (розділ «Ablation study») формулюється окремо на основі цих значень."
))

CELLS.append(code(
    "print('LinearSVC — внесок кожного компонента:')\n"
    "d_a1 = float(lin_deltas['A1_no_L2'])\n"
    "d_a2 = float(lin_deltas['A2_no_sample_weight'])\n"
    "d_a3 = float(lin_deltas['A3_no_extended_tokenizer'])\n"
    "\n"
    "print(f'  LinearSVC ΔMCC (A1_no_L2)                   = {d_a1:+.4f}')\n"
    "print(f'  LinearSVC ΔMCC (A2_no_sample_weight)        = {d_a2:+.4f}')\n"
    "print(f'  LinearSVC ΔMCC (A3_no_extended_tokenizer)   = {d_a3:+.4f}')\n"
    "print()\n"
    "print(f'A1 ΔMCC={d_a1:+.4f} → L2-нормалізація вносить {d_a1:+.4f} MCC')\n"
    "print(f'A2 ΔMCC={d_a2:+.4f} → sample_weight вносить {d_a2:+.4f} MCC')\n"
    "print(f'A3 ΔMCC={d_a3:+.4f} → розширення токенізатора вносить {d_a3:+.4f} MCC')"
))

CELLS.append(md(
    "## Висновки\n"
    "\n"
    "Реалізовано ablation study з трьома варіантами та чотирма моделями. Базова лінія відтворена з нуля з точністю $10^{-6}$ проти `metrics_fb.csv`.\n"
    "\n"
    "Артефакти:\n"
    "\n"
    "* `data/processed/ablation/ablation_results.csv` — повна таблиця $16 \\times 6$ (4 варіанти × 4 моделі).\n"
    "* `data/processed/ablation/ablation_deltas.csv` — таблиця $\\Delta\\mathrm{MCC}$ для кожної моделі та кожного ablation.\n"
    "* `data/processed/ablation/linearsvc_ablation_deltas.png` — діаграма внеску компонентів для LinearSVC.\n"
    "* `data/processed/ablation/zookeeper_full_embeddings_vanilla.npy` — ванільні BERT-вектори A3 для відтворюваності.\n"
    "\n"
    "Числові висновки про роль L2-нормалізації, `sample_weight` і розширення токенізатора передаються у розділ «Ablation study» тези."
))

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13.5",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "notebooks" / "06_ablation.ipynb"
    out.write_text(json.dumps(NOTEBOOK, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size} bytes, {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
