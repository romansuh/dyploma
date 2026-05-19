"""Training and evaluation helpers for f_B (binary anomaly detector).

Three sklearn classifiers (GaussianNB, LogisticRegression, LinearSVC) plus a
predict-majority DummyClassifier baseline. All consume L2-normalized BERT
template vectors and accept `sample_weight` (template support) to compensate
class imbalance, per thesis 2.1 and 2.3 (Перевірка 1).

The notebook orchestrates train/test split, weighted fit, metric collection,
and CSV export; this module is the pure-logic layer.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC


def make_classifiers(random_state: int = 42) -> dict[str, BaseEstimator]:
    """Return three deterministic classifiers keyed by short display name.

    - GaussianNB: closed-form fit, no stochastic component.
    - LogisticRegression: L2 / lbfgs (sklearn 1.8 default; `penalty` kwarg is
      deprecated, the default already gives L2). `max_iter` set high enough to
      converge on 77-template train sets without ConvergenceWarning.
    - LinearSVC: liblinear-backed; `dual='auto'` lets sklearn pick primal vs
      dual based on n_samples vs n_features (primal for n_features=768 > n).
    """
    return {
        "GaussianNB": GaussianNB(),
        "LogReg": LogisticRegression(
            solver="lbfgs",
            max_iter=10_000,
            random_state=random_state,
        ),
        "LinearSVC": LinearSVC(
            dual="auto",
            max_iter=10_000,
            random_state=random_state,
        ),
    }


def make_baseline(random_state: int = 42) -> BaseEstimator:
    """Predict-majority-class baseline; serves as the MCC adequacy floor."""
    return DummyClassifier(strategy="most_frequent", random_state=random_state)


def score_for_roc(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Continuous (n_samples,) score used as ROC-AUC input.

    `predict_proba[:, 1]` when supported (GaussianNB, LogReg, Dummy),
    `decision_function` otherwise (LinearSVC). Hard fallback to `predict`
    only if neither exists — should not trigger for the models above.
    """
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X))
    return np.asarray(model.predict(X), dtype=float)


def fit_with_timing(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Fit `model` in-place; return wall-clock fit time in seconds.

    `sample_weight` is forwarded to `.fit()`. Falls back to weight-free fit if
    the estimator's signature rejects the kwarg (defensive — all four target
    estimators do support it in sklearn 1.8).
    """
    t0 = time.perf_counter()
    try:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    except TypeError:
        model.fit(X_train, y_train)
    return time.perf_counter() - t0


def inference_seconds_per_record(
    model: BaseEstimator, X: np.ndarray, repeats: int = 200
) -> float:
    """Mean inference seconds per record, averaged over `repeats` predict passes.

    Returned value is `total_elapsed / (repeats * n_samples)`. Repeating predict
    dampens timer jitter on the tiny (77-row) test set.
    """
    n = X.shape[0]
    t0 = time.perf_counter()
    for _ in range(repeats):
        model.predict(X)
    elapsed = time.perf_counter() - t0
    return elapsed / (repeats * n)


def evaluate(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_test_rule: np.ndarray,
) -> dict[str, float]:
    """Compute MCC, F1-macro, ROC-AUC, and MCC vs rule-based labels.

    Per thesis 2.3 Перевірка 2: `MCC_vs_rulebased` recomputes MCC of the SAME
    predictions against the rule-only labels (no retraining). ROC-AUC is NaN
    when `y_test` is single-class — caller decides how to report.
    """
    y_pred = model.predict(X_test)
    metrics: dict[str, float] = {
        "MCC": float(matthews_corrcoef(y_test, y_pred)),
        "F1_macro": float(
            f1_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "MCC_vs_rulebased": float(matthews_corrcoef(y_test_rule, y_pred)),
    }
    if np.unique(y_test).size == 2:
        scores = score_for_roc(model, X_test)
        metrics["ROC_AUC"] = float(roc_auc_score(y_test, scores))
    else:
        metrics["ROC_AUC"] = float("nan")
    return metrics


if __name__ == "__main__":
    # Smoke test: two well-separated Gaussian blobs in R^4. All three classifiers
    # must achieve near-perfect MCC; the baseline must collapse to 0.
    rng = np.random.default_rng(42)
    X_pos = rng.normal(loc=+1.0, scale=0.3, size=(20, 4))
    X_neg = rng.normal(loc=-1.0, scale=0.3, size=(20, 4))
    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    y = np.array([1] * 20 + [0] * 20, dtype=int)
    w = np.ones(40, dtype=float)

    models = make_classifiers()
    for name, clf in models.items():
        t = fit_with_timing(clf, X, y, sample_weight=w)
        m = evaluate(clf, X, y, y)
        print(
            f"{name}: MCC={m['MCC']:.3f}, F1={m['F1_macro']:.3f}, "
            f"ROC={m['ROC_AUC']:.3f}, fit={t:.4g}s"
        )
        assert m["MCC"] > 0.9, f"{name} failed smoke test"

    baseline = make_baseline()
    fit_with_timing(baseline, X, y, sample_weight=w)
    base_m = evaluate(baseline, X, y, y)
    assert abs(base_m["MCC"]) < 1e-9, f"baseline MCC must be 0, got {base_m['MCC']}"
    print(f"Baseline: MCC={base_m['MCC']:.3f} (expected 0)")
    print("OK")
