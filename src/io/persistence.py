"""JSON and numpy save/load helpers used by every notebook in the pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_json(obj: Any, path: str | Path) -> None:
    """Write `obj` as pretty-printed UTF-8 JSON to `path`, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    """Read JSON from `path` and return the decoded Python object."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_numpy(array: np.ndarray, path: str | Path) -> None:
    """Save `array` to `path` as a .npy file, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def load_numpy(path: str | Path) -> np.ndarray:
    """Read a numpy array from a .npy file at `path`."""
    return np.load(Path(path))


def save_template_to_rule_label(
    template_stats: list[dict[str, Any]],
    path: str | Path,
) -> None:
    """Write the rule-based (pre-correction) label for every template to `path`.

    Output is a JSON list of objects, sorted by `template_id` ascending, each
    containing exactly two fields: `template_id` (int) and `rule_label` (int,
    0 or 1). Reflects ell^(0) from formula `eq:l_initial` only — expert
    corrections from notebook 03 §9 are NOT applied. Notebook 04 reads this
    file to compute the rule-vs-final cross-check MCC (thesis Перевірка 2).
    """
    rows = [
        {
            "template_id": int(r["template_id"]),
            "rule_label": int(r["rule_label"]),
        }
        for r in sorted(template_stats, key=lambda r: r["template_id"])
    ]
    save_json(rows, path)
