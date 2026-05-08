"""JSON save/load helpers used by every notebook in the pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
