"""Template-level rule-based labeling + expert corrections.

Implements thesis eq:level_pi and eq:l_initial at the TEMPLATE level
(amended architecture: cluster-level labeling was demoted to diagnostic).
For each template t we compute pi(t) = (#records with Level in
{WARN, ERROR}) / support(t), threshold at tau (default 0.5) to derive
an initial rule_label, then overlay manually-curated expert overrides
keyed by template_id.
"""

from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def compute_template_stats(
    record_template_ids: list[int],
    record_levels: list[str | None],
    id_mapping: list[dict],
    positive_levels: set[str] | None = None,
    threshold: float = 0.5,
) -> list[dict]:
    """Return per-template stats for every entry of `id_mapping`.

    Each output row has keys:
        template_id (int), template (str), support (int),
        level_counts (dict[str, int]; None -> "NONE"),
        pi (float), rule_label (int 0 or 1).

    Rows are returned in `id_mapping` iteration order (typically support-desc).
    `record_template_ids` and `record_levels` must be the same length.
    """
    # None default avoids the mutable-default-argument trap.
    pos: set[str] = (
        set(positive_levels) if positive_levels is not None else {"WARN", "ERROR"}
    )
    assert len(record_template_ids) == len(record_levels), (
        f"length mismatch: template_ids={len(record_template_ids)} "
        f"levels={len(record_levels)}"
    )

    # per_template_levels[tid] = list of Level strings (or None) seen for that template
    per_template_levels: dict[int, list[str | None]] = defaultdict(list)
    for tid, lvl in zip(record_template_ids, record_levels):
        per_template_levels[tid].append(lvl)

    rows: list[dict] = []
    for entry in id_mapping:
        tid: int = entry["template_id"]
        levels: list[str | None] = per_template_levels.get(tid, [])
        support: int = len(levels)

        # JSON-serializable level_counts (None becomes the string "NONE").
        level_counts: dict[str, int] = {}
        for lvl in levels:
            key = lvl if lvl is not None else "NONE"
            level_counts[key] = level_counts.get(key, 0) + 1

        positive_count: int = sum(1 for lvl in levels if lvl in pos)
        pi: float = (positive_count / support) if support > 0 else 0.0
        rule_label: int = 1 if pi >= threshold else 0

        rows.append(
            {
                "template_id": tid,
                "template": entry["template"],
                "support": support,
                "level_counts": level_counts,
                "pi": pi,
                "rule_label": rule_label,
            }
        )
    return rows


def apply_template_corrections(
    stats: list[dict],
    corrections: dict[int, int],
    reasons: dict[int, str] | None = None,
) -> list[dict]:
    """Overlay expert overrides onto template-level rule labels.

    For each row in `stats`, add three keys:
        final_label (int) = corrections.get(template_id, rule_label)
        corrected  (bool) = template_id in corrections
        reason     (str)  = reasons.get(template_id, "") if corrected else ""

    Corrections referencing unknown template_ids are skipped with a WARNING.
    No-op corrections (override matches the rule_label) are logged at INFO
    but still mark the row as corrected (the expert explicitly confirmed it).
    Pure function — `stats` is not mutated.
    """
    reasons = {} if reasons is None else reasons
    known_tids: set[int] = {row["template_id"] for row in stats}
    for tid in corrections:
        if tid not in known_tids:
            logger.warning(
                "correction skipped: template_id=%d not in stats", tid
            )

    out: list[dict] = []
    for row in stats:
        tid: int = row["template_id"]
        new_row = dict(row)
        if tid in corrections:
            new_label = corrections[tid]
            new_row["final_label"] = new_label
            new_row["corrected"] = True
            new_row["reason"] = reasons.get(tid, "")
            if new_label == row["rule_label"]:
                logger.info(
                    "correction no-op: template_id=%d already labeled %d",
                    tid,
                    new_label,
                )
            else:
                logger.info(
                    "correction: template_id=%d %d -> %d",
                    tid,
                    row["rule_label"],
                    new_label,
                )
        else:
            new_row["final_label"] = row["rule_label"]
            new_row["corrected"] = False
            new_row["reason"] = ""
        out.append(new_row)
    return out


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s — %(message)s"
    )

    # Toy fixture: 3 templates, 9 records.
    # tid=10: 1 WARN, 2 INFO              -> pi = 1/3 ~ 0.333 -> rule_label 0
    # tid=20: 2 ERROR, 1 WARN             -> pi = 3/3 = 1.000 -> rule_label 1
    # tid=30: 2 INFO, 1 None              -> pi = 0/3 = 0.000 -> rule_label 0
    id_mapping = [
        {"template_id": 10, "template": "alpha"},
        {"template_id": 20, "template": "beta"},
        {"template_id": 30, "template": "gamma"},
    ]
    record_template_ids = [10, 10, 10, 20, 20, 20, 30, 30, 30]
    record_levels: list[str | None] = [
        "WARN", "INFO", "INFO",
        "ERROR", "ERROR", "WARN",
        "INFO", "INFO", None,
    ]

    stats = compute_template_stats(record_template_ids, record_levels, id_mapping)
    assert len(stats) == 3
    tid_to_row = {r["template_id"]: r for r in stats}
    assert tid_to_row[10]["support"] == 3
    assert abs(tid_to_row[10]["pi"] - 1 / 3) < 1e-9
    assert tid_to_row[10]["rule_label"] == 0
    assert tid_to_row[20]["rule_label"] == 1
    assert tid_to_row[30]["rule_label"] == 0
    assert tid_to_row[30]["level_counts"] == {"INFO": 2, "NONE": 1}

    # Expert flips template 10 to informative ("noisy but actionable"),
    # confirms template 20 at 1 (no-op), leaves template 30 alone.
    corrections = {10: 1, 20: 1}
    reasons = {10: "rare but actionable noise", 20: "confirmed informative"}
    labeled = apply_template_corrections(stats, corrections, reasons)
    by_tid = {r["template_id"]: r for r in labeled}
    assert by_tid[10]["final_label"] == 1 and by_tid[10]["corrected"] is True
    assert by_tid[10]["reason"] == "rare but actionable noise"
    assert by_tid[20]["final_label"] == 1 and by_tid[20]["corrected"] is True
    assert by_tid[30]["final_label"] == 0 and by_tid[30]["corrected"] is False
    assert by_tid[30]["reason"] == ""

    # Unknown template_id is skipped with a WARNING.
    skipped = apply_template_corrections(stats, {999: 1})
    assert all(r["corrected"] is False for r in skipped)

    print("OK")
