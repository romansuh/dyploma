"""Two-stage cluster labeling: rule-based first pass + expert corrections.

Implements formulas eq:level_pi and eq:l_initial from thesis section 2.3.
Stage 1 derives an initial binary label per cluster from the share of
records whose Level field is in a "positive" set (default {WARN, ERROR}).
Stage 2 applies a manually-curated dict of overrides on top.
"""

from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class ClusterLabeler:
    """Compute per-cluster pi(c), thresholded labels, and apply overrides."""

    def __init__(
        self,
        positive_levels: set[str] | None = None,
        threshold: float = 0.5,
    ) -> None:
        # None default + body assignment avoids the mutable-default-argument trap.
        self.positive_levels: set[str] = (
            set(positive_levels) if positive_levels is not None else {"WARN", "ERROR"}
        )
        self.threshold: float = threshold

    def compute_pi(
        self,
        record_levels: list[str | None],
        record_cluster_ids: list[int],
    ) -> dict[int, float]:
        """For each cluster, return positive-level fraction (formula eq:level_pi)."""
        assert len(record_levels) == len(record_cluster_ids), (
            f"length mismatch: levels={len(record_levels)} "
            f"cluster_ids={len(record_cluster_ids)}"
        )
        total: dict[int, int] = defaultdict(int)
        positive: dict[int, int] = defaultdict(int)
        for level, cid in zip(record_levels, record_cluster_ids):
            total[cid] += 1
            if level in self.positive_levels:
                positive[cid] += 1
        return {cid: positive[cid] / total[cid] for cid in total}

    def initial_labels(self, pi: dict[int, float]) -> dict[int, int]:
        """Threshold pi(c) at `self.threshold` (formula eq:l_initial)."""
        return {cid: (1 if pi_c >= self.threshold else 0) for cid, pi_c in pi.items()}

    def apply_corrections(
        self,
        initial: dict[int, int],
        corrections: dict[int, int],
    ) -> dict[int, int]:
        """Overwrite `initial[c]` with `corrections[c]` where present; log each change."""
        final = dict(initial)
        for cid, new_label in corrections.items():
            if cid not in final:
                logger.warning(
                    "correction skipped: cluster=%d not in initial labels", cid
                )
                continue
            old_label = final[cid]
            if new_label == old_label:
                logger.info(
                    "correction no-op: cluster=%d already labeled %d", cid, old_label
                )
                continue
            logger.info(
                "correction: cluster=%d %d -> %d", cid, old_label, new_label
            )
            final[cid] = new_label
        return final


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    # Toy data: 3 clusters of 5 records each.
    # cluster 0: 1 WARN, 4 INFO   -> pi = 0.2
    # cluster 1: 3 ERROR, 2 INFO  -> pi = 0.6
    # cluster 2: 0 WARN/ERROR, 5 None -> pi = 0.0
    record_levels: list[str | None] = [
        "WARN", "INFO", "INFO", "INFO", "INFO",
        "ERROR", "ERROR", "ERROR", "INFO", "INFO",
        None, None, None, None, None,
    ]
    record_cluster_ids: list[int] = [
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        2, 2, 2, 2, 2,
    ]

    labeler = ClusterLabeler()
    pi = labeler.compute_pi(record_levels, record_cluster_ids)
    assert pi == {0: 0.2, 1: 0.6, 2: 0.0}, pi

    initial = labeler.initial_labels(pi)
    assert initial == {0: 0, 1: 1, 2: 0}, initial

    final = labeler.apply_corrections(initial, {0: 1})
    assert final == {0: 1, 1: 1, 2: 0}, final

    # A no-op correction should not flip anything.
    same = labeler.apply_corrections(initial, {2: 0})
    assert same == initial, same

    # An unknown cluster id should be skipped with a warning.
    skipped = labeler.apply_corrections(initial, {99: 1})
    assert skipped == initial, skipped

    print("OK")
