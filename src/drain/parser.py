"""Wrapper around `drain3.TemplateMiner` for the thesis pipeline.

Adds:
  - structured per-line logging of every parse decision,
  - a typed `ParseResult` return so downstream notebooks have stable field names,
  - convenience methods for batch processing, template export, and reset.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

from drain3 import TemplateMiner
from drain3.memory_buffer_persistence import MemoryBufferPersistence
from drain3.template_miner_config import TemplateMinerConfig
from tqdm.auto import tqdm

from src.io.persistence import save_json

logger = logging.getLogger(__name__)

ChangeType = Literal["new_template", "matched_existing", "template_changed"]

_DRAIN_TO_SPEC: dict[str, ChangeType] = {
    "cluster_created": "new_template",
    "cluster_template_changed": "template_changed",
    "none": "matched_existing",
}


@dataclass(frozen=True)
class ParseResult:
    """One Drain decision for a single input line."""

    template_id: int
    template: str
    change_type: ChangeType
    cluster_size: int


class DrainParser:
    """Online log-template miner with structured logging."""

    def __init__(
        self,
        similarity_threshold: float = 0.4,
        depth: int = 4,
        max_children: int = 100,
    ) -> None:
        self._sim_th = similarity_threshold
        self._depth = depth
        self._max_children = max_children
        self._miner = self._build_miner()

    def _build_miner(self) -> TemplateMiner:
        config = TemplateMinerConfig()
        config.drain_sim_th = self._sim_th
        config.drain_depth = self._depth
        config.drain_max_children = self._max_children
        return TemplateMiner(MemoryBufferPersistence(), config)

    def parse_line(self, line: str) -> ParseResult:
        """Feed one line, log the decision, return a typed result."""
        result = self._miner.add_log_message(line.rstrip("\n"))
        change_type = _DRAIN_TO_SPEC[result["change_type"]]
        parsed = ParseResult(
            template_id=int(result["cluster_id"]),
            template=result["template_mined"],
            change_type=change_type,
            cluster_size=int(result["cluster_size"]),
        )
        logger.info(
            "drain decision: id=%d size=%d change=%s template=%r",
            parsed.template_id,
            parsed.cluster_size,
            parsed.change_type,
            parsed.template,
        )
        return parsed

    def parse_stream(
        self,
        lines: Iterable[str],
        log_progress_every: int = 1000,
    ) -> list[ParseResult]:
        """Process many lines with a tqdm bar; return the per-line results."""
        results: list[ParseResult] = []
        for i, line in enumerate(tqdm(lines, desc="drain parse"), start=1):
            results.append(self.parse_line(line))
            if log_progress_every and i % log_progress_every == 0:
                logger.info(
                    "progress: %d lines, %d templates so far",
                    i,
                    len(self._miner.drain.clusters),
                )
        return results

    def get_templates(self) -> list[dict]:
        """Return current templates as a list of {id, template, support}."""
        return [
            {
                "id": int(c.cluster_id),
                "template": c.get_template(),
                "support": int(c.size),
            }
            for c in self._miner.drain.clusters
        ]

    def export_tree_json(self, path: str | Path) -> None:
        """Persist current parser state (config + cluster list) as JSON."""
        payload = {
            "config": {
                "similarity_threshold": self._sim_th,
                "depth": self._depth,
                "max_children": self._max_children,
            },
            "templates": self.get_templates(),
        }
        save_json(payload, path)

    def reset(self) -> None:
        """Drop all learned state; start fresh with the same config."""
        self._miner = self._build_miner()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    demo_lines = [
        "Connected from 192.168.1.10",
        "Connected from 192.168.1.20",
        "Established session 0xABC with timeout 30000",
        "Established session 0xDEF with timeout 30000",
        "Got 5 votes for leader election",
        "Got 7 votes for leader election",
        "Connected from 10.0.0.1",
        "Got 9 votes for leader election",
        "Disconnected client",
        "Disconnected client",
    ]

    parser = DrainParser()
    print("Online demo — feeding 10 lines:")
    for idx, line in enumerate(demo_lines, start=1):
        r = parser.parse_line(line)
        print(f"  line {idx:>2}: {r.change_type:<18} -> id={r.template_id} template={r.template!r}")

    print("\nFinal templates:")
    for t in parser.get_templates():
        print(f"  id={t['id']:>2} support={t['support']:>2} template={t['template']!r}")

    print(asdict(ParseResult(0, "x", "matched_existing", 1)))  # touch dataclass to confirm import-safe
