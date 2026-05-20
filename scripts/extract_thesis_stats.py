"""Mini extraction for thesis sections 3.1 (Drain) and 3.2 (BERT).

Re-runs Drain parsing on the full ZooKeeper corpus and beta(template)
vectorization on the 77 templates inside one fresh single-thread CPU process,
plus computes template support statistics and recovers the cosine-similarity
sanity-check pairs from notebook 02.

Output: data/processed/thesis_stats/extraction.json
"""

from __future__ import annotations

# Pin BLAS / OpenMP threads BEFORE importing numpy or torch. The env vars are
# read at native library load time, so setting them after import has no effect.
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

torch.set_num_threads(1)
torch.manual_seed(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bert.vectorizer import TemplateVectorizer  # noqa: E402
from src.drain.parser import DrainParser  # noqa: E402
from src.io.persistence import load_json, save_json  # noqa: E402
from src.preprocessing.regex_normalizer import RegexNormalizer  # noqa: E402

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
EMBEDDINGS_DIR = DATA_PROCESSED / "embeddings"
OUT_DIR = DATA_PROCESSED / "thesis_stats"
OUT_PATH = OUT_DIR / "extraction.json"

LOG_FULL = DATA_RAW / "Zookeeper.log"
MAPPING_FULL = EMBEDDINGS_DIR / "zookeeper_full_id_mapping.json"

# Same header as nb01 §3 — strips timestamp/level/node and exposes <Content>.
HEADER_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d+\s+-\s+\w+\s+\[(.+)\]\s+-\s+(.*)$"
)
N_RECORDS_EXPECTED = 74380
N_TEMPLATES_EXPECTED = 77

DRAIN_CONFIG: dict[str, Any] = {
    "similarity_threshold": 0.4,
    "depth": 4,
    "max_children": 100,
    "placeholders": [
        "<IP>",
        "<NUM>",
        "<PATH>",
        "<HEX>",
        "<EXC>",
        "<SESSION>",
        "<UUID>",
    ],
}


def extract_content(raw: str) -> str:
    """Pull <Content> out of a ZooKeeper log line; raise on header mismatch."""
    m = HEADER_RE.match(raw)
    if m is None:
        raise ValueError(f"header did not match: {raw!r}")
    return m.group(2)


def measure_drain() -> dict[str, Any]:
    """Re-parse the full corpus; time the Drain parse loop only.

    File I/O and regex normalization are excluded from the timer; the
    reported wall time is exactly the cost of feeding 74 380 normalized
    lines through DrainParser.parse_stream.
    """
    print("[drain] reading raw lines...")
    with LOG_FULL.open(encoding="utf-8") as f:
        raw_lines = [line.rstrip("\n") for line in f]
    assert len(raw_lines) == N_RECORDS_EXPECTED, (
        f"expected {N_RECORDS_EXPECTED} raw lines, got {len(raw_lines)}"
    )

    normalizer = RegexNormalizer()
    contents = [extract_content(l) for l in raw_lines]
    normalized = [normalizer.normalize(c) for c in contents]

    parser = DrainParser(
        similarity_threshold=DRAIN_CONFIG["similarity_threshold"],
        depth=DRAIN_CONFIG["depth"],
        max_children=DRAIN_CONFIG["max_children"],
    )
    # Silence per-line logger.info from parse_line; otherwise it dominates wall time.
    logging.getLogger("src.drain.parser").setLevel(logging.WARNING)

    print("[drain] parsing 74380 normalized lines...")
    t0 = time.perf_counter()
    parser.parse_stream(normalized, log_progress_every=0)
    wall_time_s = time.perf_counter() - t0

    templates = parser.get_templates()
    n_templates = len(templates)
    assert n_templates == N_TEMPLATES_EXPECTED, (
        f"expected {N_TEMPLATES_EXPECTED} templates, got {n_templates}"
    )

    us_per_record = wall_time_s / len(normalized) * 1e6
    print(
        f"[drain] wall={wall_time_s:.3f}s "
        f"per-record={us_per_record:.2f}us templates={n_templates}"
    )

    return {
        "wall_time_s": float(wall_time_s),
        "us_per_record": float(us_per_record),
        "n_records": N_RECORDS_EXPECTED,
        "n_templates": N_TEMPLATES_EXPECTED,
        "config": DRAIN_CONFIG,
        "timing_source": "rerun",
    }


def compute_support_stats(mapping: list[dict[str, Any]]) -> dict[str, Any]:
    """Top/bottom 3, mean/median/std, and max-fraction-of-corpus over 77 templates."""
    assert len(mapping) == N_TEMPLATES_EXPECTED, (
        f"expected {N_TEMPLATES_EXPECTED} entries in id mapping, got {len(mapping)}"
    )

    sorted_desc = sorted(mapping, key=lambda r: r["support"], reverse=True)
    supports = np.array([r["support"] for r in mapping], dtype=np.int64)
    total = int(supports.sum())
    # Sanity tie-back: sum of supports must equal corpus size exactly.
    assert total == N_RECORDS_EXPECTED, (
        f"support sum {total} != record count {N_RECORDS_EXPECTED}"
    )

    def slim(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "template_id": int(r["template_id"]),
            "support": int(r["support"]),
            "template": r["template"],
        }

    return {
        "top3": [slim(r) for r in sorted_desc[:3]],
        "bottom3": [slim(r) for r in sorted_desc[-3:]],
        "median": float(np.median(supports)),
        "mean": float(supports.mean()),
        "std": float(supports.std(ddof=0)),
        "max_fraction_of_corpus": float(supports.max() / N_RECORDS_EXPECTED),
    }


def measure_bert(
    mapping: list[dict[str, Any]],
) -> tuple[dict[str, Any], TemplateVectorizer]:
    """Re-vectorize all 77 templates; time tokenizer encode and model forward separately."""
    templates = [r["template"] for r in mapping]
    n = len(templates)
    assert n == N_TEMPLATES_EXPECTED

    print("[bert] loading extended vectorizer (model + tokenizer init excluded from timing)...")
    vectorizer = TemplateVectorizer(device="cpu")

    batch_size = 32  # matches nb02 §5
    starts = list(range(0, n, batch_size))

    # Stage 1: batched encode loop — measured in isolation so model forward
    # does not contaminate the tokenizer timing.
    tokenizer_s = 0.0
    encoded_batches: list[dict[str, torch.Tensor]] = []
    print("[bert] timing tokenizer (batched encode)...")
    for s in starts:
        batch = templates[s : s + batch_size]
        t0 = time.perf_counter()
        encoded = vectorizer.tokenizer(
            batch,
            add_special_tokens=True,
            max_length=vectorizer.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        tokenizer_s += time.perf_counter() - t0
        encoded_batches.append(encoded)

    # Stage 2: forward passes only.
    forward_s = 0.0
    print("[bert] timing forward passes...")
    for encoded in encoded_batches:
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = vectorizer.model(**encoded)
        forward_s += time.perf_counter() - t0

    total_s = tokenizer_s + forward_s
    ms_per_template = total_s / n * 1000.0
    print(
        f"[bert] tokenizer={tokenizer_s:.3f}s forward={forward_s:.3f}s "
        f"total={total_s:.3f}s per-template={ms_per_template:.2f}ms"
    )

    return (
        {
            "tokenizer_s": float(tokenizer_s),
            "forward_s": float(forward_s),
            "total_s": float(total_s),
            "ms_per_template": float(ms_per_template),
            "timing_source": "rerun",
        },
        vectorizer,
    )


def _find_template(substring: str, mapping: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the first mapping entry whose template contains `substring`."""
    for r in mapping:
        if substring in r["template"]:
            return r
    raise ValueError(f"no template contains substring: {substring!r}")


def measure_pairs(
    mapping: list[dict[str, Any]], vectorizer: TemplateVectorizer
) -> dict[str, Any]:
    """Reconstruct pairs A/B/C from nb02 §6 and re-measure their cosines.

    nb02 already prints cosines 0.8494 / 0.9588 / 0.9109 for these exact
    template pairs and explicitly labels A as 'worker lifecycle' (related),
    B as 'connections' (related) and C as 'connection vs snapshot' — the
    semantically distant pair whose still-high cosine flags BERT [CLS]
    anisotropy. Labels and substrings are copied verbatim from that cell.
    """
    a1 = _find_template("Send worker leaving thread", mapping)
    a2 = _find_template("Interrupted while waiting for message", mapping)
    b1 = _find_template("Received connection request", mapping)
    b2 = _find_template("Accepted socket connection", mapping)
    c1 = b1  # Pair C reuses the "Received connection request" template.
    c2 = _find_template("Reading snapshot", mapping)

    def pair(
        t1: dict[str, Any], t2: dict[str, Any], interpretation: str
    ) -> dict[str, Any]:
        cos = vectorizer.cosine_similarity(t1["template"], t2["template"])
        return {
            "template_ids": [int(t1["template_id"]), int(t2["template_id"])],
            "templates": [t1["template"], t2["template"]],
            "cosine": float(cos),
            "interpretation": interpretation,
        }

    pair_a = pair(
        a1,
        a2,
        "semantically related templates from the worker-thread lifecycle "
        "(Send worker leaving thread vs Interrupted while waiting for message)",
    )
    pair_b = pair(
        b1,
        b2,
        "paraphrase-like pair about incoming connection events "
        "(Received connection request vs Accepted socket connection)",
    )
    pair_c = pair(
        c1,
        c2,
        "semantically distant pair (incoming connection vs snapshot read); "
        "its still-high cosine is the anisotropy red flag for BERT [CLS] "
        "discussed in thesis §1.7",
    )

    return {
        "A": pair_a,
        "B": pair_b,
        "C": pair_c,
        "recovered_from_nb02": True,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    drain_stats = measure_drain()

    mapping = load_json(MAPPING_FULL)
    support_stats = compute_support_stats(mapping)
    bert_timing, vectorizer = measure_bert(mapping)
    bert_pairs = measure_pairs(mapping, vectorizer)

    extraction = {
        "drain": drain_stats,
        "support_stats": support_stats,
        "bert_pairs": bert_pairs,
        "bert_timing": bert_timing,
    }
    save_json(extraction, OUT_PATH)

    # Inline checks (asserts + prints in lieu of pytest, per CLAUDE.md §8).
    loaded = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    for key in ("drain", "support_stats", "bert_pairs", "bert_timing"):
        assert key in loaded and loaded[key], f"missing or empty top-level key: {key}"
    assert loaded["drain"]["n_templates"] == N_TEMPLATES_EXPECTED
    assert loaded["drain"]["n_records"] == N_RECORDS_EXPECTED
    for k in ("A", "B", "C"):
        p = loaded["bert_pairs"][k]
        assert len(p["template_ids"]) == 2, f"pair {k} must have two template_ids"
        assert isinstance(p["cosine"], float), f"pair {k} cosine must be float"

    d = drain_stats
    s = support_stats
    b = bert_pairs
    t = bert_timing
    print()
    print("=" * 72)
    print("THESIS EXTRACTION — SUMMARY")
    print("=" * 72)
    print(f"Drain   wall      : {d['wall_time_s']:.3f} s on {d['n_records']} records")
    print(f"Drain   per-rec   : {d['us_per_record']:.2f} us")
    print(f"Templates         : {d['n_templates']}")
    print(f"Support median    : {s['median']:.1f}")
    print(f"Support mean      : {s['mean']:.1f}  std={s['std']:.1f}")
    print(f"Top1 fraction     : {s['max_fraction_of_corpus']:.4f}")
    print(f"Pair A cosine     : {b['A']['cosine']:.4f}  ids={b['A']['template_ids']}")
    print(f"Pair B cosine     : {b['B']['cosine']:.4f}  ids={b['B']['template_ids']}")
    print(f"Pair C cosine     : {b['C']['cosine']:.4f}  ids={b['C']['template_ids']}")
    print(f"BERT tokenizer    : {t['tokenizer_s']:.3f} s")
    print(f"BERT forward      : {t['forward_s']:.3f} s")
    print(f"BERT total        : {t['total_s']:.3f} s")
    print(f"BERT per-template : {t['ms_per_template']:.2f} ms")
    print(f"Wrote             : {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
