"""Spherical KMeans + Silhouette-based k selection for BERT template vectors.

Pretrained BERT [CLS] vectors live in an anisotropic cone of R^768
(Ethayarajh 2019, thesis section 1.7). Plain Euclidean k-means on raw
vectors is therefore unreliable; we L2-normalize first so that Euclidean
distance on the unit hypersphere becomes monotone with cosine distance —
this is the spherical k-means trick.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class ClusteringPipeline:
    """KMeans over L2-normalized vectors, scored by cosine Silhouette."""

    def __init__(self, k_values: list[int], random_state: int = 42) -> None:
        self.k_values = list(k_values)
        self.random_state = random_state

    def get_l2_normalized(self, embeddings: np.ndarray) -> np.ndarray:
        """Return L2-normalized copy of `embeddings` (each row has unit norm)."""
        return normalize(embeddings, norm="l2")

    def fit_and_score(self, embeddings: np.ndarray) -> dict[int, float]:
        """Fit KMeans for every k in `self.k_values`, return {k: silhouette}."""
        x_norm = self.get_l2_normalized(embeddings)
        scores: dict[int, float] = {}
        for k in self.k_values:
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labels = km.fit_predict(x_norm)
            # Silhouette on the unit sphere with cosine — consistent with the
            # geometry KMeans actually optimizes after L2 normalization.
            score = float(silhouette_score(x_norm, labels, metric="cosine"))
            scores[k] = score
            logger.info("k=%d silhouette=%.4f", k, score)
        return scores

    def fit_best(
        self, embeddings: np.ndarray
    ) -> tuple[int, KMeans, np.ndarray]:
        """Pick k with the highest silhouette, refit, return (k, model, labels)."""
        scores = self.fit_and_score(embeddings)
        # Ties broken by smaller k (simpler model wins).
        best_k = max(scores, key=lambda k: (scores[k], -k))
        x_norm = self.get_l2_normalized(embeddings)
        km = KMeans(n_clusters=best_k, n_init=10, random_state=self.random_state)
        labels = km.fit_predict(x_norm)
        logger.info("best k=%d silhouette=%.4f", best_k, scores[best_k])
        return best_k, km, labels


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    rng = np.random.default_rng(42)
    centers = np.array([
        [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    blobs = np.vstack([
        rng.normal(loc=c, scale=0.2, size=(20, 8)) for c in centers
    ]).astype(np.float32)

    pipeline = ClusteringPipeline(k_values=[2, 3, 4], random_state=42)
    scores = pipeline.fit_and_score(blobs)
    assert max(scores, key=scores.get) == 3, f"expected best k=3, got scores={scores}"

    best_k, model, labels = pipeline.fit_best(blobs)
    assert best_k == 3
    assert labels.shape == (60,)
    assert len(set(labels.tolist())) == 3

    print("OK")
