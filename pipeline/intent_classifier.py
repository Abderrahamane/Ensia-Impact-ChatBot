"""Tiny multilingual intent classifier for bot routing."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from sentence_transformers import SentenceTransformer


@dataclass
class IntentPrediction:
    label: str
    confidence: float
    scores: dict[str, float]


class IntentRouter:
    def __init__(
        self,
        model_name: str,
        seed_path: Path,
        min_confidence: float = 0.42,
    ) -> None:
        self._embedder = SentenceTransformer(model_name)
        self._seed_path = seed_path
        self._min_confidence = min_confidence
        self._class_centroids: dict[str, list[float]] = {}
        self._train_from_seed()

    def _train_from_seed(self) -> None:
        if not self._seed_path.exists():
            raise RuntimeError(f"Intent seed file not found: {self._seed_path}")
        payload = json.loads(self._seed_path.read_text(encoding="utf-8"))
        classes = payload.get("classes", {})
        if not classes:
            raise RuntimeError("Intent seed file has no classes.")

        for label, examples in classes.items():
            texts = [self._normalize_text(str(t)) for t in (examples or []) if str(t).strip()]
            if not texts:
                continue
            vectors = self._embedder.encode(texts, normalize_embeddings=True)
            dim = len(vectors[0])
            centroid = [0.0] * dim
            for vec in vectors:
                for i, val in enumerate(vec):
                    centroid[i] += float(val)
            count = float(len(vectors))
            centroid = [v / count for v in centroid]
            norm = math.sqrt(sum(v * v for v in centroid)) or 1.0
            self._class_centroids[label] = [v / norm for v in centroid]

        if not self._class_centroids:
            raise RuntimeError("Intent classifier failed to build centroids from seed data.")

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _cosine(self, a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def predict(self, text: str) -> IntentPrediction:
        norm = self._normalize_text(text)
        if norm.startswith("/"):
            return IntentPrediction("admin_op", 1.0, {"admin_op": 1.0})

        vec_raw = self._embedder.encode([norm], normalize_embeddings=True)[0]
        vec = [float(x) for x in vec_raw]

        raw_scores = {label: self._cosine(vec, centroid) for label, centroid in self._class_centroids.items()}
        best_label = max(raw_scores, key=raw_scores.get)

        exps = {k: math.exp(v) for k, v in raw_scores.items()}
        denom = sum(exps.values()) or 1.0
        probs = {k: v / denom for k, v in exps.items()}
        best_conf = probs.get(best_label, 0.0)

        if best_conf < self._min_confidence:
            return IntentPrediction("ensia_query", best_conf, probs)
        return IntentPrediction(best_label, best_conf, probs)


