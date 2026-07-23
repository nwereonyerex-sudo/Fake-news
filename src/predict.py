"""
Inference on new/unseen text using the saved model and vectorizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from src.save_model import ModelArtifact, artifact_exists, load_artifact


class ModelNotTrainedError(RuntimeError):
    """No trained model artifact exists yet - run src/train.py first."""


@dataclass
class Prediction:
    label: str
    label_id: int
    confidence: float


@lru_cache(maxsize=1)
def _get_artifact() -> ModelArtifact:
    if not artifact_exists():
        raise ModelNotTrainedError(
            "No trained model found in models/fake_news_classifier.joblib. "
            "Run `python -m src.train` on a labeled dataset containing both "
            "real and fake examples first."
        )
    return load_artifact()


def is_model_ready() -> bool:
    return artifact_exists()


def predict_text(text: str) -> Prediction:
    if not text or not text.strip():
        raise ValueError("text must be a non-empty string")

    artifact = _get_artifact()
    probs = artifact.model.predict_proba([text])[0]
    label_id = int(probs.argmax())

    return Prediction(
        label=artifact.label_names.get(label_id, str(label_id)),
        label_id=label_id,
        confidence=float(probs[label_id]),
    )


def predict_batch(texts: list[str]) -> list[Prediction]:
    return [predict_text(t) for t in texts]
