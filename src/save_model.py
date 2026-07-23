"""
Model persistence using joblib (saves trained model + fitted vectorizer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ARTIFACT_PATH = MODELS_DIR / "fake_news_classifier.joblib"


@dataclass
class ModelArtifact:
    """Everything predict.py needs, bundled as one unit.

    `model` must be fitted and expose `predict_proba(texts: list[str]) ->
    ndarray[n_samples, n_classes]` end-to-end - i.e. it owns its own text
    preprocessing/vectorization internally (e.g. an sklearn Pipeline built
    from src.preprocess.TextPreprocessor + a vectorizer + a classifier, or
    an equivalent wrapper around a PyTorch model). predict.py never
    preprocesses text itself; the artifact is the single source of truth
    for how raw text becomes a prediction, so training and inference can
    never drift apart.
    """

    model: Any
    label_names: dict[int, str] = field(default_factory=lambda: {0: "real", 1: "fake"})
    metadata: dict = field(default_factory=dict)


def save_artifact(artifact: ModelArtifact, path: Path = ARTIFACT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_artifact(path: Path = ARTIFACT_PATH) -> ModelArtifact:
    if not path.exists():
        raise FileNotFoundError(f"No trained model artifact at {path}. Run src/train.py first.")
    return joblib.load(path)


def artifact_exists(path: Path = ARTIFACT_PATH) -> bool:
    return path.exists()
