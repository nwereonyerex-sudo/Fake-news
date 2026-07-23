"""
Feature extraction / vectorization: TF-IDF, Bag-of-Words, GloVe,
and Transformer embeddings (BERT/RoBERTa).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

from src.preprocess import LinguisticFeatureExtractor, TextPreprocessor

EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent / "data" / "embeddings"


def build_tfidf_pipeline(**tfidf_kwargs) -> Pipeline:
    """TextPreprocessor -> TfidfVectorizer, ready to fit on a raw text Series."""
    params = dict(max_features=20000, ngram_range=(1, 2), min_df=2)
    params.update(tfidf_kwargs)
    return Pipeline([("prep", TextPreprocessor()), ("tfidf", TfidfVectorizer(**params))])


def build_bow_pipeline(**bow_kwargs) -> Pipeline:
    """TextPreprocessor -> CountVectorizer (Bag-of-Words), ready to fit on a raw text Series."""
    params = dict(max_features=20000, ngram_range=(1, 1), min_df=2)
    params.update(bow_kwargs)
    return Pipeline([("prep", TextPreprocessor()), ("bow", CountVectorizer(**params))])


def with_linguistic_features(vectorizer_pipeline: Pipeline) -> FeatureUnion:
    """Combine a vectorized-text pipeline with the raw-text linguistic
    features (sensational/emotional word counts, caps ratio, ...) side by
    side, per CLAUDE.md's Pipeline/ColumnTransformer preference.
    """
    return FeatureUnion([("text", vectorizer_pipeline), ("linguistic", LinguisticFeatureExtractor())])


class GloVeEmbedder(BaseEstimator, TransformerMixin):
    """Sentence embeddings via averaged pretrained GloVe word vectors.

    Expects a standard GloVe text file (one token followed by
    space-separated floats per line), e.g. glove.6B.100d.txt from
    https://nlp.stanford.edu/projects/glove/ - not shipped with the repo
    (data/embeddings/ is gitignored); download and place it there.
    Out-of-vocabulary tokens are skipped; a text with no known tokens maps
    to a zero vector rather than raising.
    """

    def __init__(self, glove_path: str | Path, dim: int | None = None):
        self.glove_path = Path(glove_path)
        self.dim = dim

    def _load(self):
        if getattr(self, "_vectors", None) is not None:
            return
        if not self.glove_path.exists():
            raise FileNotFoundError(
                f"GloVe file not found at {self.glove_path}. Download one from "
                "https://nlp.stanford.edu/projects/glove/ (e.g. glove.6B.100d.txt) "
                "and place it there, or point glove_path at an existing file."
            )
        vectors: dict[str, np.ndarray] = {}
        dim = self.dim
        with open(self.glove_path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip().split(" ")
                word, values = parts[0], parts[1:]
                vec = np.asarray(values, dtype=np.float32)
                if dim is None:
                    dim = len(vec)
                elif len(vec) != dim:
                    continue
                vectors[word] = vec
        self._vectors = vectors
        self._dim = dim if dim is not None else 0

    def fit(self, X, y=None):
        self._load()
        return self

    def transform(self, X) -> np.ndarray:
        self._load()
        out = np.zeros((len(X), self._dim), dtype=np.float32)
        for i, text in enumerate(X):
            tokens = str(text).lower().split()
            vecs = [self._vectors[t] for t in tokens if t in self._vectors]
            if vecs:
                out[i] = np.mean(vecs, axis=0)
        return out


class TransformerEmbedder(BaseEstimator, TransformerMixin):
    """Sentence embeddings via mean-pooled pretrained Transformer hidden
    states (BERT/RoBERTa or any AutoModel-compatible checkpoint).

    Attention-mask-weighted mean pooling over the last hidden layer -
    padding tokens don't contribute to the sentence vector. Model and
    tokenizer load lazily on first fit/transform so importing this module
    never triggers a network call.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 256,
        batch_size: int = 16,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device

    def _load(self):
        if getattr(self, "_model", None) is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()

    def fit(self, X, y=None):
        self._load()
        return self

    def transform(self, X) -> np.ndarray:
        self._load()
        torch = self._torch
        texts = [str(t) for t in X]
        chunks = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self._device)
                hidden = self._model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts
                chunks.append(pooled.cpu().numpy())
        return np.concatenate(chunks, axis=0)


def main() -> None:
    """Demo: fit TF-IDF and BoW on data/raw/Fake.csv and report shapes.

    GloVe/Transformer embedders need external resources (an embeddings
    file / a model download) and are exercised separately, not by default
    here.
    """
    from src.preprocess import load_raw_data

    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    df = load_raw_data(raw_dir / "Fake.csv", label=1).head(200)

    tfidf = build_tfidf_pipeline()
    tfidf_matrix = tfidf.fit_transform(df["text"])
    print(f"TF-IDF matrix: {tfidf_matrix.shape}")

    bow = build_bow_pipeline()
    bow_matrix = bow.fit_transform(df["text"])
    print(f"Bag-of-Words matrix: {bow_matrix.shape}")


if __name__ == "__main__":
    main()
