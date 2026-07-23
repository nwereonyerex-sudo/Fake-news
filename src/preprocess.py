"""
Text preprocessing: tokenization, stopword removal, lemmatization,
and linguistic feature engineering (emotional bias, sensational headlines).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

_URL_RE = re.compile(r"http\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")

# Matches a leading wire-service dateline, e.g. "WASHINGTON (Reuters) - " or
# "SEATTLE/WASHINGTON (Reuters) - " or just "(Reuters) - " with no city.
_LEADING_DATELINE_RE = re.compile(r"^\s*(?:[A-Z][A-Za-z.,'/\- ]{0,80}?)?\([^()]{0,40}\)\s*-\s*")

# Small, dependency-free lexicons for linguistic feature engineering.
# These are heuristic signals, not ground truth - they feed the reasoning
# layer (src/reasoning.py) and the model as auxiliary features, not a
# standalone verdict.
SENSATIONAL_WORDS = {
    "shocking", "bombshell", "explosive", "unbelievable", "outrageous",
    "scandal", "exposed", "secret", "conspiracy", "cover-up", "coverup",
    "miracle", "urgent", "breaking", "warning", "alert", "you won't believe",
    "destroys", "slams", "annihilates", "devastating", "terrifying",
}

EMOTIONAL_WORDS = {
    "hate", "love", "furious", "outraged", "disgusting", "horrific",
    "amazing", "incredible", "heartbreaking", "shameful", "disturbing",
    "fear", "angry", "afraid", "panic", "crisis", "chaos", "betrayal",
}


_nltk_data_ready = False


def _ensure_nltk_data() -> None:
    """Download required NLTK corpora on first use, quietly and idempotently.

    Called from tokenize()/remove_stopwords()/lemmatize(), i.e. up to 3x
    per row of text - the module-level flag makes every call after the
    first a no-op. Without it, this was ~16.5ms/call purely in
    nltk.data.find()'s filesystem scan, i.e. ~49.5ms of pure overhead per
    row (3 calls), which was in fact the dominant cost of preprocessing,
    not the tokenization/lemmatization work itself.
    """
    global _nltk_data_ready
    if _nltk_data_ready:
        return

    import nltk

    resources = {
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, package in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)
    _nltk_data_ready = True


def strip_leading_dateline(text: str) -> str:
    """Remove a leading wire-service dateline (e.g. "WASHINGTON (Reuters) - ")
    if present, otherwise return the text unchanged.

    Call this - on both classes, uniformly - *before* clean_text() whenever
    training data may come from wire-service sources. It exists because
    the outlet tag inside the dateline can become a trivial source-based
    shortcut: a model can learn "(Reuters) present" instead of any actual
    fake-vs-real content signal (see the caveat in src/train.py). Applying
    it uniformly, not just to the real-news class, keeps the cleaning step
    itself label-independent.
    """
    if not isinstance(text, str):
        return text
    return _LEADING_DATELINE_RE.sub("", text, count=1)


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/HTML/punctuation/digits, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    _ensure_nltk_data()
    from nltk.tokenize import word_tokenize

    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    _ensure_nltk_data()
    from nltk.corpus import stopwords

    stop_set = set(stopwords.words("english"))
    return [t for t in tokens if t not in stop_set and len(t) > 1]


def lemmatize(tokens: list[str]) -> list[str]:
    _ensure_nltk_data()
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess_text(text: str) -> str:
    """Full pipeline: clean -> tokenize -> remove stopwords -> lemmatize."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)


def extract_linguistic_features(raw_text: str) -> dict:
    """Heuristic linguistic signals computed on the *raw* (uncleaned) text,
    since punctuation/casing carry the signal here and would otherwise be
    stripped by clean_text().
    """
    if not isinstance(raw_text, str) or not raw_text:
        return {
            "text_length": 0,
            "word_count": 0,
            "avg_word_length": 0.0,
            "exclamation_count": 0,
            "question_count": 0,
            "all_caps_word_ratio": 0.0,
            "sensational_word_count": 0,
            "emotional_word_count": 0,
        }

    words = raw_text.split()
    word_count = len(words)
    all_caps_words = [w for w in words if w.isalpha() and w.isupper() and len(w) > 1]
    lower_text = raw_text.lower()

    return {
        "text_length": len(raw_text),
        "word_count": word_count,
        "avg_word_length": (sum(len(w) for w in words) / word_count) if word_count else 0.0,
        "exclamation_count": raw_text.count("!"),
        "question_count": raw_text.count("?"),
        "all_caps_word_ratio": (len(all_caps_words) / word_count) if word_count else 0.0,
        "sensational_word_count": sum(lower_text.count(w) for w in SENSATIONAL_WORDS),
        "emotional_word_count": sum(lower_text.count(w) for w in EMOTIONAL_WORDS),
    }


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer: raw text Series -> cleaned/lemmatized text Series.

    Fits into a Pipeline ahead of a vectorizer (see src/features.py).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        series = X if isinstance(X, pd.Series) else pd.Series(X)
        return series.astype(str).apply(preprocess_text).to_numpy()


class LinguisticFeatureExtractor(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer: raw text Series -> numeric linguistic feature matrix.

    Meant to be combined with vectorized text features via ColumnTransformer/
    FeatureUnion in src/features.py.
    """

    feature_names_ = [
        "text_length", "word_count", "avg_word_length", "exclamation_count",
        "question_count", "all_caps_word_ratio", "sensational_word_count",
        "emotional_word_count",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        series = X if isinstance(X, pd.Series) else pd.Series(X)
        records = series.astype(str).apply(extract_linguistic_features)
        return pd.DataFrame(list(records))[self.feature_names_].to_numpy()

    def get_feature_names_out(self, input_features=None):
        return list(self.feature_names_)


def load_raw_data(path: str | Path, label: int | None = None) -> pd.DataFrame:
    """Load a raw news CSV (title, text, subject, date) from data/raw/.

    If `label` is given, a constant `label` column is attached - this is how
    single-class source files like Fake.csv (all label=1) get merged with a
    real-news counterpart (label=0) upstream in train.py.
    """
    df = pd.read_csv(path)
    if label is not None:
        df["label"] = label
    return df


def build_preprocessed_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Attach cleaned_text + linguistic feature columns to a raw dataframe."""
    out = df.copy()
    out["cleaned_text"] = out[text_col].astype(str).apply(preprocess_text)
    feature_records = out[text_col].astype(str).apply(extract_linguistic_features)
    feature_df = pd.DataFrame(list(feature_records), index=out.index)
    return pd.concat([out, feature_df], axis=1)


def main() -> None:
    """CLI entry point: data/raw/Fake.csv -> data/processed/fake_preprocessed.csv.

    NOTE: Fake.csv alone is single-class (all label=1). This produces a
    preprocessed feature table for that file only; a genuine train/test
    split needs a real-news counterpart merged in (see src/train.py).
    """
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    fake_path = raw_dir / "Fake.csv"
    if not fake_path.exists():
        raise FileNotFoundError(f"Expected raw dataset at {fake_path}")

    df = load_raw_data(fake_path, label=1)
    processed = build_preprocessed_dataframe(df)
    out_path = processed_dir / "fake_preprocessed.csv"
    processed.to_csv(out_path, index=False)
    print(f"Wrote {len(processed)} rows to {out_path}")


if __name__ == "__main__":
    main()
