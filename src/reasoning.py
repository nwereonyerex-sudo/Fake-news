"""
Logic-based / rule-based text reasoning checks used as a supporting
signal alongside the model prediction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocess import EMOTIONAL_WORDS, SENSATIONAL_WORDS, extract_linguistic_features

_REPEATED_PUNCT_RE = re.compile(r"[!?]{2,}")
_ABSOLUTIST_WORDS = {
    "always", "never", "everyone", "no one", "nobody", "everybody",
    "completely", "totally", "entirely", "proven", "undeniable",
    "guaranteed", "100%", "definitely", "absolutely", "certainly",
}
_VAGUE_SOURCE_PHRASES = {
    "sources say", "some people say", "many believe", "it is said",
    "reports suggest", "allegedly", "insiders say", "many are saying",
    "people are saying", "some say", "critics say", "experts say",
}
_NAMED_ATTRIBUTION_RE = re.compile(
    r"\b[A-Z][a-zA-Z.]+(?:\s+[A-Z][a-zA-Z.]+)?\s+(?:said|says|told|stated|wrote|tweeted|announced)\b"
    r"|according to [A-Z]"
)


@dataclass
class RuleResult:
    name: str
    triggered: bool
    score: float
    explanation: str


@dataclass
class ReasoningReport:
    overall_score: float
    flags: list[str] = field(default_factory=list)
    results: list[RuleResult] = field(default_factory=list)


def rule_excessive_caps(text: str, threshold: float = 0.15) -> RuleResult:
    ratio = extract_linguistic_features(text)["all_caps_word_ratio"]
    triggered = ratio > threshold
    return RuleResult(
        name="excessive_caps",
        triggered=triggered,
        score=min(ratio / threshold, 1.0) if threshold else 0.0,
        explanation=f"{ratio:.0%} of words are ALL-CAPS" + (" (above threshold)" if triggered else ""),
    )


def rule_sensational_language(text: str, min_hits: int = 2) -> RuleResult:
    lower = text.lower() if isinstance(text, str) else ""
    hits = sum(lower.count(w) for w in SENSATIONAL_WORDS) + sum(lower.count(w) for w in EMOTIONAL_WORDS)
    triggered = hits >= min_hits
    return RuleResult(
        name="sensational_language",
        triggered=triggered,
        score=min(hits / (min_hits * 2), 1.0),
        explanation=f"{hits} sensational/emotionally-charged word occurrence(s)",
    )


def rule_absolutist_language(text: str, min_hits: int = 2) -> RuleResult:
    lower = text.lower() if isinstance(text, str) else ""
    hits = sum(lower.count(w) for w in _ABSOLUTIST_WORDS)
    triggered = hits >= min_hits
    return RuleResult(
        name="absolutist_language",
        triggered=triggered,
        score=min(hits / (min_hits * 2), 1.0),
        explanation=f"{hits} absolutist/superlative word occurrence(s) (e.g. 'always', 'proven', 'guaranteed')",
    )


def rule_vague_sourcing(text: str) -> RuleResult:
    lower = text.lower() if isinstance(text, str) else ""
    vague_hits = sum(lower.count(p) for p in _VAGUE_SOURCE_PHRASES)
    named_hits = len(_NAMED_ATTRIBUTION_RE.findall(text)) if isinstance(text, str) else 0
    triggered = vague_hits > 0 and vague_hits > named_hits
    return RuleResult(
        name="vague_sourcing",
        triggered=triggered,
        score=min(vague_hits / (named_hits + 1), 1.0),
        explanation=(
            f"{vague_hits} vague-attribution phrase(s) ('sources say', ...) vs "
            f"{named_hits} named attribution(s) ('X said', 'according to X')"
        ),
    )


def rule_excessive_punctuation(text: str, min_hits: int = 1) -> RuleResult:
    hits = len(_REPEATED_PUNCT_RE.findall(text)) if isinstance(text, str) else 0
    triggered = hits >= min_hits
    return RuleResult(
        name="excessive_punctuation",
        triggered=triggered,
        score=min(hits / (min_hits * 3), 1.0),
        explanation=f"{hits} run(s) of repeated '!'/'?' (e.g. '!!!', '?!')",
    )


RULES = (
    rule_excessive_caps,
    rule_sensational_language,
    rule_absolutist_language,
    rule_vague_sourcing,
    rule_excessive_punctuation,
)


def evaluate(text: str) -> ReasoningReport:
    """Run every rule against `text` and aggregate into one report.

    `overall_score` is the mean of the individual rule scores (each in
    [0, 1]) - a coarse, interpretable signal meant to sit alongside the
    model's prediction, not replace it. This is a heuristic, not ground
    truth: real news can trip these rules (a quote-heavy piece about an
    "unbelievable, shocking" event) and fake news can avoid all of them.
    """
    results = [rule(text) for rule in RULES]
    flags = [r.name for r in results if r.triggered]
    overall_score = sum(r.score for r in results) / len(results) if results else 0.0
    return ReasoningReport(overall_score=overall_score, flags=flags, results=results)


def check_headline_body_mismatch(title: str, text: str, overlap_threshold: float = 0.2) -> RuleResult:
    """Flag a sensational headline whose keywords barely appear in the body -
    a common clickbait pattern (needs both title and body, so it's kept
    separate from the single-text rules above / the transformer below).
    """
    title_lower = (title or "").lower()
    is_sensational = any(w in title_lower for w in SENSATIONAL_WORDS)

    title_words = {w for w in re.findall(r"[a-z]+", title_lower) if len(w) > 3}
    body_lower = (text or "").lower()
    overlap = (
        sum(1 for w in title_words if w in body_lower) / len(title_words)
        if title_words else 1.0
    )

    triggered = is_sensational and overlap < overlap_threshold
    return RuleResult(
        name="headline_body_mismatch",
        triggered=triggered,
        score=(1.0 - overlap) if is_sensational else 0.0,
        explanation=(
            f"sensational headline, only {overlap:.0%} of its keywords appear in the body"
            if is_sensational
            else "headline not flagged as sensational"
        ),
    )


class RuleBasedReasoner(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer: raw text Series -> one score column per
    rule in RULES, for combining with vectorized/linguistic features via
    FeatureUnion (see src/features.py).
    """

    feature_names_ = [rule.__name__.removeprefix("rule_") for rule in RULES]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        series = X if isinstance(X, pd.Series) else pd.Series(X)
        rows = [[rule(t).score for rule in RULES] for t in series.astype(str)]
        return pd.DataFrame(rows, columns=self.feature_names_).to_numpy()

    def get_feature_names_out(self, input_features=None):
        return list(self.feature_names_)


def main() -> None:
    """Demo: run the reasoning layer against a few real Fake.csv rows."""
    from src.preprocess import load_raw_data

    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    df = load_raw_data(raw_dir / "Fake.csv", label=1).head(5)

    for _, row in df.iterrows():
        report = evaluate(row["text"])
        mismatch = check_headline_body_mismatch(row["title"], row["text"])
        print(f"\n{row['title'][:70]!r}")
        print(f"  overall_score={report.overall_score:.2f} flags={report.flags}")
        print(f"  {mismatch.name}: triggered={mismatch.triggered} ({mismatch.explanation})")


if __name__ == "__main__":
    main()
