"""
Evaluation: precision, recall, F1-score, and confusion matrix.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


class SuspiciousPerfectScoreWarning(UserWarning):
    """precision/recall/F1 all hit a perfect 1.0.

    Per CLAUDE.md, that's a red flag on real-world data, not a result to
    celebrate - check for train/test overlap or a source-based shortcut
    before trusting it, and re-validate with cross-validation and a
    held-out set from a different source/time period.
    """


@dataclass
class EvaluationReport:
    accuracy: float
    precision: dict[str, float]
    recall: dict[str, float]
    f1: dict[str, float]
    support: dict[str, int]
    confusion_matrix: np.ndarray
    labels: list[str]
    perfect_score_flagged: bool = False

    def summary(self) -> str:
        lines = [
            "Evaluation report",
            "------------------",
            f"accuracy: {self.accuracy:.4f}  (never reported alone - see per-class metrics below)",
            "",
            f"{'class':<10}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}",
        ]
        for label in self.labels:
            lines.append(
                f"{label:<10}{self.precision[label]:>10.4f}{self.recall[label]:>10.4f}"
                f"{self.f1[label]:>10.4f}{self.support[label]:>10d}"
            )
        lines.append("")
        lines.append("confusion matrix (rows=true, cols=predicted):")
        lines.append("        " + "".join(f"{label:>10}" for label in self.labels))
        for label, row in zip(self.labels, self.confusion_matrix):
            lines.append(f"{label:<8}" + "".join(f"{v:>10d}" for v in row))
        if self.perfect_score_flagged:
            lines.append("")
            lines.append(
                "WARNING: precision/recall/F1 are all a perfect 1.0. Treat this as a "
                "red flag, not a win - check for train/test overlap or a source-based "
                "shortcut before trusting this number (see CLAUDE.md)."
            )
        return "\n".join(lines)


def evaluate_predictions(y_true, y_pred, label_names: dict[int, str] | None = None) -> EvaluationReport:
    """Precision/recall/F1/confusion-matrix together, always - never
    accuracy reported alone (CLAUDE.md's evaluation convention)."""
    label_names = label_names or {0: "real", 1: "fake"}
    class_ids = sorted(label_names)
    labels = [label_names[c] for c in class_ids]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_ids, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

    perfect = bool(np.all(precision == 1.0) and np.all(recall == 1.0) and np.all(f1 == 1.0))
    if perfect:
        warnings.warn(
            "precision/recall/F1 are all a perfect 1.0 - check for data leakage "
            "(train/test overlap, source-based shortcuts) before trusting this.",
            SuspiciousPerfectScoreWarning,
            stacklevel=2,
        )

    return EvaluationReport(
        accuracy=accuracy,
        precision=dict(zip(labels, precision.tolist())),
        recall=dict(zip(labels, recall.tolist())),
        f1=dict(zip(labels, f1.tolist())),
        support=dict(zip(labels, support.tolist())),
        confusion_matrix=cm,
        labels=labels,
        perfect_score_flagged=perfect,
    )


def evaluate_artifact(artifact, texts: list[str], y_true) -> EvaluationReport:
    """Evaluate a ModelArtifact (src/save_model.py) against held-out
    texts/labels. Works for any model satisfying the predict_proba(texts)
    contract - the sklearn baseline or the PyTorch LSTM wrapper alike -
    since it never touches the model internals directly. Not applicable
    to the GNN (src/graph_analysis.py): it classifies propagation graphs,
    not raw text, so it doesn't implement this contract.
    """
    probs = artifact.model.predict_proba(texts)
    y_pred = np.argmax(probs, axis=1)
    return evaluate_predictions(y_true, y_pred, label_names=artifact.label_names)


def cross_validate_pipeline(pipeline_factory, texts, labels, n_splits: int = 5, seed: int = 42) -> dict:
    """Stratified k-fold cross-validation for an sklearn-compatible
    pipeline (e.g. features.build_tfidf_pipeline() + a classifier).

    `pipeline_factory` is a zero-arg callable returning a fresh, unfitted
    estimator on every call - folds must never share fitted state, or
    the leakage this function exists to catch would just move one level
    up. Addresses CLAUDE.md's "validate with cross-validation ... before
    treating a perfect score as genuine".
    """
    from sklearn.model_selection import StratifiedKFold

    texts = np.asarray(texts, dtype=object)
    labels = np.asarray(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_reports = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
        estimator = pipeline_factory()
        estimator.fit(texts[train_idx], labels[train_idx])
        y_pred = estimator.predict(texts[test_idx])
        report = evaluate_predictions(labels[test_idx], y_pred)
        fold_reports.append(report)
        print(f"fold {fold + 1}/{n_splits}: accuracy={report.accuracy:.4f}")

    return {
        "fold_reports": fold_reports,
        "mean_accuracy": float(np.mean([r.accuracy for r in fold_reports])),
        "std_accuracy": float(np.std([r.accuracy for r in fold_reports])),
    }


def main() -> None:
    """CLI entry point: cross-validate a TF-IDF + LogisticRegression
    baseline and report held-out metrics.

    Reuses src.train.load_training_data(), so this hits the exact same
    SingleClassDataError guard as training does if only Fake.csv is
    present - evaluation on single-class data is exactly the scenario
    CLAUDE.md warns would produce a meaningless perfect score.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    from src.features import build_tfidf_pipeline
    from src.train import load_training_data

    df = load_training_data()
    texts, labels = df["text"].astype(str).to_numpy(), df["label"].to_numpy()

    def make_pipeline() -> Pipeline:
        pipeline = build_tfidf_pipeline()
        pipeline.steps.append(("clf", LogisticRegression(max_iter=1000)))
        return pipeline

    print("Cross-validation (5-fold, TF-IDF + LogisticRegression baseline):")
    cv_results = cross_validate_pipeline(make_pipeline, texts, labels)
    print(f"mean accuracy: {cv_results['mean_accuracy']:.4f} (+/- {cv_results['std_accuracy']:.4f})\n")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    pipeline = make_pipeline()
    pipeline.fit(train_texts, train_labels)
    y_pred = pipeline.predict(test_texts)

    report = evaluate_predictions(test_labels, y_pred)
    print(report.summary())


if __name__ == "__main__":
    main()
