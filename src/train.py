"""
Model training script.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import TorchTextClassifier, Vocabulary, build_model
from src.preprocess import load_raw_data, preprocess_text, strip_leading_dateline
from src.save_model import ModelArtifact, save_artifact

# Backstop for the ~0.8% of rows where strip_leading_dateline() doesn't
# catch the outlet tag (e.g. a correction notice or byline pushes it past
# the leading position) - specific to this dataset's known Reuters-dateline
# shortcut (see load_training_data()'s docstring), not a general cleaning rule.
_RESIDUAL_OUTLET_TAG_RE = re.compile(r"\(Reuters\)\s*-?\s*")

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


class SingleClassDataError(RuntimeError):
    """The training data doesn't contain both real and fake examples.

    Per CLAUDE.md's leakage-guard rule: a classifier "trained" on a single
    class trivially reports a perfect, meaningless 1.0 on itself. Refuse
    rather than silently producing that number.
    """


def load_training_data(fake_path: Path = RAW_DIR / "Fake.csv", real_path: Path | None = None) -> pd.DataFrame:
    """Load and concatenate Fake.csv (+ True.csv if present).

    Strips any leading wire-service dateline (e.g. "WASHINGTON (Reuters) -")
    from `text`, uniformly across both classes, before returning. This
    dataset pairing has a well-documented leakage shortcut: ~99% of the
    real-news file carries a "(Reuters)" tag that ~0% of the fake-news
    file does, which a model can trivially key off instead of learning
    real fake-vs-real content signal. See the "Corrections & Clarifications"
    build log and the project memory for the full investigation (a trivial
    "(Reuters) in text" rule alone scored 99.5% on held-out data prior to
    this fix).
    """
    fake_path = Path(fake_path)
    real_path = Path(real_path) if real_path is not None else RAW_DIR / "True.csv"

    frames = [load_raw_data(fake_path, label=1)]
    if real_path.exists():
        frames.append(load_raw_data(real_path, label=0))

    df = pd.concat(frames, ignore_index=True)
    n_classes = df["label"].nunique()
    if n_classes < 2:
        raise SingleClassDataError(
            f"Training data only contains label(s) {sorted(df['label'].unique())} - "
            f"{fake_path.name} alone is all one class. Add a real-news counterpart at "
            f"{real_path} (e.g. the ISOT True.csv) before training; a classifier fit on "
            "a single class would trivially report perfect (and meaningless) metrics."
        )

    df["text"] = df["text"].astype(str).apply(strip_leading_dateline).apply(
        lambda t: _RESIDUAL_OUTLET_TAG_RE.sub("", t)
    )
    return df


def train(
    architecture: str = "lstm",
    epochs: int = 5,
    batch_size: int = 32,
    max_len: int = 200,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    min_freq: int = 2,
    test_size: float = 0.2,
    seed: int = 42,
    patience: int = 3,
    fake_path: Path = RAW_DIR / "Fake.csv",
    real_path: Path | None = None,
    save_path: Path | None = None,
) -> dict:
    """Train and save a model artifact to models/fake_news_classifier.joblib.

    `save_path` overrides where the artifact is written - pass a scratch
    path when experimenting or testing, or a real run will silently
    overwrite the production model with a throwaway one (this bit during
    development: a quick sanity-check call clobbered a full-dataset,
    early-stopped model with a 2,000-row test run's).
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs} - epochs=0 would save an untrained, random-weights model")
    if patience < 1:
        raise ValueError(
            f"patience must be >= 1, got {patience} - patience=0 stops after the very first epoch "
            "regardless of whether it improved, which is never what's intended"
        )

    df = load_training_data(fake_path, real_path)

    texts = df["text"].astype(str).tolist()
    labels = df["label"].to_numpy()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    # Vocabulary is built from the training split only - fitting it on
    # validation text too would leak validation-set word statistics into
    # what the model can "see", the same leakage risk CLAUDE.md flags for
    # metrics.
    train_tokens = [preprocess_text(t).split() for t in train_texts]
    val_tokens = [preprocess_text(t).split() for t in val_texts]
    vocab = Vocabulary(min_freq=min_freq).build(train_tokens)

    def encode_all(token_lists: list[list[str]]) -> torch.Tensor:
        arr = np.array([vocab.encode(toks, max_len) for toks in token_lists], dtype=np.int64)
        return torch.from_numpy(arr)

    X_train, X_val = encode_all(train_tokens), encode_all(val_tokens)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    y_val = torch.tensor(val_labels, dtype=torch.long)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = build_model(
        architecture, vocab_size=len(vocab), embed_dim=embed_dim, hidden_dim=hidden_dim, num_classes=2
    ).to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Early stopping: keep the weights from whichever epoch had the lowest
    # val_loss, not just whatever epoch training happened to end on. Loss
    # is judged on a real per-run basis (a well-trained run's val_loss
    # legitimately jitters a little), so "improved" needs to clear a small
    # epsilon rather than any decrease at all, or noise alone would keep
    # resetting the patience counter and early stopping would never fire.
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_loader.dataset)

        model.eval()
        running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                running += criterion(model(xb), yb).item() * xb.size(0)
        val_loss = running / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            marker = " (best)"
        else:
            epochs_without_improvement += 1
            marker = f" (no improvement, {epochs_without_improvement}/{patience})"

        print(f"epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}{marker}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}: no val_loss improvement for {patience} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    wrapped = TorchTextClassifier(model=model.to("cpu"), vocab=vocab, max_len=max_len, device="cpu")
    artifact = ModelArtifact(
        model=wrapped,
        metadata={
            "architecture": architecture,
            "epochs_requested": epochs,
            "epochs_trained": len(history["train_loss"]),
            "best_val_loss": best_val_loss,
            "vocab_size": len(vocab),
            "max_len": max_len,
            "train_size": len(train_texts),
            "val_size": len(val_texts),
        },
    )
    path = save_artifact(artifact, save_path) if save_path is not None else save_artifact(artifact)
    print(f"Saved trained model to {path} (best val_loss={best_val_loss:.4f})")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", default="lstm", choices=["lstm", "lstm_gru", "lstm_cnn"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--patience", type=int, default=3, help="stop after this many epochs with no val_loss improvement")
    args = parser.parse_args()
    train(
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
