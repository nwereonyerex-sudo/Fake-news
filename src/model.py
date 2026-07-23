"""
LSTM / LSTM-GRU / hybrid LSTM-CNN model architecture definitions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class Vocabulary:
    """Token <-> integer-id mapping built from tokenized text, with a
    frequency cutoff.

    The sequence models below need word order preserved, unlike
    features.py's TF-IDF/BoW/averaged-embedding vectors which collapse a
    document into one fixed-size vector - so text is encoded here as a
    padded/truncated integer sequence for nn.Embedding instead.
    """

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.token_to_id: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.id_to_token: list[str] = [PAD_TOKEN, UNK_TOKEN]

    def build(self, tokenized_texts: list[list[str]]) -> "Vocabulary":
        counts = Counter(tok for toks in tokenized_texts for tok in toks)
        for token, count in counts.items():
            if count >= self.min_freq and token not in self.token_to_id:
                self.token_to_id[token] = len(self.id_to_token)
                self.id_to_token.append(token)
        return self

    def encode(self, tokens: list[str], max_len: int) -> list[int]:
        unk_id = self.token_to_id[UNK_TOKEN]
        ids = [self.token_to_id.get(t, unk_id) for t in tokens[:max_len]]
        ids += [self.token_to_id[PAD_TOKEN]] * (max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.id_to_token)


def _init_embedding(embedding: nn.Embedding, pretrained: torch.Tensor | None, freeze: bool) -> None:
    if pretrained is not None:
        with torch.no_grad():
            embedding.weight.copy_(pretrained)
        embedding.weight.requires_grad = not freeze


def _masked_mean_pool(outputs: torch.Tensor, input_ids: torch.Tensor, pad_idx: int) -> torch.Tensor:
    mask = (input_ids != pad_idx).unsqueeze(-1).to(outputs.dtype)
    summed = (outputs * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class LSTMClassifier(nn.Module):
    """Embedding -> (bi)LSTM -> masked mean-pool over non-pad timesteps -> Linear."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 2,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
        pad_idx: int = 0,
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        _init_embedding(self.embedding, pretrained_embeddings, freeze_embeddings)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        outputs, _ = self.lstm(embedded)
        pooled = _masked_mean_pool(outputs, input_ids, self.pad_idx)
        return self.classifier(self.dropout(pooled))


class LSTMGRUClassifier(nn.Module):
    """Embedding -> LSTM -> GRU -> masked mean-pool -> Linear (hybrid recurrent stack)."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        _init_embedding(self.embedding, pretrained_embeddings, freeze_embeddings)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        gru_out, _ = self.gru(lstm_out)
        pooled = _masked_mean_pool(gru_out, input_ids, self.pad_idx)
        return self.classifier(self.dropout(pooled))


class LSTMCNNClassifier(nn.Module):
    """Embedding -> LSTM -> parallel Conv1d filters (multiple kernel sizes)
    over the LSTM outputs -> max-pool -> Linear (hybrid LSTM-CNN)."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 2,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        num_filters: int = 64,
        dropout: float = 0.3,
        pad_idx: int = 0,
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        _init_embedding(self.embedding, pretrained_embeddings, freeze_embeddings)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        lstm_out_dim = hidden_dim * 2
        self.convs = nn.ModuleList(
            [nn.Conv1d(lstm_out_dim, num_filters, kernel_size=k, padding=k // 2) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        conv_in = lstm_out.transpose(1, 2)
        pooled = [torch.amax(torch.relu(conv(conv_in)), dim=2) for conv in self.convs]
        concat = torch.cat(pooled, dim=1)
        return self.classifier(self.dropout(concat))


ARCHITECTURES = {
    "lstm": LSTMClassifier,
    "lstm_gru": LSTMGRUClassifier,
    "lstm_cnn": LSTMCNNClassifier,
}


def build_model(architecture: str, vocab_size: int, **kwargs) -> nn.Module:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture!r}; choose from {list(ARCHITECTURES)}")
    return ARCHITECTURES[architecture](vocab_size, **kwargs)


@dataclass
class TorchTextClassifier:
    """Wraps a trained sequence model + its vocabulary into the
    `predict_proba(texts) -> ndarray` contract src/predict.py expects (see
    src/save_model.py's ModelArtifact docstring), so predict.py never
    needs to know this is a PyTorch LSTM rather than an sklearn Pipeline.
    """

    model: nn.Module
    vocab: Vocabulary
    max_len: int
    device: str = "cpu"

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        from src.preprocess import preprocess_text

        self.model.eval()
        encoded = [self.vocab.encode(preprocess_text(t).split(), self.max_len) for t in texts]
        batch = torch.tensor(encoded, dtype=torch.long, device=self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(batch), dim=1)
        return probs.cpu().numpy()
