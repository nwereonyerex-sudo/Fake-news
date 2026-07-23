# CLAUDE.md

This file provides guidance to Claude when working on this repository.

## Project Overview

This project trains a **Neural Network (LSTM-based) deep learning model**, enhanced with **Graph Neural Network (GNN)** propagation analysis, to detect fake news from a given news corpus. It combines classical NLP feature engineering with sequence-based deep learning and social-graph structural signals to classify news content as **real** or **fake**.

See `SKILLS.md` for the full list of underlying ML/NLP competencies this project draws on.

## Core Technologies

- **Language:** Python
- **Data handling:** numpy
- **ML utilities:** scikit-learn
- **Feature extraction:** TF-IDF Vectorizer, Bag-of-Words, GloVe embeddings, Transformer embeddings (BERT/RoBERTa)
- **Deep learning:** LSTM, LSTM-GRU, hybrid LSTM-CNN architectures
- **Graph analysis:** Graph Neural Networks (GNNs) for social propagation / bot-network detection
- **Model persistence:** joblib
- **Deployment:** REST API (Flask / FastAPI)

## Project Structure

```
fake-news-detector/
├── CLAUDE.md
├── SKILLS.md
├── data/
│   ├── raw/                # original datasets
│   └── processed/          # cleaned, tokenized data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── preprocess.py       # tokenization, stopword removal, lemmatization, linguistic feature engineering
│   ├── features.py         # TF-IDF / BoW / GloVe / Transformer embedding logic
│   ├── reasoning.py        # logic-based / rule-based text reasoning checks
│   ├── graph_analysis.py   # GNN-based propagation & bot-network detection
│   ├── model.py            # LSTM / LSTM-GRU / hybrid LSTM-CNN architecture
│   ├── train.py            # model training script
│   ├── evaluate.py         # precision, recall, F1, confusion matrix
│   ├── save_model.py       # joblib model persistence
│   ├── predict.py          # inference on new text
│   └── api.py              # REST API for deployment
├── models/                 # saved model files (.joblib)
├── requirements.txt
└── README.md
```

## Workflow Claude Should Follow

1. **Data collection** — use a labeled news corpus (e.g. Kaggle Fake/Real News, LIAR dataset).
2. **EDA** — check class balance, article length distribution, common words in fake vs. real articles.
3. **Preprocessing & feature engineering** — tokenize, remove stopwords, lemmatize; engineer linguistic features such as emotional bias and sensational-headline indicators.
4. **Vectorization** — convert text to numerical tensors via TF-IDF / BoW as a baseline, and GloVe or Transformer embeddings (BERT/RoBERTa) for richer representations.
5. **Logic reasoning layer** — apply rule-based checks as a supporting signal alongside model predictions.
6. **Sequence modeling** — build and train an LSTM, LSTM-GRU, or hybrid LSTM-CNN model to capture temporal and contextual patterns in the text.
7. **Graph analysis (optional/advanced)** — if social-sharing data is available, use a GNN to analyze propagation patterns and flag structural signals like bot-network spreading.
8. **Evaluation** — compute precision, recall, F1-score, and confusion matrix on a held-out test set.
9. **Model saving** — persist the trained model and vectorizer using joblib.
10. **Deployment** — wrap the saved model in a REST API (Flask/FastAPI) with a `/predict` endpoint.

## Model Performance Target

| Metric | Target |
|---|---|
| Precision | 1.0 |
| Recall | 1.0 |
| F1-score | 1.0 |
| Confusion Matrix | No misclassifications |

**Caution for Claude:** if evaluation ever produces a perfect 1.0 across precision/recall/F1 on real-world data, flag it and check for data leakage (train/test overlap, source-based shortcuts) before treating it as a genuine result. Validate with cross-validation and a held-out set from a different source/time period.

## Coding Conventions

- Keep preprocessing, feature extraction, modeling, and evaluation in separate modules under `src/` (see structure above) rather than one monolithic script.
- Save the fitted vectorizer alongside the trained model (both via joblib) so `predict.py` and `api.py` can reproduce the same feature space at inference time.
- Prefer scikit-learn's `Pipeline`/`ColumnTransformer` where it simplifies preprocessing + modeling steps.
- Write evaluation code that always reports precision, recall, F1, and confusion matrix together — never accuracy alone.
- New API endpoints go in `src/api.py`; keep inference logic itself in `src/predict.py` so it can be tested independently of the web layer.

## What Claude Should NOT Do

- Don't silently accept perfect (1.0) metrics without flagging the leakage risk noted above.
- Don't hardcode file paths outside of `data/`, `models/`, and `src/`.
- Don't mix notebook exploration code into `src/` modules — notebooks are for exploration, `src/` is for production-ready code.
