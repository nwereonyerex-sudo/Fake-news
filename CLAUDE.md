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

## Known Issues & Fixes (Engineering Log)

Real issues hit while building this project, and the fix applied for each. Read this before re-deriving a solution to something already solved here.

### Environment / toolchain

- **No `gh` CLI, `GITHUB_TOKEN`, or `brew` in the dev environment**, so a GitHub repo couldn't be created programmatically. *Fix:* used a local-only `git init` first; pushed later once the user supplied a GitHub URL. The system's `osxkeychain` git credential helper already had cached credentials, so `git push` worked without needing `gh` at all.
- **TensorFlow has no PyPI wheel for Python 3.14** (too new a Python release). *Fix:* standardized on **PyTorch** for both the LSTM (`src/model.py`) and the GNN (`src/graph_analysis.py`, which needs `torch-geometric` anyway) — don't add `tensorflow` back to `requirements.txt` on this environment without checking wheel availability first.
- **`gensim` fails to build on Python 3.14** — its Cython extensions reference `PyDictObject.ma_version_tag`, a CPython internal removed in 3.14. Because `pip install -r requirements.txt` installs everything as one transaction, this single failure blocked installing every other package too (even `numpy`). *Fix:* dropped `gensim` entirely; `src/features.py`'s `GloVeEmbedder` parses the plain-text GloVe format directly instead (a GloVe file is just whitespace-separated text — no library needed).
- **NLTK's downloader fails with `SSL: CERTIFICATE_VERIFY_FAILED`** on a bare venv (no CA bundle wired up for `urllib` by default on macOS). *Fix:* `src/preprocess.py` points `SSL_CERT_FILE` at `certifi`'s bundle before any download is attempted.

### Data

- **Only `Fake.csv` was available initially** — no real-news counterpart, so a "trained" classifier would trivially and meaninglessly score 1.0. *Fix:* `train.load_training_data()` detects single-class data and raises `SingleClassDataError` rather than silently producing a fake perfect score. Don't remove this guard to "make training work" — if it fires, the data is the problem, not the code.
- **The dataset arrived as duplicate copies in multiple locations** (Downloads root, project root, `notebooks/`). *Fix:* consolidated to `data/raw/`; `data/raw/*` and `models/*` are gitignored (only `.gitkeep` tracked) so large data/model files never bloat git history or get pushed to GitHub.
- **Severe source-based leakage in the Fake/True pairing once `True.csv` was added:** ~99% of `True.csv` rows start with a literal wire-service dateline (e.g. `"WASHINGTON (Reuters) - "`), vs. ~0.04% of `Fake.csv`. A trivial no-training rule (`"(Reuters)" in text` → real, else fake) scored **99.53%** on held-out data — nearly matching the first trained LSTM's 99.81% — proving the model had mostly learned the dateline, not real content signal. (`subject` also perfectly separates the two files with zero overlap, but that one's harmless since `subject` is never fed to the model.) *Fix:* `preprocess.strip_leading_dateline()` strips the dateline (handles multi-city/no-city/no-source variants), applied uniformly to **both** classes in `train.load_training_data()` — never just the real-news side, or the cleaning step itself would encode label information — plus a small backstop regex for the ~0.8% of rows where a byline/correction-notice prefix pushes the tag past the leading position. After the fix, the trivial-rule baseline collapsed to 52% (chance) while the real model held at 98.5%, which is good evidence of genuine learned signal. **Any future real-news source pulled from a wire service should be checked for this same shortcut before trusting near-perfect metrics** — see the "Model Performance Target" caution above.

### Performance

- **`preprocess._ensure_nltk_data()` re-scanned the filesystem for all 4 NLTK resources on every call** to `tokenize()`/`remove_stopwords()`/`lemmatize()` — 3x per row of text, measured at ~49.5ms of pure redundant overhead per row. This was effectively the *entire* preprocessing cost (full ~44,900-row dataset: ~34 minutes). *Fix:* cache the check behind a one-time module-level flag. Full-dataset preprocessing dropped to ~2.4 minutes (~14x). If preprocessing ever feels slow again, check for this pattern first (a check-every-call where a check-once would do) before assuming the NLP work itself is the bottleneck.

### Correctness

- **Train/inference preprocessing skew:** `TorchTextClassifier.predict_proba()` (used by the live API, `src/model.py`) called `preprocess_text()` but never `strip_leading_dateline()`, even though `train.py` strips it before training. Text submitted to `/predict` with a dateline would hit the model with a token distribution it never trained on. *Fix:* apply `strip_leading_dateline()` in `predict_proba()` too. **General rule:** any preprocessing step applied before training must be mirrored exactly at inference time, or the model sees a different input distribution than it learned on — this is why `save_model.ModelArtifact` is documented as owning its preprocessing end-to-end rather than leaving it to `predict.py`.
- **`graph_analysis._simulate_bot_cascade()` relied entirely on `add_edge()` to implicitly create its core nodes** — a too-small core (or, rarely, an edge-less one by chance) silently produced fewer nodes than requested (e.g. `n_nodes=1` produced a 0-node graph). `extract_node_features()` separately returned a malformed `(0,)` shape instead of `(0, 4)` for an empty graph. *Fix:* explicit `add_nodes_from()` for the core; shape-guarded the empty-graph case. Both were dormant under current usage (`n_nodes` always ≥15 in real call paths) but worth keeping fixed since `graph_to_data()`/`extract_node_features()` are the integration point for real propagation data later, where small/degenerate graphs are plausible.

### Process

- **Two Claude Code sessions running against the same repo directory at once raced `git checkout`/`merge`/`commit` commands against each other** on one shared working tree (git doesn't handle concurrent operations on a single working tree safely). One session's uncommitted edits got carried across an unrelated branch switch, and a `git merge` that should have done real work instead silently no-opped because the other session had already done it. *Fix (standing rule):* notify the user before starting work in any new session. If signs of another active session turn up mid-task (unexpected commits/branches, uncommitted changes you didn't make, running processes tied to the project), stop making further git changes and ask rather than guessing.

## What Claude Should NOT Do

- Don't silently accept perfect (1.0) metrics without flagging the leakage risk noted above.
- Don't hardcode file paths outside of `data/`, `models/`, and `src/`.
- Don't mix notebook exploration code into `src/` modules — notebooks are for exploration, `src/` is for production-ready code.
