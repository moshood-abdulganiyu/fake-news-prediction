"""
Train, save, and load the fake news classifier.

Pipeline: TfidfVectorizer + LinearSVC, fit on title+text content built
by preprocess.build_content(). Chosen in Step 4 CV comparison (98.31%
accuracy vs 93.85% for title-only, beating LogisticRegression and
RandomForest on every metric but a precision tie).

Usage:
    # one-time, from repo root, after Step 4's holdout split exists:
    python -m src.model

    # from the Streamlit app:
    from src.model import predict
    label, confidence = predict(headline, text)
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import joblib
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocess import build_content

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"
MODEL_PATH = MODELS_DIR / "model.joblib"

TRAIN_SPLIT_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "train_holdout_split_train.pkl"
)


CONTENT_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "content_cache.pkl"


def _get_content(train_df):
    """
    Get the built content column for train_df's rows, fast path first.

    Tries content_cache.pkl and slices it by train_df's index - no
    re-stemming, seconds not minutes. If the cache is missing, or its
    index doesn't cover every row in train_df (stale cache, or a plain
    array/list with no index), falls back to rebuilding from scratch via
    build_content(), which takes several minutes over ~33k rows. Either
    way this function runs once, ever - not on every predict() call.
    """
    if CONTENT_CACHE_PATH.exists():
        with open(CONTENT_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        if hasattr(cache, "index") and set(train_df.index).issubset(set(cache.index)):
            aligned = cache.loc[train_df.index]
            if hasattr(aligned, "columns"):
                # cache is a DataFrame, not a flat Series - pull out the
                # actual content string column instead of handing the
                # whole DataFrame to the vectorizer (which would iterate
                # column names, not row values).
                candidates = [c for c in ("content_title_text", "content", "text") if c in aligned.columns]
                if candidates:
                    content = aligned[candidates[0]]
                elif aligned.shape[1] == 1:
                    content = aligned.iloc[:, 0]
                else:
                    raise ValueError(
                        f"content_cache.pkl has columns {list(aligned.columns)} - "
                        "none match the expected names and there's more than one "
                        "column, so I can't tell which holds the built content "
                        "string. Add the right column name to the `candidates` "
                        "list in _get_content()."
                    )
            else:
                content = aligned
            print("Using content_cache.pkl (fast path, no re-stemming).")
            return content
        print("content_cache.pkl found but index doesn't match train_df - rebuilding.")
    else:
        print("No content_cache.pkl found - rebuilding.")

    print("Rebuilding content column via build_content() - this takes several minutes.")
    return train_df.apply(
        lambda row: build_content(row["title"], row["text"]), axis=1
    )


def train_and_save() -> None:
    """
    Fit TfidfVectorizer + LinearSVC on the full training split and save
    both to MODELS_DIR with joblib.

    Reads data/train_holdout_split_train.pkl (built in Step 4 with
    random_state=42). Does not touch the holdout split - that's Step 8.
    """
    with open(TRAIN_SPLIT_PATH, "rb") as f:
        train_df = pickle.load(f)

    content = _get_content(train_df)
    labels = train_df["label"]

    print(f"Fitting TfidfVectorizer on {len(content)} documents...")
    t0 = time.time()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(content)
    print(f"  done in {time.time() - t0:.1f}s")

    print("Fitting LinearSVC...")
    t0 = time.time()
    model = LinearSVC()
    model.fit(X, labels)
    print(f"  done in {time.time() - t0:.1f}s")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)

    print(f"Saved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved model to {MODEL_PATH}")


def _load():
    """Load the saved vectorizer and model. Raises if train_and_save() hasn't run yet."""
    if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model files not found. Run `python -m src.model` first to train and save."
        )
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model


def predict(headline: str, text: str) -> tuple[str, float]:
    """
    Predict fake/real for a single headline+text pair.

    Returns (label, confidence) where label is "Fake" or "Real" and
    confidence is a 0-1 float derived from the LinearSVC decision
    function distance, squashed through a sigmoid. LinearSVC has no
    predict_proba by default (that requires Platt scaling, which is
    expensive to fit), so this is an approximation, not a calibrated
    probability - good enough for a confidence bar in the UI, not for
    anything that needs to be statistically precise.
    """
    vectorizer, model = _load()

    content = build_content(headline, text)
    X = vectorizer.transform([content])

    pred = model.predict(X)[0]
    decision = model.decision_function(X)[0]
    confidence = 1 / (1 + pow(2.718281828, -abs(decision)))

    label = "Fake" if pred == 1 else "Real"
    return label, confidence


if __name__ == "__main__":
    train_and_save()