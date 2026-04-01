"""
Train depression detection models on the Reddit-style CSV dataset.
Saves model.pkl, vectorizer.pkl, and metrics.json for the Streamlit app.
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from utils import preprocess_text

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
DATA_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"


def load_dataset(path: Path) -> pd.DataFrame:
    """Load CSV; support columns named 'text' or 'post' plus 'label'."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Add dataset.csv (columns: text/post, label)."
        )
    df = pd.read_csv(path)
    # Normalize text column name
    text_col = None
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("text", "post", "content", "body"):
            text_col = c
            break
    if text_col is None:
        # assume first column is text
        text_col = df.columns[0]

    label_col = None
    for c in df.columns:
        if c.lower().strip() in ("label", "labels", "target", "class"):
            label_col = c
            break
    if label_col is None:
        label_col = df.columns[-1]

    out = pd.DataFrame(
        {
            "text": df[text_col].astype(str),
            "label": pd.to_numeric(df[label_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    out = out[out["text"].str.strip() != ""]
    return out.reset_index(drop=True)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    """Compute classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n=== {name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print("\nClassification report:\n", classification_report(y_true, y_pred, zero_division=0))
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def save_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    """Save confusion matrix figure for documentation / optional app use."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not depressed", "Depressed"],
        yticklabels=["Not depressed", "Depressed"],
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Logistic Regression — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> None:
    # NLTK data for consistency with utils
    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            continue

    print("Loading dataset...")
    df = load_dataset(DATA_PATH)
    print(f"Rows: {len(df)} | Label distribution:\n{df['label'].value_counts().sort_index()}")

    print("Preprocessing text...")
    df["processed"] = df["text"].apply(preprocess_text)
    df = df[df["processed"].str.len() > 0].reset_index(drop=True)

    X = df["processed"]
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # --- Logistic Regression (primary) ---
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    log_reg.fit(X_train_tfidf, y_train)
    y_pred_lr = log_reg.predict(X_test_tfidf)
    metrics_lr = evaluate(y_test, y_pred_lr, "Logistic Regression")

    # --- Naive Bayes (comparison) ---
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    y_pred_nb = nb.predict(X_test_tfidf)
    metrics_nb = evaluate(y_test, y_pred_nb, "Multinomial Naive Bayes")

    # Persist artifacts
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(log_reg, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    cm_arr = confusion_matrix(y_test, y_pred_lr)
    payload = {
        "logistic_regression": metrics_lr,
        "naive_bayes": metrics_nb,
        "primary_model": "logistic_regression",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        # For interactive Plotly heatmap in the app (rows=true, cols=predicted; sklearn order)
        "confusion_matrix_logreg": cm_arr.tolist(),
        "train_label_counts": {
            "not_depressed": int((y_train == 0).sum()),
            "depressed": int((y_train == 1).sum()),
        },
        "test_label_counts": {
            "not_depressed": int((y_test == 0).sum()),
            "depressed": int((y_test == 1).sum()),
        },
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = ASSETS_DIR / "confusion_matrix.png"
    save_confusion_matrix_plot(y_test, y_pred_lr, cm_path)

    print(f"\nSaved model → {MODEL_PATH}")
    print(f"Saved vectorizer → {VECTORIZER_PATH}")
    print(f"Saved metrics → {METRICS_PATH}")
    print(f"Saved confusion matrix plot → {cm_path}")


if __name__ == "__main__":
    main()
