"""
NLP utilities: cleaning, preprocessing, sentiment, word counts,
prediction helpers, and chart builders (matplotlib + plotly).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import nltk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from wordcloud import WordCloud


def _ensure_nltk_resources() -> None:
    """Download required NLTK data if missing."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                if name != "punkt_tab":
                    raise


_ensure_nltk_resources()

_LEMMATIZER = WordNetLemmatizer()
_STOPWORDS = set(stopwords.words("english"))


def clean_text(text: Optional[str]) -> str:
    """
    Lowercase, remove URLs, punctuation, and English stopwords.
    Returns space-separated tokens (before lemmatization).
    """
    if text is None or not str(text).strip():
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t and t not in _STOPWORDS]
    return " ".join(tokens)


def preprocess_text(text: Optional[str]) -> str:
    """Full pipeline for model input: clean + tokenize + lemmatize."""
    cleaned = clean_text(text)
    if not cleaned:
        return ""

    try:
        tokens = word_tokenize(cleaned)
    except Exception:
        tokens = cleaned.split()

    lemmas = [_LEMMATIZER.lemmatize(t) for t in tokens if t]
    return " ".join(lemmas)


def predict_text(text: str, model: Any, vectorizer: Any) -> tuple[Optional[int], Optional[np.ndarray], str]:
    """
    Return (prediction 0/1, probability vector shape (2,), processed string).
    If input is empty after preprocessing, returns (None, None, processed).
    """
    processed = preprocess_text(text)
    if not processed.strip():
        return None, None, processed

    X = vectorizer.transform([processed])
    proba = model.predict_proba(X)[0]
    pred = int(model.predict(X)[0])
    return pred, proba, processed


def get_sentiment(text: Optional[str]) -> dict:
    """TextBlob polarity, subjectivity, and coarse label."""
    if text is None or not str(text).strip():
        return {"polarity": 0.0, "subjectivity": 0.0, "label": "Neutral"}

    blob = TextBlob(str(text))
    pol = float(blob.sentiment.polarity)
    subj = float(blob.sentiment.subjectivity)
    if pol > 0.1:
        lbl = "Positive"
    elif pol < -0.1:
        lbl = "Negative"
    else:
        lbl = "Neutral"
    return {"polarity": pol, "subjectivity": subj, "label": lbl}


def word_count(text: Optional[str]) -> int:
    """Word count on raw user text."""
    if text is None or not str(text).strip():
        return 0
    return len(str(text).split())


def get_top_keywords(vectorizer: Any, processed: str, n: int = 8) -> list[tuple[str, float]]:
    """Top TF-IDF weighted terms for a single preprocessed document."""
    if not processed.strip():
        return []
    X = vectorizer.transform([processed])
    scores = X.toarray().flatten()
    names = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(scores)[::-1]
    out: list[tuple[str, float]] = []
    for i in order:
        if scores[i] <= 0:
            break
        out.append((str(names[i]), float(scores[i])))
        if len(out) >= n:
            break
    return out


def text_statistics(raw: str, processed: str) -> dict:
    """Counts for charts and cards."""
    raw = raw or ""
    words = word_count(raw)
    chars = len(raw)
    sents = len([s for s in re.split(r"[.!?]+", raw) if s.strip()]) or (1 if raw.strip() else 0)
    avg_wlen = (sum(len(w) for w in raw.split()) / words) if words else 0.0
    proc_tokens = len(processed.split()) if processed else 0
    return {
        "words": words,
        "chars": chars,
        "sentences": max(1, sents),
        "avg_word_length": round(avg_wlen, 2),
        "processed_tokens": proc_tokens,
    }


def generate_wordcloud(text: str, dark: bool = False) -> plt.Figure:
    """Word cloud from user input text (raw). Uses wordcloud + matplotlib."""
    bg = "#1e293b" if dark else "#ffffff"
    colormap = "viridis" if dark else "Greens"
    wc = WordCloud(
        width=900,
        height=360,
        background_color=bg,
        colormap=colormap,
        prefer_horizontal=0.85,
        max_words=80,
    ).generate((text or "").strip() or " ")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    fig.patch.set_facecolor(bg)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.suptitle("Word cloud (input text)", fontsize=12, y=1.02, color="#94a3b8" if dark else "#475569")
    plt.tight_layout()
    return fig


def plot_sentiment_blob_detail(sent: dict, dark: bool = False) -> go.Figure:
    """Side-by-side bars: polarity (−1…1) vs subjectivity (0…1) — TextBlob detail."""
    color = "#38bdf8" if dark else "#0d9488"
    color2 = "#a78bfa" if dark else "#6366f1"
    tpl = "plotly_dark" if dark else "plotly_white"
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Polarity", "Subjectivity"))
    fig.add_trace(go.Bar(x=["Score"], y=[sent["polarity"]], marker_color=color, showlegend=False), row=1, col=1)
    fig.add_trace(
        go.Bar(x=["Score"], y=[sent["subjectivity"]], marker_color=color2, showlegend=False), row=1, col=2
    )
    fig.update_yaxes(range=[-1.05, 1.05], title_text="−1 … +1", row=1, col=1)
    fig.update_yaxes(range=[0, 1.05], title_text="0 … 1", row=1, col=2)
    fig.update_layout(height=320, margin=dict(t=48, b=36), title_text="Sentiment detail (TextBlob)", template=tpl)
    return fig


def plot_sentiment_chart(sent: dict, dark: bool = False) -> go.Figure:
    """
    Positive / Neutral / Negative as a bar chart (from TextBlob polarity).
    Single strongest category gets weight 1.0; others 0 for crisp display,
    with subtitle showing continuous polarity.
    """
    tpl = "plotly_dark" if dark else "plotly_white"
    pol = float(sent.get("polarity", 0.0))
    lbl = sent.get("label", "Neutral")
    if pol > 0.1:
        vals = [1.0, 0.0, 0.0]
        colors = ["#22c55e", "#94a3b8", "#f43f5e"]
    elif pol < -0.1:
        vals = [0.0, 0.0, 1.0]
        colors = ["#22c55e", "#94a3b8", "#f43f5e"]
    else:
        vals = [0.0, 1.0, 0.0]
        colors = ["#22c55e", "#64748b", "#f43f5e"]

    cats = ["Positive", "Neutral", "Negative"]
    fig = go.Figure(
        go.Bar(
            x=cats,
            y=vals,
            marker_color=colors,
            text=[f"{v:.0%}" if v > 0 else "" for v in vals],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Sentiment category (TextBlob · polarity {:.2f} · {})".format(pol, lbl),
        yaxis=dict(range=[0, 1.15], title="Strength"),
        height=320,
        template=tpl,
        showlegend=False,
    )
    return fig


def plot_sentiment_pie(sent: dict, dark: bool = False) -> go.Figure:
    """Pie chart: soft positive / neutral / negative shares derived from polarity."""
    tpl = "plotly_dark" if dark else "plotly_white"
    pol = float(sent.get("polarity", 0.0))
    pos = max(0.0, pol)
    neg = max(0.0, -pol)
    neu = max(0.0, 1.0 - abs(pol))
    s = pos + neu + neg
    if s <= 0:
        pos, neu, neg = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    else:
        pos, neu, neg = pos / s, neu / s, neg / s
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Positive", "Neutral", "Negative"],
                values=[pos, neu, neg],
                marker=dict(colors=["#22c55e", "#94a3b8", "#f43f5e"]),
                hole=0.35,
                sort=False,
                textinfo="percent+label",
            )
        ]
    )
    fig.update_layout(
        title="Sentiment composition (polarity → shares)",
        template=tpl,
        height=360,
        showlegend=True,
    )
    return fig


def plot_probability_chart(proba: np.ndarray, dark: bool = False) -> go.Figure:
    """Horizontal bar chart for class probabilities."""
    labels = ["Not depressed", "Depressed"]
    vals = [float(proba[0]), float(proba[1])]
    colors = ["#22c55e", "#f43f5e"] if not dark else ["#4ade80", "#fb7185"]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1%}" for v in vals],
            textposition="auto",
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(t=36, l=120),
        title="Prediction probabilities",
        template="plotly_dark" if dark else "plotly_white",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
    )
    return fig


def plot_word_frequency(input_text: str, top_n: int = 10, dark: bool = False) -> go.Figure:
    """Top N most frequent words in the raw input text (letters only)."""
    tpl = "plotly_dark" if dark else "plotly_white"
    tokens = re.findall(r"[a-zA-Z]+", (input_text or "").lower())
    if not tokens:
        fig = go.Figure()
        fig.add_annotation(text="No words in input", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=280, template=tpl, title="Word frequency (input text)")
        return fig

    counts = Counter(tokens)
    top = counts.most_common(top_n)
    words = [w for w, _ in top][::-1]
    freqs = [c for _, c in top][::-1]
    fig = go.Figure(
        go.Bar(
            x=freqs,
            y=words,
            orientation="h",
            marker_color="#0d9488" if not dark else "#2dd4bf",
        )
    )
    fig.update_layout(
        title=f"Top {top_n} word frequencies (input text)",
        height=max(300, 20 * top_n),
        margin=dict(l=100, t=48),
        template=tpl,
        xaxis_title="Count",
    )
    return fig


def plot_word_frequency_processed(processed: str, top_n: int = 15, dark: bool = False) -> go.Figure:
    """Most common tokens after preprocessing (lemmas) — for model diagnostics."""
    tokens = processed.split() if processed else []
    if not tokens:
        fig = go.Figure()
        fig.add_annotation(text="No tokens to plot", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=280, template="plotly_dark" if dark else "plotly_white")
        return fig

    counts = Counter(tokens)
    top = counts.most_common(top_n)
    words = [w for w, _ in top]
    freqs = [c for _, c in top]
    fig = go.Figure(
        go.Bar(x=freqs, y=words, orientation="h", marker_color="#0d9488" if not dark else "#2dd4bf")
    )
    fig.update_layout(
        title=f"Most common lemmas (top {top_n})",
        height=max(320, 18 * top_n),
        margin=dict(l=100, t=40),
        template="plotly_dark" if dark else "plotly_white",
        xaxis_title="Count",
    )
    return fig


def plot_text_statistics_chart(stats: dict, dark: bool = False) -> go.Figure:
    """Horizontal bar chart: words, characters, sentences, average word length."""
    keys = ["Words", "Characters", "Sentences", "Avg word length"]
    vals = [
        float(stats["words"]),
        float(stats["chars"]),
        float(stats["sentences"]),
        float(stats["avg_word_length"]),
    ]
    tpl = "plotly_dark" if dark else "plotly_white"
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=keys,
            orientation="h",
            marker_color="#6366f1" if not dark else "#818cf8",
            text=[str(int(v)) if k != "Avg word length" else f"{v:.2f}" for k, v in zip(keys, vals)],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Text statistics",
        height=300,
        template=tpl,
        xaxis_title="Value",
    )
    return fig


def plot_confidence_indicator(confidence: float, dark: bool = False) -> go.Figure:
    """Gauge-style chart for model confidence on the predicted class (0–100%)."""
    tpl = "plotly_dark" if dark else "plotly_white"
    pct = float(max(0.0, min(1.0, confidence))) * 100.0
    bar = "#0d9488" if not dark else "#2dd4bf"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(pct, 1),
            number={"suffix": "%", "font": {"size": 28}},
            title={"text": "Confidence (predicted class)"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": bar},
                "bgcolor": "rgba(100,100,100,0.15)" if dark else "rgba(200,200,200,0.35)",
                "steps": [
                    {"range": [0, 33], "color": "rgba(148,163,184,0.25)"},
                    {"range": [33, 66], "color": "rgba(251,191,36,0.2)"},
                    {"range": [66, 100], "color": "rgba(34,197,94,0.2)"},
                ],
                "threshold": {
                    "line": {"color": "white" if dark else "#334155", "width": 2},
                    "thickness": 0.8,
                    "value": pct,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(t=48, b=24), template=tpl)
    return fig


def plot_prediction_probability(proba: np.ndarray, dark: bool = False) -> go.Figure:
    """Bar chart: probability of Not depressed vs Depressed (same as model output)."""
    return plot_probability_chart(proba, dark)


def plot_text_statistics(stats: dict, dark: bool = False) -> go.Figure:
    """Alias for text statistics bar chart (words, chars, sentences, avg word length)."""
    return plot_text_statistics_chart(stats, dark)


def plot_top_tfidf_keywords(keywords: list[tuple[str, float]], dark: bool = False) -> go.Figure:
    """Horizontal bar chart of TF-IDF weights for the current document."""
    tpl = "plotly_dark" if dark else "plotly_white"
    if not keywords:
        fig = go.Figure()
        fig.add_annotation(text="No keywords", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(height=260, template=tpl)
        return fig
    terms = [k[0] for k in keywords][::-1]
    weights = [k[1] for k in keywords][::-1]
    fig = go.Figure(
        go.Bar(
            x=weights,
            y=terms,
            orientation="h",
            marker_color="#0d9488" if not dark else "#2dd4bf",
        )
    )
    fig.update_layout(
        title="Top TF-IDF weights (this post)",
        height=max(280, 22 * len(keywords)),
        margin=dict(l=120, t=40),
        xaxis_title="TF-IDF weight",
        template=tpl,
    )
    return fig


def plot_confusion_matrix_heatmap(cm: list | np.ndarray, dark: bool = False) -> go.Figure:
    """
    Interactive confusion matrix (test set, logistic regression).
    cm[i][j] = true class i, predicted class j (binary 0/1).
    """
    z = np.asarray(cm, dtype=float)
    text = [[str(int(z[i, j])) for j in range(z.shape[1])] for i in range(z.shape[0])]
    labels = ["Not depressed", "Depressed"]
    tpl = "plotly_dark" if dark else "plotly_white"
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[f"Predicted: {labels[j]}" for j in range(z.shape[1])],
            y=[f"Actual: {labels[i]}" for i in range(z.shape[0])],
            colorscale="Blues" if not dark else "Viridis",
            text=text,
            texttemplate="%{text}",
            textfont={"size": 16},
            showscale=True,
            colorbar=dict(title="Count"),
        )
    )
    fig.update_layout(
        title="Confusion matrix (hold-out test set · Logistic Regression)",
        height=360,
        margin=dict(t=50, l=160),
        template=tpl,
        xaxis=dict(side="bottom"),
    )
    return fig


def plot_model_metrics_comparison(metrics: dict, dark: bool = False) -> go.Figure:
    """Grouped bar: Logistic Regression vs Naive Bayes on accuracy, precision, recall, F1."""
    if not (metrics.get("logistic_regression") and metrics.get("naive_bayes")):
        fig = go.Figure()
        fig.add_annotation(
            text="Train models to compare",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(height=280, template="plotly_dark" if dark else "plotly_white")
        return fig

    lr = metrics["logistic_regression"]
    nb = metrics["naive_bayes"]
    cats = ["Accuracy", "Precision", "Recall", "F1"]
    lr_vals = [lr["accuracy"], lr["precision"], lr["recall"], lr["f1"]]
    nb_vals = [nb["accuracy"], nb["precision"], nb["recall"], nb["f1"]]
    tpl = "plotly_dark" if dark else "plotly_white"
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Logistic Regression",
            x=cats,
            y=lr_vals,
            marker_color="#0d9488" if not dark else "#2dd4bf",
            text=[f"{v:.1%}" for v in lr_vals],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Naive Bayes",
            x=cats,
            y=nb_vals,
            marker_color="#6366f1" if not dark else "#a78bfa",
            text=[f"{v:.1%}" for v in nb_vals],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Model comparison (test set)",
        barmode="group",
        height=400,
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        template=tpl,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_train_test_label_balance(metrics: dict, dark: bool = False) -> go.Figure:
    """Bar chart of class counts in train vs test split."""
    tr = metrics.get("train_label_counts") or {}
    te = metrics.get("test_label_counts") or {}
    if not tr or not te:
        fig = go.Figure()
        fig.add_annotation(text="Re-run training to record label counts", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(height=280, template="plotly_dark" if dark else "plotly_white")
        return fig

    x = ["Not depressed", "Depressed"]
    y_tr = [tr.get("not_depressed", 0), tr.get("depressed", 0)]
    y_te = [te.get("not_depressed", 0), te.get("depressed", 0)]
    tpl = "plotly_dark" if dark else "plotly_white"
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Training", x=x, y=y_tr, marker_color="#0ea5e9" if not dark else "#38bdf8"))
    fig.add_trace(go.Bar(name="Test (hold-out)", x=x, y=y_te, marker_color="#f97316" if not dark else "#fb923c"))
    fig.update_layout(
        title="Class balance: train vs test",
        barmode="group",
        height=360,
        yaxis_title="Number of posts",
        template=tpl,
    )
    return fig


def plot_session_prediction_distribution(history: list, dark: bool = False) -> go.Figure:
    """Counts of Depressed vs Not depressed in current browser session."""
    tpl = "plotly_dark" if dark else "plotly_white"
    if not history:
        fig = go.Figure()
        fig.add_annotation(
            text="Run Analyze to build session stats",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(height=300, template=tpl)
        return fig

    n_dep = sum(1 for h in history if h.get("label") == "Depressed")
    n_ok = len(history) - n_dep
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Not depressed", "Depressed"],
                values=[n_ok, n_dep],
                hole=0.45,
                marker_colors=["#22c55e", "#f43f5e"] if not dark else ["#4ade80", "#fb7185"],
            )
        ]
    )
    fig.update_layout(
        title=f"Session predictions (n={len(history)})",
        height=340,
        template=tpl,
        showlegend=True,
    )
    return fig
