"""
Advanced Streamlit dashboard: NLP depression detection (Reddit-style text).
TF-IDF + Logistic Regression; visualizations via Plotly and Matplotlib.

Paths are resolved relative to this file so the app runs locally and on Streamlit Community Cloud.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import streamlit as st

# NLTK data (needed on fresh installs and Streamlit Cloud workers)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

from utils import (
    clean_text,
    generate_wordcloud,
    get_sentiment,
    get_top_keywords,
    plot_confusion_matrix_heatmap,
    plot_model_metrics_comparison,
    plot_confidence_indicator,
    plot_prediction_probability,
    plot_sentiment_blob_detail,
    plot_sentiment_chart,
    plot_sentiment_pie,
    plot_session_prediction_distribution,
    plot_text_statistics,
    plot_text_statistics_chart,
    plot_top_tfidf_keywords,
    plot_train_test_label_balance,
    plot_word_frequency,
    plot_word_frequency_processed,
    predict_text,
    preprocess_text,
    text_statistics,
    word_count,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"
CM_IMAGE_PATH = BASE_DIR / "assets" / "confusion_matrix.png"

EXAMPLE_POST = (
    "I've been feeling really empty lately and barely have energy to get through the day. "
    "I used to enjoy things but nothing feels worth it now."
)


def inject_theme_css(dark: bool) -> None:
    """Light / dark dashboard theme with readable contrast on all Streamlit widgets."""
    if dark:
        bg = "#0f172a"
        card = "#1e293b"
        text = "#f1f5f9"
        text_secondary = "#cbd5e1"
        muted = "#94a3b8"
        border = "#334155"
        accent = "#2dd4bf"
        shadow = "0 4px 24px rgba(0,0,0,0.35)"
        input_bg = "#0f172a"
        input_text = "#f8fafc"
        alert_text = "#e2e8f0"
    else:
        bg = "#f1f5f9"
        card = "#ffffff"
        text = "#0f172a"
        text_secondary = "#334155"
        muted = "#475569"
        border = "#cbd5e1"
        accent = "#0d9488"
        shadow = "0 2px 12px rgba(15,23,42,0.08)"
        input_bg = "#ffffff"
        input_text = "#0f172a"
        alert_text = "#0f172a"

    st.markdown(
        f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&display=swap');

    html, body {{ font-family: 'DM Sans', sans-serif; }}

    /* App shell */
    .stApp {{
        background: {bg} !important;
        color: {text} !important;
    }}

    /* Main content: force body text color (fixes white-on-white in light mode) */
    section.main, section.main p, section.main span, section.main li, section.main label,
    section.main h1, section.main h2, section.main h3, section.main h4, section.main h5,
    .main .stMarkdown, .main .stMarkdown p, .main .stMarkdown span, .main .stMarkdown li,
    .main [data-testid="stMarkdownContainer"] p,
    .main [data-testid="stMarkdownContainer"] span {{
        color: {text} !important;
    }}

    [data-testid="stCaptionContainer"], .stCaption, section.main .stCaption {{
        color: {muted} !important;
    }}

    /* Metrics */
    [data-testid="stMetricValue"] {{
        color: {text} !important;
    }}
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{
        color: {muted} !important;
    }}

    /* Widget labels */
    [data-testid="stWidgetLabel"] p, label[data-testid="stWidgetLabel"] {{
        color: {text_secondary} !important;
    }}

    /* Text area & inputs */
    .stTextArea textarea, .stTextInput input {{
        color: {input_text} !important;
        background-color: {input_bg} !important;
        border: 1px solid {border} !important;
        caret-color: {input_text} !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab"] {{
        color: {muted} !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {text} !important;
    }}

    /* Alerts / info / success */
    div[data-testid="stAlert"] {{
        color: {alert_text} !important;
    }}
    div[data-testid="stAlert"] p, div[data-testid="stAlert"] span {{
        color: {alert_text} !important;
    }}

    /* Dataframe */
    [data-testid="stDataFrame"] {{
        color: {text} !important;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {card} !important;
        border-right: 1px solid {border};
    }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {{
        color: {text} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
        color: {muted} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {{
        color: {text} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {{
        color: {muted} !important;
    }}

    .main .block-container {{ padding-top: 0.5rem; max-width: 1400px; }}

    .dash-header {{
        background: {card};
        border: 1px solid {border};
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: {shadow};
    }}
    .dash-header h1 {{
        margin: 0 0 0.35rem 0;
        font-size: 1.65rem;
        font-weight: 700;
        color: {text} !important;
    }}
    .dash-header p {{ margin: 0; color: {muted} !important; font-size: 1rem; }}

    .metric-card {{
        background: {card};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 1rem;
        box-shadow: {shadow};
    }}

    .footer-box {{
        margin-top: 2rem;
        padding: 1.25rem;
        border-top: 1px solid {border};
        color: {muted} !important;
        font-size: 0.88rem;
        text-align: center;
    }}

    .warn-dep {{
        background: {"#450a0a" if dark else "#fef2f2"};
        border: 1px solid {"#f87171" if dark else "#fecaca"};
        color: {"#fecaca" if dark else "#991b1b"} !important;
        padding: 1rem 1.15rem;
        border-radius: 12px;
        margin: 0.75rem 0;
    }}
    .warn-dep strong {{
        color: {"#fecaca" if dark else "#991b1b"} !important;
    }}

    div[data-testid="stExpander"] details {{
        background: {card} !important;
        border: 1px solid {border} !important;
        border-radius: 12px !important;
    }}
    div[data-testid="stExpander"] summary {{
        color: {text} !important;
    }}
    div[data-testid="stExpander"] details p, div[data-testid="stExpander"] details span {{
        color: {text} !important;
    }}

    .stButton > button[kind="primary"] {{
        background: {accent} !important;
        border: none !important;
        font-weight: 600 !important;
        color: #ffffff !important;
    }}
    .stButton > button[kind="secondary"] {{
        color: {text} !important;
        border-color: {border} !important;
    }}

    /* Toggle / checkbox label */
    .stCheckbox label, [data-baseweb="checkbox"] + div {{
        color: {text} !important;
    }}
</style>
""",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    """
    Load trained classifier and vectorizer from the app directory (relative paths).
    Cached so Streamlit reruns do not reload pickles on every interaction.
    """
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None, None
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except (pickle.UnpicklingError, OSError, EOFError, AttributeError):
        return None, None


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def init_session() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "theme_dark" not in st.session_state:
        st.session_state.theme_dark = False
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None


def append_history(entry: dict) -> None:
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:40]


def build_report(
    raw: str,
    label: str,
    confidence: float,
    sent: dict,
    wc: int,
    keywords: list[tuple[str, float]],
) -> str:
    kw_lines = "\n".join(f"  - {k}: {v:.4f}" for k, v in keywords)
    return "\n".join(
        [
            "NLP Depression Detection — Report",
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "=== Input ===",
            raw[:3000],
            "",
            "=== Prediction ===",
            f"Class: {label}",
            f"Confidence: {confidence:.2%}",
            "",
            "=== Sentiment ===",
            f"Polarity: {sent['polarity']:.3f}",
            f"Subjectivity: {sent['subjectivity']:.3f}",
            f"Label: {sent['label']}",
            "",
            "=== Word count ===",
            str(wc),
            "",
            "=== Top TF-IDF terms (this post) ===",
            kw_lines or "(none)",
            "",
            "Disclaimer: Educational research only; not a medical diagnosis.",
        ]
    )


def render_sidebar(metrics: dict, model_ok: bool) -> None:
    with st.sidebar:
        st.markdown("### NLP Depression Detection")
        st.caption("Social media text · Reddit dataset labels")
        st.markdown("---")

        st.markdown("#### About")
        st.write(
            "This app scores short posts with a **TF-IDF** vectorizer and **logistic regression** "
            "trained on depressed vs. not-depressed Reddit-style text (**0** / **1**)."
        )

        with st.expander("Dataset", expanded=False):
            st.write(
                "CSV columns: `text` or `post` plus `label`. "
                "Replace **`dataset.csv`** and run **`python train_model.py`** to retrain."
            )

        st.markdown("#### Model performance")
        if metrics.get("logistic_regression"):
            m = metrics["logistic_regression"]
            st.metric("Test accuracy", f"{m['accuracy']:.1%}")
            st.caption(f"Precision {m['precision']:.2f} · Recall {m['recall']:.2f} · F1 {m['f1']:.2f}")
            if metrics.get("naive_bayes"):
                nb = metrics["naive_bayes"]
                st.caption(f"Naive Bayes (train compare): {nb['accuracy']:.1%} acc")
        else:
            st.info("Run training to create `metrics.json`.")

        st.markdown("---")
        st.markdown("#### Instructions")
        st.markdown(
            """
1. Install: `pip install -r requirements.txt`  
2. Train: `python train_model.py`  
3. Run: `streamlit run app.py`  
4. Paste text → **Analyze**
"""
        )

        st.markdown("---")
        st.markdown("#### Developer")
        st.caption("Course / research project · modular `utils.py` + `train_model.py`")

        st.markdown("---")
        st.caption("Tip: use the **Appearance** toggle at the top of the page for light/dark theme.")

        with st.expander("About project", expanded=False):
            st.write(
                "NLP pipeline: TF-IDF + logistic regression on Reddit-style labels (0/1). "
                "Not for clinical use."
            )
            st.write("**Stack:** scikit-learn, NLTK, TextBlob, Plotly, Matplotlib, WordCloud.")

        with st.expander("Session history", expanded=False):
            if not st.session_state.get("history"):
                st.caption("No analyses yet this session.")
            else:
                for h in st.session_state.history[:15]:
                    st.caption(f"**{h['label']}** · {h['confidence']:.0%} · {h['time']}")
                    st.text(h["snippet"][:120] + ("…" if len(h["snippet"]) > 120 else ""))


def main() -> None:
    # Must be the first Streamlit command in this script (required by Streamlit).
    st.set_page_config(
        page_title="Depression NLP Dashboard",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session()
    model, vectorizer = load_artifacts()
    metrics = load_metrics()
    model_ok = model is not None and vectorizer is not None

    # Theme toggle first — updates session state before CSS injection on the same run
    row_theme_l, row_theme_r = st.columns([4, 1])
    with row_theme_l:
        st.caption("Appearance")
    with row_theme_r:
        # Widget value is bound to st.session_state.theme_dark
        st.toggle(
            "Dark mode",
            help="Switch between light and dark theme",
            key="theme_dark",
        )

    inject_theme_css(st.session_state.theme_dark)
    dark = st.session_state.theme_dark

    render_sidebar(metrics, model_ok)

    st.markdown(
        f"""
<div class="dash-header">
  <h1>NLP-based depression signal · social text</h1>
  <p>Research dashboard — not for clinical diagnosis. If you are in crisis, contact local emergency services or a mental health professional.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    if not model_ok:
        st.error(
            "Could not load **`model.pkl`** and **`vectorizer.pkl`** from the app folder. "
            "If they are missing, run **`python train_model.py`** (requires **`dataset.csv`**). "
            "If they exist but fail to load, retrain or ensure they were built with this project version."
        )
        st.stop()

    # —— Input row ——
    with st.container():
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown("##### Input text")
            st.text_area(
                "post",
                height=200,
                label_visibility="collapsed",
                placeholder="Paste a Reddit-style or social media post here…",
                key="input_text",
            )
        with c_right:
            st.markdown("##### Actions")
            run = st.button("Analyze", type="primary", use_container_width=True)
            clr = st.button("Clear", use_container_width=True)
            ex = st.button("Load example", use_container_width=True)

    if clr:
        st.session_state.input_text = ""
        st.session_state.last_prediction = None
        st.rerun()

    if ex:
        st.session_state.input_text = EXAMPLE_POST
        st.rerun()

    user_text = st.session_state.input_text
    sent_live = get_sentiment(user_text)
    wc_live = word_count(user_text)
    stats_live = text_statistics(user_text, preprocess_text(user_text) if user_text else "")

    quick_cols = st.columns(4)
    with quick_cols[0]:
        st.metric("Words", wc_live)
    with quick_cols[1]:
        st.metric("Sentiment", sent_live["label"])
    with quick_cols[2]:
        st.metric("Polarity", f"{sent_live['polarity']:.2f}")
    with quick_cols[3]:
        st.metric("Characters", stats_live["chars"])

    if run:
        if not user_text or not str(user_text).strip():
            st.warning("Please enter some text before analyzing.")
        else:
            with st.spinner("Running model and building visualizations…"):
                pred, proba, processed = predict_text(user_text, model, vectorizer)

            if pred is None or proba is None:
                st.warning("Text is empty after preprocessing. Try a longer message.")
            else:
                confidence = float(proba[pred])
                label_str = "Depressed" if pred == 1 else "Not Depressed"
                sent = get_sentiment(user_text)
                keywords = get_top_keywords(vectorizer, processed, n=10)
                stats = text_statistics(user_text, processed)

                st.session_state.last_prediction = {
                    "pred": pred,
                    "proba": proba,
                    "processed": processed,
                    "label_str": label_str,
                    "confidence": confidence,
                    "sent": sent,
                    "raw": user_text,
                    "wc": word_count(user_text),
                    "keywords": keywords,
                    "stats": stats,
                }
                append_history(
                    {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "label": label_str,
                        "confidence": confidence,
                        "snippet": user_text[:160] + ("…" if len(user_text) > 160 else ""),
                        "polarity": sent["polarity"],
                    }
                )
                st.success("Analysis complete.")
                st.toast("Prediction updated", icon="✅")

    lp = st.session_state.last_prediction

    tab_pred, tab_text, tab_viz = st.tabs(["Prediction", "Text Analysis", "Visualizations"])

    # —— Prediction tab (results only — no charts here) ——
    with tab_pred:
        if not lp:
            st.info("Run **Analyze** to see prediction, confidence, sentiment, word count, and exports.")
        else:
            if lp["pred"] == 1:
                st.markdown(
                    '<div class="warn-dep"><strong>Warning:</strong> The model flags this sample as closer to '
                    "<strong>depressed</strong> patterns in the training data. This is not a diagnosis.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.success("**Not depressed** for this sample (relative to training distribution).")

            mcols = st.columns(4)
            with mcols[0]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Prediction", lp["label_str"])
                st.markdown("</div>", unsafe_allow_html=True)
            with mcols[1]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Confidence", f"{lp['confidence']:.1%}")
                st.markdown("</div>", unsafe_allow_html=True)
            with mcols[2]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Sentiment score", f"{lp['sent']['polarity']:.2f} ({lp['sent']['label']})")
                st.markdown("</div>", unsafe_allow_html=True)
            with mcols[3]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Word count", lp["wc"])
                st.markdown("</div>", unsafe_allow_html=True)

            try:
                st.progress(lp["confidence"], text="Confidence for predicted class")
            except TypeError:
                st.progress(lp["confidence"])

            if lp["keywords"]:
                st.markdown("##### Top keywords (TF-IDF) for this post")
                df_kw = pd.DataFrame(lp["keywords"], columns=["Term", "Weight"])
                st.dataframe(df_kw, use_container_width=True, hide_index=True)

            r_dl, r_dl2 = st.columns(2)
            with r_dl:
                st.download_button(
                    "Download full report (.txt)",
                    data=build_report(
                        lp["raw"],
                        lp["label_str"],
                        lp["confidence"],
                        lp["sent"],
                        lp["wc"],
                        lp["keywords"],
                    ),
                    file_name="depression_nlp_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with r_dl2:
                st.download_button(
                    "Download JSON summary",
                    data=json.dumps(
                        {
                            "prediction": lp["label_str"],
                            "confidence": lp["confidence"],
                            "polarity": lp["sent"]["polarity"],
                            "words": lp["wc"],
                        },
                        indent=2,
                    ),
                    file_name="summary.json",
                    mime="application/json",
                    use_container_width=True,
                )

            st.divider()
            st.subheader("Visual overview")
            st.caption("Quick charts for this prediction — see **Visualizations** for the full set (word cloud, frequencies, etc.).")

            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(
                    plot_prediction_probability(lp["proba"], dark=dark),
                    use_container_width=True,
                    key="pred_tab_plot_probability",
                )
            with g2:
                st.plotly_chart(
                    plot_sentiment_chart(lp["sent"], dark=dark),
                    use_container_width=True,
                    key="pred_tab_plot_sentiment_bar",
                )

            g3, g4 = st.columns(2)
            with g3:
                st.plotly_chart(
                    plot_confidence_indicator(lp["confidence"], dark=dark),
                    use_container_width=True,
                    key="pred_tab_plot_confidence_gauge",
                )
            with g4:
                st.plotly_chart(
                    plot_sentiment_pie(lp["sent"], dark=dark),
                    use_container_width=True,
                    key="pred_tab_plot_sentiment_pie",
                )

            if lp.get("keywords"):
                st.plotly_chart(
                    plot_top_tfidf_keywords(lp["keywords"], dark=dark),
                    use_container_width=True,
                    key="pred_tab_plot_tfidf",
                )

    # —— Text Analysis tab ——
    with tab_text:
        with st.expander("Preprocessing steps (how text is cleaned before the model)", expanded=True):
            st.markdown(
                """
**Pipeline**
1. Lowercase; remove URLs and punctuation  
2. Remove English stopwords  
3. Tokenize and **lemmatize** (NLTK)  
4. **TF-IDF** vectorization (same settings as training)
"""
            )
            if user_text and str(user_text).strip():
                st.markdown("**After cleaning (no lemma yet)**")
                st.code(clean_text(user_text)[:2500] or "(empty)", language=None)
                st.markdown("**Final processed string (model input)**")
                st.code(preprocess_text(user_text)[:2500] or "(empty)", language=None)
            else:
                st.caption("Type or load text above to preview intermediate steps.")

        stats_for_tab = lp["stats"] if lp else stats_live
        st.plotly_chart(
            plot_text_statistics_chart(stats_for_tab, dark=dark),
            use_container_width=True,
            key="plotly_text_stats_tab",
        )
        st.caption("Quick view (native Streamlit)")
        st.bar_chart(
            pd.DataFrame(
                {
                    "Value": [
                        stats_for_tab["words"],
                        stats_for_tab["chars"],
                        stats_for_tab["sentences"],
                        stats_for_tab["avg_word_length"],
                    ]
                },
                index=["Words", "Characters", "Sentences", "Avg word length"],
            )
        )

        sent_for_detail = lp["sent"] if lp else sent_live
        st.plotly_chart(
            plot_sentiment_blob_detail(sent_for_detail, dark=dark),
            use_container_width=True,
            key="plotly_sentiment_blob_detail_tab",
        )

    # —— Visualizations tab (only after a successful prediction) ——
    with tab_viz:
        if not lp:
            st.info("Run **Analyze** on some text first. Charts appear here after a successful prediction.")
        else:
            with st.container():
                st.markdown("### Post-prediction analytics")
                st.caption("Based on your last analyzed text.")

                st.subheader("1 · Prediction probabilities")
                st.plotly_chart(
                    plot_prediction_probability(lp["proba"], dark=dark),
                    use_container_width=True,
                    key="viz_plot_prediction_probability",
                )

                c_sent, c_pie = st.columns(2)
                with c_sent:
                    st.subheader("2 · Sentiment (Positive / Neutral / Negative)")
                    st.plotly_chart(
                        plot_sentiment_chart(lp["sent"], dark=dark),
                        use_container_width=True,
                        key="viz_plot_sentiment_categories",
                    )
                with c_pie:
                    st.subheader("Sentiment (pie)")
                    st.plotly_chart(
                        plot_sentiment_pie(lp["sent"], dark=dark),
                        use_container_width=True,
                        key="viz_plot_sentiment_pie",
                    )

                st.subheader("3 · Word cloud (input text)")
                _wc = generate_wordcloud(lp["raw"], dark=dark)
                st.pyplot(_wc, clear_figure=True)
                plt.close(_wc)

                st.subheader("4 · Top 10 word frequencies (input text)")
                st.plotly_chart(
                    plot_word_frequency(lp["raw"], top_n=10, dark=dark),
                    use_container_width=True,
                    key="viz_plot_word_frequency",
                )

                st.subheader("5 · Text statistics")
                st.plotly_chart(
                    plot_text_statistics(lp["stats"], dark=dark),
                    use_container_width=True,
                    key="viz_plot_text_statistics",
                )

            with st.expander("Training & model evaluation (dataset / test metrics)", expanded=False):
                st.caption("From **`python train_model.py`** — not per-post.")
                r_eval1, r_eval2 = st.columns(2)
                with r_eval1:
                    cm_data = metrics.get("confusion_matrix_logreg")
                    if cm_data:
                        st.plotly_chart(
                            plot_confusion_matrix_heatmap(cm_data, dark=dark),
                            use_container_width=True,
                            key="plotly_cm_heatmap_interactive",
                        )
                    else:
                        st.info("Re-run training to embed the confusion matrix in `metrics.json`.")
                with r_eval2:
                    st.plotly_chart(
                        plot_model_metrics_comparison(metrics, dark=dark),
                        use_container_width=True,
                        key="plotly_model_metrics_grouped",
                    )

                st.plotly_chart(
                    plot_train_test_label_balance(metrics, dark=dark),
                    use_container_width=True,
                    key="plotly_train_test_balance",
                )

                st.markdown("##### Session activity (this browser)")
                st.plotly_chart(
                    plot_session_prediction_distribution(st.session_state.history, dark=dark),
                    use_container_width=True,
                    key="plotly_session_pie",
                )

                if lp.get("keywords"):
                    st.plotly_chart(
                        plot_top_tfidf_keywords(lp["keywords"], dark=dark),
                        use_container_width=True,
                        key="plotly_tfidf_keywords_bar",
                    )

                proc = lp["processed"]
                st.markdown("##### Lemma frequencies (processed text)")
                if proc.strip():
                    st.plotly_chart(
                        plot_word_frequency_processed(proc, top_n=15, dark=dark),
                        use_container_width=True,
                        key="plotly_word_frequency_processed",
                    )

                st.markdown("##### Confusion matrix (PNG)")
                if CM_IMAGE_PATH.exists():
                    st.image(str(CM_IMAGE_PATH), use_container_width=True)
                else:
                    st.warning("Run **`python train_model.py`** to generate `assets/confusion_matrix.png`.")

        if metrics.get("logistic_regression"):
            st.divider()
            m = metrics["logistic_regression"]
            st.caption(
                f"Hold-out test metrics · Accuracy {m['accuracy']:.1%} · Precision {m['precision']:.3f} · "
                f"Recall {m['recall']:.3f} · F1 {m['f1']:.3f}"
            )

    st.markdown(
        f"""
<div class="footer-box">
  NLP depression detection · educational use · TF-IDF + logistic regression · {datetime.now().year}
</div>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
