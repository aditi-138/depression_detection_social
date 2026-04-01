# NLP-Based Detection of Depression Using Social Media Text Analysis (Reddit Dataset)

A **Streamlit** web application that classifies short **social-media-style** posts (Reddit-like text) as **depressed (1)** or **not depressed (0)** using **TF–IDF** features and **logistic regression**, with **TextBlob** sentiment, rich **visual analytics**, and optional training-time comparison with **Naive Bayes**.

> **Disclaimer:** This project is for **education and research** only. It does **not** provide medical advice or a clinical diagnosis. If you or someone you know is in crisis, contact local emergency services or a qualified mental health professional.

---

## Project description

The app ingests user text, applies the same **preprocessing pipeline** as training (cleaning, stopword removal, lemmatization via NLTK), vectorizes with the saved **TF–IDF** vectorizer, and outputs a **binary prediction** with **confidence**, **sentiment** scores, **word statistics**, and **interactive charts** (Plotly, Matplotlib, WordCloud).

The machine learning **training logic** lives in `train_model.py` and is unchanged by deployment setup; this repository is configured to run **locally** and on **Streamlit Community Cloud** using **paths relative to `app.py`**.

---

## Features

- **Binary classification:** Depressed vs not depressed with **confidence** (predicted-class probability).
- **NLP preprocessing:** Cleaning, stopwords, lemmatization (aligned with `train_model.py`).
- **Sentiment analysis:** TextBlob polarity, subjectivity, and category-style views.
- **Dashboard UI:** Tabs for **Prediction**, **Text Analysis**, and **Visualizations**; dark/light theme.
- **Visual analytics:** Prediction probabilities, sentiment bar/pie/gauge, word cloud, word frequencies, text statistics, TF–IDF keyword bars, optional training metrics (confusion matrix heatmap, model comparison, train/test balance, session history).
- **Exports:** Downloadable text report and JSON summary.
- **Artifacts:** Loads **`model.pkl`** and **`vectorizer.pkl`** from the project root via `pickle` (with error handling if files are missing or invalid).

---

## Dataset used

**Reddit-style depression dataset** (CSV): posts labeled **0** (not depressed) or **1** (depressed).  
The training script accepts a column named **`text`**, **`post`**, **`content`**, or **`body`**, plus a **`label`** (or similar) column. Replace **`dataset.csv`** with your dataset and run **`python train_model.py`** to regenerate artifacts.

---

## Technologies used

| Area | Stack |
|------|--------|
| UI | [Streamlit](https://streamlit.io/) |
| ML | [scikit-learn](https://scikit-learn.org/) (TF–IDF, Logistic Regression, MultinomialNB) |
| NLP | [NLTK](https://www.nltk.org/), [TextBlob](https://textblob.readthedocs.io/) |
| Data | [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| Visualization | [Plotly](https://plotly.com/python/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [WordCloud](https://github.com/amueller/word_cloud) |
| Serialization | `pickle` (via `pickle-mixin` listed for compatibility) |

---

## Screenshots

_Add screenshots here after deployment or local run, for example:_

1. Main input area and **Analyze** button  
2. **Prediction** tab with metrics and quick charts  
3. **Visualizations** tab (word cloud, frequencies)  
4. Dark mode (optional)

---

## How to run locally

### 1. Clone or copy the project

```bash
cd depression_detection_streamlit
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Optional (if TextBlob corpora warnings appear):

```bash
python -m textblob.download_corpora
```

### 4. Train the model (if `model.pkl` / `vectorizer.pkl` are missing)

```bash
python train_model.py
```

This writes **`model.pkl`**, **`vectorizer.pkl`**, **`metrics.json`**, and **`assets/confusion_matrix.png`** (paths relative to the project folder).

### 5. Launch the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (often `http://localhost:8501`).

---

## Deployment instructions

### GitHub

1. Ensure **`model.pkl`**, **`vectorizer.pkl`**, and (optionally) **`metrics.json`** and **`assets/`** are committed if you want the Cloud app to run **without** training on the server.  
2. Do **not** commit secrets (`.env` is gitignored).

**Example commands** (replace the remote URL with your repository):

```bash
cd depression_detection_streamlit
git init
git add .
git commit -m "Final Depression Detection Streamlit App"
git branch -M main
git remote add origin https://github.com/yourusername/repository-name.git
git push -u origin main
```

### Streamlit Community Cloud (step by step)

1. Sign in at [https://share.streamlit.io/](https://share.streamlit.io/) with your GitHub account.  
2. **New app** → **Deploy an app**.  
3. **Repository:** select the GitHub repo containing this project.  
4. **Branch:** usually **`main`**.  
5. **Main file path:** **`app.py`**.  
6. **Deploy** and wait for the build to finish (`requirements.txt` is installed automatically).  
7. Your app URL will look like:  
   **`https://<your-app-subdomain>.streamlit.app`**  
   (Streamlit shows the exact link on the app page.)

**Notes**

- NLTK resources are downloaded at app startup in `app.py` (punkt, punkt_tab, stopwords, wordnet).  
- The app resolves **`model.pkl`** and **`vectorizer.pkl`** next to `app.py`, which matches Streamlit Cloud’s working directory for the repo root.  
- If the app fails to load artifacts, retrain locally, commit the new pickles, and redeploy.

---

## Project structure

```
depression_detection_streamlit/
│
├── app.py                 # Streamlit UI, NLTK bootstrap, model loading, dashboard
├── train_model.py         # Training pipeline (unchanged deployment logic)
├── utils.py               # Preprocessing, prediction helpers, chart builders
├── model.pkl              # Trained classifier (generate via train_model.py)
├── vectorizer.pkl         # Fitted TF-IDF vectorizer
├── dataset.csv            # Training data (text/post + label)
├── metrics.json           # Optional; metrics from last training run
├── requirements.txt       # Python dependencies for pip / Streamlit Cloud
├── README.md              # This file
├── .gitignore             # Ignores cache, env files, OS junk
└── assets/
    └── confusion_matrix.png   # Written by train_model.py (optional for UI)
```

---

## Future improvements

- Probability **calibration** and tunable decision thresholds  
- **Explainability** (e.g. LIME/SHAP) for top contributing terms  
- **Privacy-preserving** logging and optional user accounts (with ethics review)  
- **Multilingual** or domain-adapted models  
- **REST API** (e.g. FastAPI) alongside the Streamlit UI  

---

## Author

**[Your Name]** — *[Institution or affiliation, optional]*  
Update this section with your name and contact links (GitHub, LinkedIn) as appropriate.

---

## Deployment checklist

- [ ] `requirements.txt` lists all required packages.  
- [ ] `model.pkl` and `vectorizer.pkl` exist and are committed (or CI builds them—this repo assumes committed artifacts for Cloud).  
- [ ] `dataset.csv` present for local retraining (optional on Cloud if pickles are committed).  
- [ ] `app.py` uses paths relative to the script (`Path(__file__).resolve().parent`).  
- [ ] NLTK downloads run without blocking `st.set_page_config` (first `st` call remains `set_page_config` in `main()`).  
- [ ] `.gitignore` excludes `__pycache__`, `.env`, `.DS_Store`.  
- [ ] README updated with your GitHub username and screenshots.  
- [ ] Tested: `pip install -r requirements.txt` and `streamlit run app.py` locally.  
- [ ] Streamlit Cloud: branch **main**, entrypoint **`app.py`**, redeploy after pushing changes.

---

## License / ethics

Use this software responsibly. Do not present outputs as medical diagnoses. Dataset sources and user text must comply with applicable terms of use and privacy rules.
