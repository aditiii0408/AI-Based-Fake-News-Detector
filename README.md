# 🔍 TruthLens — Fake News Detector

## ⚡ Setup (One Time)

### Step 1 — Install Python packages
```bash
pip install -r requirements.txt
```

### Step 2 — Download spaCy language model
```bash
python -m spacy download en_core_web_sm
```

### Step 3 — Run the app
```bash
streamlit run app.py
```
Opens at **http://localhost:8501**

> 💡 First run downloads the BERT model (~17MB) automatically. After that it's cached locally.

---

## 🤖 AI Stack (No API Key Required)

| Model | Purpose |
|---|---|
| `mrm8488/bert-tiny-finetuned-fake-news-detection` | Fake/Real classification |
| `spaCy en_core_web_sm` | Named entity recognition, claim extraction |
| `NLTK VADER` | Sentiment analysis |
| Custom NLP rules | Clickbait, bias, source signal detection |

---

## ✨ Features

- 📊 **Credibility Score** — 0–100% weighted from 4 models
- ⚖️ **Bias Detection** — Left / Right / Neutral / Mixed
- 🚩 **Red Flags** — Clickbait, anonymous sources, unverified claims
- ✅ **Positive Signals** — Citations, official sources, data references
- 🔎 **Key Claims** — Extracted sentences with strong assertions
- 🏷️ **Named Entities** — People, orgs, locations, dates detected
- 📋 **AI Summary** — Extractive summary of the article
- 🥧 **Sentiment Chart** — Positive / Negative / Neutral breakdown
- 📈 **Score Breakdown Chart** — Contribution of each model
- 🌐 **Trusted News Links** — BBC, Reuters, AP, FactCheck, Snopes
- 🔍 **Topic Search Links** — Auto-generated fact-check searches
- 📄 **Export Report** — Download full analysis as .txt
- 🕐 **Session History** — Sidebar log of past analyses
- 🔗 **URL Scraping** — Paste a link to auto-fetch article text
