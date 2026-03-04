import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TruthLens — Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# LAZY IMPORTS (cached so they load once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load all AI models once and cache them."""
    import nltk
    import spacy
    from transformers import pipeline

    # Download NLTK data
    for pkg in ["vader_lexicon", "punkt", "stopwords", "averaged_perceptron_tagger"]:
        try:
            nltk.download(pkg, quiet=True)
        except:
            pass

    # Load spaCy
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], capture_output=True)
        nlp = spacy.load("en_core_web_sm")

    # Load HuggingFace fake news classifier
    try:
        classifier = pipeline(
            "text-classification",
            model="mrm8488/bert-tiny-finetuned-fake-news-detection",
            truncation=True,
            max_length=512
        )
    except:
        classifier = None

    # Load sentiment analyzer
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    return nlp, classifier, sia


# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }

.stApp { background: #0d0d14; color: #e8e8f0; }

section[data-testid="stSidebar"] {
    background: #111118 !important;
    border-right: 1px solid #1e1e2e;
}

header[data-testid="stHeader"] { background: transparent !important; }

h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.5px; }

.card {
    background: #16161f;
    border: 1px solid #1e1e2e;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 18px;
}
.card-green { border-left: 3px solid #00e5a0; }
.card-red   { border-left: 3px solid #ff4d6d; }
.card-yellow{ border-left: 3px solid #ffd166; }

.verdict-real  { background:rgba(0,229,160,0.12);color:#00e5a0;border:1px solid rgba(0,229,160,0.3);border-radius:30px;padding:6px 20px;font-size:14px;font-weight:700;display:inline-block; }
.verdict-fake  { background:rgba(255,77,109,0.12);color:#ff4d6d;border:1px solid rgba(255,77,109,0.3);border-radius:30px;padding:6px 20px;font-size:14px;font-weight:700;display:inline-block; }
.verdict-mixed { background:rgba(255,209,102,0.12);color:#ffd166;border:1px solid rgba(255,209,102,0.3);border-radius:30px;padding:6px 20px;font-size:14px;font-weight:700;display:inline-block; }

.red-flag   { background:rgba(255,77,109,0.08);border-left:3px solid #ff4d6d;border-radius:8px;padding:10px 14px;margin:6px 0;font-size:13px;color:#ffb3bf; }
.green-flag { background:rgba(0,229,160,0.08);border-left:3px solid #00e5a0;border-radius:8px;padding:10px 14px;margin:6px 0;font-size:13px;color:#b3ffe0; }
.claim-pill { background:rgba(124,58,237,0.12);border:1px solid rgba(124,58,237,0.3);border-radius:8px;padding:8px 14px;margin:5px 0;font-size:13px;color:#c4b5fd; }

.metric-box   { background:#111118;border:1px solid #1e1e2e;border-radius:10px;padding:16px;text-align:center; }
.metric-value { font-family:'Syne',sans-serif;font-size:28px;font-weight:800; }
.metric-label { font-size:11px;color:#6b6b80;margin-top:4px;letter-spacing:1px;text-transform:uppercase; }

.entity-tag { display:inline-block;background:#1e1e2e;border-radius:6px;padding:3px 10px;font-size:11px;color:#9090c0;margin:2px;letter-spacing:0.5px; }

.news-link { background:#16161f;border:1px solid #1e1e2e;border-radius:10px;padding:14px 18px;margin:7px 0;display:block;text-decoration:none; }

.history-item { background:#16161f;border:1px solid #1e1e2e;border-radius:10px;padding:12px 16px;margin:6px 0;font-size:12px; }

.stButton > button {
    background: #00e5a0 !important; color: #0a0a0f !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 15px !important; padding: 14px 36px !important; width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stTextArea textarea {
    background: #16161f !important; border: 1px solid #1e1e2e !important;
    color: #e8e8f0 !important; border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important; font-size: 13px !important;
}
.stTextInput input {
    background: #16161f !important; border: 1px solid #1e1e2e !important;
    color: #e8e8f0 !important; border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important; font-size: 13px !important;
}

[data-testid="stTab"] { font-family: 'DM Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "history"  not in st.session_state: st.session_state.history  = []
if "analysis" not in st.session_state: st.session_state.analysis = None

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def fetch_url(url: str) -> str:
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script","style","nav","footer","header"]): t.decompose()
        return " ".join(p.get_text() for p in soup.find_all("p"))[:6000]
    except:
        return ""

def score_color(s):
    return "#00e5a0" if s >= 70 else "#ffd166" if s >= 40 else "#ff4d6d"

def verdict_html(s):
    if s >= 70: return '<span class="verdict-real">✅ LIKELY CREDIBLE</span>'
    if s >= 40: return '<span class="verdict-mixed">⚠️ MIXED / UNVERIFIED</span>'
    return '<span class="verdict-fake">🚨 LIKELY FAKE / MISLEADING</span>'

# Clickbait & sensationalist patterns
CLICKBAIT_PATTERNS = [
    r"\b(you won'?t believe|shocking|mind.?blowing|jaw.?dropping|unbelievable|insane)\b",
    r"\b(this is why|here'?s why|the truth about|they don'?t want you to know)\b",
    r"\b(secret|exposed|revealed|what they'?re hiding|cover.?up)\b",
    r"\b(miracle|cure|breakthrough|revolutionary|game.?changer)\b",
    r"\b(BREAKING|URGENT|EXCLUSIVE|BOMBSHELL|SHOCKING)\b",
    r"[!]{2,}",
    r"\b(destroy|obliterate|demolish|annihilate|crushing)\b",
    r"\b(everyone is|nobody is|all experts|scientists say)\b",
]

BIAS_LEFT_WORDS  = ["progressive","social justice","inequality","systemic","marginalized","oppression","equity","defund","woke","diversity"]
BIAS_RIGHT_WORDS = ["traditional","patriot","conservative","socialism","communist","radical left","deep state","fake news","make america","freedom fighters"]
EMOTIONAL_WORDS  = ["terrifying","horrifying","outrageous","disgusting","shameful","devastating","catastrophic","alarming","disturbing","appalling","heartbreaking","explosive"]

def detect_clickbait(text: str) -> int:
    text_lower = text.lower()
    hits = sum(1 for p in CLICKBAIT_PATTERNS if re.search(p, text_lower, re.IGNORECASE))
    return min(100, hits * 15)

def detect_bias(text: str) -> tuple:
    t = text.lower()
    l = sum(1 for w in BIAS_LEFT_WORDS  if w in t)
    r = sum(1 for w in BIAS_RIGHT_WORDS if w in t)
    if l == 0 and r == 0: return "NEUTRAL", "#00e5a0"
    if l > r * 1.5:        return "LEFT-LEANING", "#60a5fa"
    if r > l * 1.5:        return "RIGHT-LEANING", "#f87171"
    return "MIXED BIAS", "#ffd166"

def emotional_score(text: str) -> int:
    t = text.lower()
    hits = sum(1 for w in EMOTIONAL_WORDS if w in t)
    return min(100, hits * 12)

def extract_claims(text: str, nlp) -> list:
    doc = nlp(text[:3000])
    claims = []
    for sent in doc.sents:
        s = sent.text.strip()
        if len(s) > 40 and any(tok.pos_ == "VERB" for tok in sent):
            claims.append(s)
        if len(claims) >= 5:
            break
    return claims

def extract_entities(text: str, nlp) -> dict:
    doc = nlp(text[:3000])
    entities = {}
    for ent in doc.ents:
        label = ent.label_
        if label in ["PERSON","ORG","GPE","DATE","EVENT","NORP"]:
            entities.setdefault(label, [])
            if ent.text not in entities[label]:
                entities[label].append(ent.text)
    return {k: v[:6] for k, v in entities.items()}

def source_signals(text: str) -> tuple:
    positive, negative = [], []
    t = text.lower()

    if re.search(r'according to|said|stated|confirmed|reported', t):
        positive.append("Uses attributed quotes or sources")
    if re.search(r'study|research|report|survey|data', t):
        positive.append("References studies or data")
    if re.search(r'university|institute|government|official|ministry', t):
        positive.append("Cites official institutions")
    if re.search(r'\d{4}|\btoday\b|\byesterday\b|\blast week\b', t):
        positive.append("Contains specific dates/timeframes")
    if len(text.split()) > 300:
        positive.append("Substantial article length (>300 words)")

    if re.search(r'anonymous|unnamed source|insider|source close to', t):
        negative.append("Relies on anonymous sources")
    if re.search(r'apparently|allegedly|rumor|unconfirmed|could be|might be', t):
        negative.append("Contains unconfirmed claims")
    if detect_clickbait(text) > 40:
        negative.append("High clickbait language detected")
    if emotional_score(text) > 50:
        negative.append("Excessive emotional/charged language")
    if len(text.split()) < 100:
        negative.append("Very short article — lacks depth")
    if not re.search(r'according to|said|stated|confirmed', t):
        negative.append("No attributed sources found")

    return positive, negative

def summarize_text(text: str, nlp) -> str:
    """Extractive summary — pick top 3 most information-dense sentences."""
    doc = nlp(text[:4000])
    sents = [s.text.strip() for s in doc.sents if len(s.text.split()) > 10]
    if not sents:
        return text[:300]
    # Score by named entity + verb density
    def sent_score(s):
        d = nlp(s)
        return len(d.ents) + sum(1 for t in d if t.pos_ == "VERB")
    ranked = sorted(sents, key=sent_score, reverse=True)
    return " ".join(ranked[:3])

def analyze_article(text: str, nlp, classifier, sia) -> dict:
    """Run full analysis pipeline."""
    # 1. HuggingFace model
    hf_score = 50  # default
    hf_label = "UNKNOWN"
    if classifier:
        try:
            chunk = text[:512]
            res = classifier(chunk)[0]
            hf_label = res["label"].upper()
            hf_conf  = res["score"]
            # Model labels: LABEL_0=fake, LABEL_1=real (or similar)
            if "fake" in hf_label.lower() or hf_label == "LABEL_0":
                hf_score = int((1 - hf_conf) * 100)
            else:
                hf_score = int(hf_conf * 100)
        except:
            hf_score = 50

    # 2. NLP signals
    pos_signals, neg_flags = source_signals(text)
    signal_score = min(100, len(pos_signals) * 18) - min(60, len(neg_flags) * 12)
    signal_score = max(0, min(100, signal_score + 40))

    # 3. Sentiment
    sentiment = sia.polarity_scores(text)
    compound  = sentiment["compound"]
    emotional = emotional_score(text)

    # 4. Clickbait
    clickbait = detect_clickbait(text)

    # 5. Bias
    bias_type, bias_color = detect_bias(text)

    # 6. Combined credibility score (weighted)
    credibility = int(
        hf_score    * 0.50 +
        signal_score* 0.30 +
        (100 - clickbait) * 0.10 +
        (100 - emotional) * 0.10
    )
    credibility = max(0, min(100, credibility))

    # 7. Verdict
    if credibility >= 70:   verdict = "CREDIBLE"
    elif credibility >= 40: verdict = "MIXED / UNVERIFIED"
    else:                   verdict = "LIKELY FAKE"

    # 8. Key claims & entities & summary
    claims   = extract_claims(text, nlp)
    entities = extract_entities(text, nlp)
    summary  = summarize_text(text, nlp)

    # 9. Topics for news links
    topics = []
    if entities.get("ORG"):   topics += entities["ORG"][:2]
    if entities.get("GPE"):   topics += entities["GPE"][:1]
    if entities.get("EVENT"): topics += entities["EVENT"][:1]
    if not topics:
        words = [w for w in text.split() if len(w) > 5][:3]
        topics = words

    # 10. Student tip
    if credibility < 40:
        tip = "🚨 This article shows multiple fake news indicators. Always verify with at least 2 trusted sources before sharing."
    elif credibility < 70:
        tip = "⚠️ This article has mixed signals. Cross-check the key claims on FactCheck.org or Reuters before drawing conclusions."
    else:
        tip = "✅ This article appears credible, but always check the original source and publication date before sharing."

    return {
        "credibility_score": credibility,
        "verdict": verdict,
        "hf_score": hf_score,
        "signal_score": signal_score,
        "summary": summary,
        "key_claims": claims,
        "entities": entities,
        "red_flags": neg_flags,
        "positive_signals": pos_signals,
        "bias_type": bias_type,
        "bias_color": bias_color,
        "emotional_score": emotional,
        "clickbait_score": clickbait,
        "sentiment": sentiment,
        "topics": topics[:3],
        "student_tip": tip,
        "word_count": len(text.split()),
    }

def get_news_links(topics: list) -> list:
    base = [
        {"name": "BBC News",        "url": "https://www.bbc.com/news",           "icon": "🇬🇧", "tag": "outlet"},
        {"name": "Reuters",         "url": "https://www.reuters.com",             "icon": "📡", "tag": "outlet"},
        {"name": "AP News",         "url": "https://apnews.com",                  "icon": "📰", "tag": "outlet"},
        {"name": "The Guardian",    "url": "https://www.theguardian.com",         "icon": "🌍", "tag": "outlet"},
        {"name": "NPR",             "url": "https://www.npr.org/sections/news/",  "icon": "📻", "tag": "outlet"},
        {"name": "Al Jazeera",      "url": "https://www.aljazeera.com",           "icon": "🌐", "tag": "outlet"},
        {"name": "FactCheck.org",   "url": "https://www.factcheck.org",           "icon": "🔍", "tag": "fact-check"},
        {"name": "Snopes",          "url": "https://www.snopes.com",              "icon": "🧐", "tag": "fact-check"},
        {"name": "PolitiFact",      "url": "https://www.politifact.com",          "icon": "⚖️", "tag": "fact-check"},
        {"name": "Full Fact",       "url": "https://fullfact.org",                "icon": "✅", "tag": "fact-check"},
    ]
    for t in topics[:3]:
        enc = t.replace(" ", "+")
        base.append({"name": f'Search "{t}" on Reuters',      "url": f"https://www.reuters.com/search/news?blob={enc}",  "icon": "🔎", "tag": "search"})
        base.append({"name": f'Fact-check "{t}" on Snopes',   "url": f"https://www.snopes.com/?s={enc}",                 "icon": "✅", "tag": "search"})
    return base

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 TruthLens")
    st.markdown('<p style="color:#6b6b80;font-size:12px;">AI-Powered Fake News Detector for Students</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📖 How to Use")
    st.markdown("""
<div style="font-size:12px;color:#6b6b80;line-height:1.9;">
1. Paste news text OR enter a URL<br>
2. Click <b style="color:#00e5a0">Analyze</b><br>
3. Review the full credibility report<br>
4. Cross-check with trusted sources
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🤖 AI Models Used")
    st.markdown("""
<div style="font-size:11px;color:#6b6b80;line-height:1.9;">
🧠 <b style="color:#c4b5fd;">BERT</b> — Fake news classifier<br>
🔤 <b style="color:#c4b5fd;">spaCy</b> — Entity & claim extraction<br>
💬 <b style="color:#c4b5fd;">VADER</b> — Sentiment analysis<br>
📊 <b style="color:#c4b5fd;">NLP Rules</b> — Bias & clickbait detection
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🕐 History")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-8:]):
            c = score_color(item["score"])
            st.markdown(f"""
<div class="history-item">
  <span style="color:{c};font-weight:700;">{item['score']}%</span>
  &nbsp;·&nbsp;<span style="color:#6b6b80;">{item['time']}</span><br>
  <span style="font-size:11px;color:#9090a0;">{item['preview']}</span>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#3a3a4a;font-size:12px;">No analyses yet.</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="color:#2a2a3a;font-size:10px;text-align:center;">No API key needed · 100% Local AI</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding:20px 0 10px;">
  <h1 style="font-size:44px;font-weight:800;letter-spacing:-1px;margin-bottom:8px;">
    Truth<span style="color:#00e5a0;">Lens</span>
  </h1>
  <p style="color:#6b6b80;font-size:14px;max-width:540px;line-height:1.7;">
    Paste any news article or URL. Our local AI stack — BERT + spaCy + VADER — 
    analyses credibility, bias, emotional language, and links you to verified sources.
    <b style="color:#4a4a5a;">No API key. No internet required after setup.</b>
  </p>
</div>
<hr style="border:none;border-top:1px solid #1e1e2e;margin:16px 0 28px;">
""", unsafe_allow_html=True)

# Load models
with st.spinner("⚙️ Loading AI models (first run may take ~30 seconds)..."):
    nlp, classifier, sia = load_models()

# Input tabs
tab1, tab2 = st.tabs(["📝 Paste Article", "🔗 Enter URL"])
article_text = ""

with tab1:
    pasted = st.text_area("Paste article text", height=200,
        placeholder="Paste the full text of any news article, social media post, or claim you want to fact-check...",
        label_visibility="collapsed")
    article_text = pasted

with tab2:
    url_in = st.text_input("Article URL", placeholder="https://example.com/news-article", label_visibility="collapsed")
    if url_in:
        with st.spinner("🌐 Fetching article..."):
            fetched = fetch_url(url_in)
        if fetched:
            st.success(f"✓ Fetched {len(fetched.split())} words")
            article_text = fetched
            with st.expander("Preview"):
                st.text(fetched[:400] + "...")
        else:
            st.error("Could not fetch URL. Try pasting the text instead.")

st.markdown("<br>", unsafe_allow_html=True)
run = st.button("🔍 Analyze Article", use_container_width=True)

# ─────────────────────────────────────────────
# RUN ANALYSIS
# ─────────────────────────────────────────────
if run:
    if not article_text.strip():
        st.warning("⚠️ Please paste some text or enter a URL.")
    elif len(article_text.split()) < 15:
        st.warning("⚠️ Text too short — please provide at least 15 words.")
    else:
        with st.spinner("🤖 Analyzing with BERT + spaCy + VADER..."):
            result = analyze_article(article_text, nlp, classifier, sia)
            st.session_state.analysis = result
            st.session_state.history.append({
                "score":   result["credibility_score"],
                "time":    datetime.now().strftime("%H:%M"),
                "preview": article_text[:55].strip() + "..."
            })

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
if st.session_state.analysis:
    r     = st.session_state.analysis
    score = r["credibility_score"]
    col   = score_color(score)

    st.markdown("<hr style='border:none;border-top:1px solid #1e1e2e;margin:28px 0;'>", unsafe_allow_html=True)
    st.markdown("## 📊 Analysis Results")

    # ── Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Credibility Score", "font": {"size": 16, "color": "#6b6b80"}},
        number={"font": {"size": 52, "color": col, "family": "Syne"}, "suffix": "%"},
        gauge={
            "axis":  {"range": [0, 100], "tickcolor": "#2a2a3a", "tickfont": {"color": "#6b6b80"}},
            "bar":   {"color": col, "thickness": 0.25},
            "bgcolor": "#16161f",
            "bordercolor": "#1e1e2e",
            "steps": [
                {"range": [0,  40], "color": "rgba(255,77,109,0.15)"},
                {"range": [40, 70], "color": "rgba(255,209,102,0.15)"},
                {"range": [70,100], "color": "rgba(0,229,160,0.15)"},
            ],
            "threshold": {"line": {"color": col, "width": 3}, "thickness": 0.8, "value": score},
        }
    ))
    fig.update_layout(
        paper_bgcolor="#0d0d14", plot_bgcolor="#0d0d14",
        font={"color": "#e8e8f0"}, height=280, margin=dict(t=40, b=10, l=30, r=30)
    )

    c1, c2 = st.columns([1, 1.6])
    with c1:
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<div style='text-align:center;margin-top:-10px;'>{verdict_html(score)}</div>", unsafe_allow_html=True)

    with c2:
        bias_color = r["bias_color"]
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""<div class="metric-box">
              <div class="metric-value" style="color:{bias_color};font-size:16px;">{r['bias_type']}</div>
              <div class="metric-label">Bias Type</div></div>""", unsafe_allow_html=True)
        with m2:
            ec = score_color(100 - r["emotional_score"])
            st.markdown(f"""<div class="metric-box">
              <div class="metric-value" style="color:{ec};">{r['emotional_score']}%</div>
              <div class="metric-label">Emotional Lang</div></div>""", unsafe_allow_html=True)
        with m3:
            cc = score_color(100 - r["clickbait_score"])
            st.markdown(f"""<div class="metric-box">
              <div class="metric-value" style="color:{cc};">{r['clickbait_score']}%</div>
              <div class="metric-label">Clickbait Score</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Score breakdown bar chart
        breakdown = {
            "BERT Model":      r["hf_score"],
            "Source Signals":  r["signal_score"],
            "Low Clickbait":   100 - r["clickbait_score"],
            "Low Emotion":     100 - r["emotional_score"],
        }
        fig2 = go.Figure(go.Bar(
            x=list(breakdown.values()),
            y=list(breakdown.keys()),
            orientation="h",
            marker_color=["#7c3aed","#00e5a0","#ffd166","#60a5fa"],
            marker_line_width=0,
        ))
        fig2.update_layout(
            paper_bgcolor="#0d0d14", plot_bgcolor="#111118",
            font={"color":"#6b6b80","size":11},
            xaxis={"range":[0,100],"gridcolor":"#1e1e2e","tickfont":{"color":"#6b6b80"}},
            yaxis={"gridcolor":"#1e1e2e","tickfont":{"color":"#c8c8d8"}},
            height=180, margin=dict(t=10,b=10,l=10,r=10),
            title={"text":"Score Breakdown","font":{"size":12,"color":"#6b6b80"}},
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Summary + Sentiment
    ca, cb = st.columns(2)
    with ca:
        st.markdown(f"""
<div class="card card-green">
  <div style="font-size:11px;color:#00e5a0;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">📋 AI Summary (Extractive)</div>
  <p style="font-size:13px;line-height:1.8;color:#c8c8d8;">{r['summary']}</p>
  <div style="margin-top:12px;font-size:11px;color:#6b6b80;">Word count: {r['word_count']}</div>
</div>""", unsafe_allow_html=True)

    with cb:
        sent = r["sentiment"]
        pos_pct = int(sent["pos"] * 100)
        neg_pct = int(sent["neg"] * 100)
        neu_pct = int(sent["neu"] * 100)
        fig3 = go.Figure(go.Pie(
            labels=["Positive","Negative","Neutral"],
            values=[pos_pct, neg_pct, neu_pct],
            hole=0.6,
            marker_colors=["#00e5a0","#ff4d6d","#3a3a5a"],
            textfont={"color":"#e8e8f0","size":11},
        ))
        fig3.update_layout(
            paper_bgcolor="#16161f", plot_bgcolor="#16161f",
            font={"color":"#e8e8f0"},
            showlegend=True,
            legend={"font":{"color":"#6b6b80","size":11}},
            height=220, margin=dict(t=20,b=10,l=10,r=10),
            title={"text":"Sentiment Breakdown","font":{"size":12,"color":"#6b6b80"}},
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Red Flags + Positive Signals
    rd, rg = st.columns(2)
    with rd:
        st.markdown('<div style="font-size:11px;color:#ff4d6d;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">🚩 Red Flags</div>', unsafe_allow_html=True)
        flags = r["red_flags"]
        if flags:
            for f in flags:
                st.markdown(f'<div class="red-flag">⚑ {f}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="green-flag">No major red flags detected.</div>', unsafe_allow_html=True)

    with rg:
        st.markdown('<div style="font-size:11px;color:#00e5a0;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">✅ Positive Signals</div>', unsafe_allow_html=True)
        sigs = r["positive_signals"]
        if sigs:
            for s in sigs:
                st.markdown(f'<div class="green-flag">✓ {s}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="red-flag">No strong credibility signals found.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key Claims
    if r["key_claims"]:
        st.markdown('<div style="font-size:11px;color:#c4b5fd;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">🔎 Key Claims Identified</div>', unsafe_allow_html=True)
        for claim in r["key_claims"]:
            st.markdown(f'<div class="claim-pill">💬 {claim}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Named Entities
    if r["entities"]:
        st.markdown('<div style="font-size:11px;color:#ffd166;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">🏷️ Named Entities Detected</div>', unsafe_allow_html=True)
        label_names = {"PERSON":"👤 People","ORG":"🏢 Organizations","GPE":"🌍 Locations","DATE":"📅 Dates","EVENT":"📰 Events","NORP":"🚩 Groups"}
        for label, ents in r["entities"].items():
            nice = label_names.get(label, label)
            tags = " ".join(f'<span class="entity-tag">{e}</span>' for e in ents)
            st.markdown(f'<div style="margin:6px 0;"><span style="font-size:11px;color:#6b6b80;">{nice}: </span>{tags}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Student Tip
    st.markdown(f"""
<div style="background:rgba(255,209,102,0.07);border:1px solid rgba(255,209,102,0.2);border-radius:12px;padding:18px 22px;margin-bottom:24px;">
  <span style="font-size:11px;color:#ffd166;letter-spacing:2px;text-transform:uppercase;">💡 Student Tip</span>
  <p style="font-size:13px;color:#ffe599;margin-top:8px;line-height:1.7;">{r['student_tip']}</p>
</div>""", unsafe_allow_html=True)

    # ── Trusted Sources
    st.markdown("---")
    st.markdown("## 🌐 Verified News Sources")
    st.markdown('<p style="color:#6b6b80;font-size:13px;margin-bottom:20px;">Cross-check this story using these trusted outlets and fact-checking platforms.</p>', unsafe_allow_html=True)

    links = get_news_links(r["topics"])

    ta, tb, tc = st.tabs(["📰 News Outlets", "🔍 Fact Checkers", "🔎 Topic Search"])
    with ta:
        outlets = [l for l in links if l["tag"] == "outlet"]
        cols = st.columns(2)
        for i, lk in enumerate(outlets):
            with cols[i % 2]:
                st.markdown(f"""<a href="{lk['url']}" target="_blank" style="text-decoration:none;">
<div class="news-link">{lk['icon']} <span style="color:#e8e8f0;margin-left:8px;font-size:13px;">{lk['name']}</span>
<span style="color:#6b6b80;font-size:11px;float:right;margin-top:2px;">↗</span></div></a>""", unsafe_allow_html=True)

    with tb:
        checkers = [l for l in links if l["tag"] == "fact-check"]
        for lk in checkers:
            st.markdown(f"""<a href="{lk['url']}" target="_blank" style="text-decoration:none;">
<div class="news-link">{lk['icon']} <span style="color:#e8e8f0;margin-left:8px;font-size:13px;">{lk['name']}</span>
<span style="color:#6b6b80;font-size:11px;float:right;margin-top:2px;">↗</span></div></a>""", unsafe_allow_html=True)

    with tc:
        searches = [l for l in links if l["tag"] == "search"]
        if searches:
            for lk in searches:
                st.markdown(f"""<a href="{lk['url']}" target="_blank" style="text-decoration:none;">
<div class="news-link">{lk['icon']} <span style="color:#a0a0c0;margin-left:8px;font-size:13px;">{lk['name']}</span>
<span style="color:#6b6b80;font-size:11px;float:right;margin-top:2px;">↗</span></div></a>""", unsafe_allow_html=True)
        else:
            st.info("Analyze an article to get topic-specific search links.")

    # ── Export + Reset
    st.markdown("---")
    ec1, ec2 = st.columns(2)
    with ec1:
        report = f"""TRUTHLENS ANALYSIS REPORT
Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*52}
CREDIBILITY SCORE : {score}/100
VERDICT           : {r['verdict']}
BIAS              : {r['bias_type']}
EMOTIONAL LANG    : {r['emotional_score']}%
CLICKBAIT SCORE   : {r['clickbait_score']}%
WORD COUNT        : {r['word_count']}

SUMMARY:
{r['summary']}

KEY CLAIMS:
{chr(10).join(f'  - {c}' for c in r['key_claims'])}

RED FLAGS:
{chr(10).join(f'  - {f}' for f in r['red_flags'])}

POSITIVE SIGNALS:
{chr(10).join(f'  - {s}' for s in r['positive_signals'])}

STUDENT TIP:
{r['student_tip']}

TRUSTED SOURCES:
  - BBC News     : https://www.bbc.com/news
  - Reuters      : https://www.reuters.com
  - AP News      : https://apnews.com
  - FactCheck.org: https://www.factcheck.org
  - Snopes       : https://www.snopes.com
"""
        st.download_button("📄 Download Report", data=report,
            file_name=f"truthlens_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain", use_container_width=True)
    with ec2:
        if st.button("🔄 Analyze Another Article", use_container_width=True):
            st.session_state.analysis = None
            st.rerun()

else:
    # ── Empty state
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
<div style="text-align:center;padding:60px 20px;border:1px dashed #1e1e2e;border-radius:16px;">
  <div style="font-size:52px;margin-bottom:16px;">🔍</div>
  <h3 style="font-family:'Syne',sans-serif;color:#3a3a5a;font-weight:700;">Ready to Analyse</h3>
  <p style="color:#2a2a4a;font-size:13px;max-width:420px;margin:10px auto;line-height:1.7;">
    Paste a news article above and click <b style="color:#1a5a3a;">Analyze</b> to get a full AI-powered credibility report.
  </p>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(4)
    feats = [
        ("🤖","BERT Classifier","Fine-tuned fake news detection model"),
        ("📊","Credibility Score","Weighted multi-model 0–100% rating"),
        ("🧠","NLP Analysis","spaCy entities, claims & bias"),
        ("🌐","Verified Sources","Links to BBC, Reuters, Snopes & more"),
    ]
    for col, (icon, title, desc) in zip(cols, feats):
        with col:
            st.markdown(f"""
<div style="background:#16161f;border:1px solid #1e1e2e;border-radius:12px;padding:20px;text-align:center;height:150px;">
  <div style="font-size:28px;margin-bottom:8px;">{icon}</div>
  <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:13px;margin-bottom:6px;">{title}</div>
  <div style="font-size:11px;color:#6b6b80;line-height:1.5;">{desc}</div>
</div>""", unsafe_allow_html=True)
