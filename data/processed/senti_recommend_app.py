"""
╔══════════════════════════════════════════════════════════════╗
║         SENTI-RECOMMEND — AI Tool Discovery Engine          ║
║   Hybrid Collaborative Filtering + Sentiment Analysis        ║
╚══════════════════════════════════════════════════════════════╝

HOW TO RUN:
-----------
1. Install dependencies (run once in your terminal):
   pip install streamlit pandas nltk scikit-learn
   python -c "import nltk; nltk.download('vader_lexicon')"

2. Place your data files in the same folder as this script:
   - cleaned_metadata.csv
   - cleaned_reviews.csv

3. Launch the app:
   python -m streamlit run senti_recommend_app.py

4. Open your browser at: http://localhost:8501
"""

# ─────────────────────────── IMPORTS ───────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER data silently on first run
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

# ─────────────────────── PAGE CONFIGURATION ────────────────────
st.set_page_config(
    page_title="Senti-Recommend",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── STYLING ───────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

  .stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1528 40%, #0a1a2e 100%);
    min-height: 100vh;
  }

  .hero-header {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(16,185,129,0.10));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px; padding: 2.5rem 2rem; margin-bottom: 2rem;
    text-align: center; position: relative; overflow: hidden;
  }
  .hero-header::before {
    content: ""; position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(99,102,241,0.05) 0%, transparent 60%);
    animation: pulse 6s ease-in-out infinite;
  }
  @keyframes pulse {
    0%,100% { transform: scale(1); opacity:0.5; }
    50%      { transform: scale(1.1); opacity:1; }
  }
  .hero-title {
    font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #34d399, #818cf8);
    background-size: 200%;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite; margin: 0 0 0.5rem;
  }
  @keyframes shimmer { 0% { background-position:0% } 100% { background-position:200% } }
  .hero-subtitle { color: rgba(196,213,255,0.75); font-size: 1.1rem; }

  .app-card {
    background: rgba(15,23,42,0.85); border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px; padding: 1.5rem 1.75rem; margin-bottom: 1.2rem;
    transition: border-color 0.25s, transform 0.2s; position: relative; overflow: hidden;
  }
  .app-card:hover { border-color: rgba(99,102,241,0.65); transform: translateY(-2px); }
  .app-card::after {
    content: ""; position: absolute; top: 0; left: 0;
    width: 4px; height: 100%; border-radius: 4px 0 0 4px;
  }
  .card-rank-1::after { background: linear-gradient(180deg, #ffd700, #ffa500); }
  .card-rank-2::after { background: linear-gradient(180deg, #c0c0c0, #a8a8a8); }
  .card-rank-3::after { background: linear-gradient(180deg, #cd7f32, #b8722d); }
  .card-rank-other::after { background: linear-gradient(180deg, #6366f1, #4f46e5); }

  .app-name  { font-size: 1.25rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.2rem; }
  .app-dev   { font-size: 0.82rem; color: #94a3b8; margin-bottom: 0.8rem; }
  .app-desc  { font-size: 0.9rem; color: #cbd5e1; line-height: 1.6; margin-bottom: 1rem; }
  .app-badges { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.75rem; }

  .badge {
    display: inline-block; padding: 0.25rem 0.75rem;
    border-radius: 99px; font-size: 0.76rem; font-weight: 600; letter-spacing: 0.04em;
  }
  .badge-category { background: rgba(99,102,241,0.2);  color: #a5b4fc; border: 1px solid rgba(99,102,241,0.4); }
  .badge-free     { background: rgba(16,185,129,0.2);  color: #6ee7b7; border: 1px solid rgba(16,185,129,0.4); }
  .badge-freemium { background: rgba(245,158,11,0.2);  color: #fcd34d; border: 1px solid rgba(245,158,11,0.4); }
  .badge-paid     { background: rgba(239,68,68,0.2);   color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }

  .score-row { display: flex; align-items: center; gap: 0.75rem; margin-top: 0.4rem; }
  .score-label {
    font-size: 0.78rem; color: #94a3b8; min-width: 120px;
    font-family: 'JetBrains Mono', monospace;
  }
  .score-bar-bg { flex: 1; height: 6px; background: rgba(255,255,255,0.07); border-radius: 99px; overflow: hidden; }
  .score-bar-fill { height: 100%; border-radius: 99px; }
  .fill-hybrid    { background: linear-gradient(90deg, #818cf8, #34d399); }
  .fill-rating    { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
  .fill-sentiment { background: linear-gradient(90deg, #10b981, #6ee7b7); }
  .score-val {
    font-size: 0.8rem; font-family: 'JetBrains Mono', monospace;
    color: #e2e8f0; min-width: 40px; text-align: right;
  }

  .sent-pill {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.2rem 0.65rem; border-radius: 99px; font-size: 0.8rem; font-weight: 600;
  }
  .sent-positive { background: rgba(16,185,129,0.18); color: #34d399; border: 1px solid rgba(16,185,129,0.35); }
  .sent-neutral  { background: rgba(148,163,184,0.15); color: #94a3b8; border: 1px solid rgba(148,163,184,0.3); }
  .sent-negative { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }

  .not-found-box {
    background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.3);
    border-radius: 16px; padding: 2.5rem; text-align: center; margin-top: 1.5rem;
  }
  .not-found-icon  { font-size: 3rem; margin-bottom: 0.5rem; }
  .not-found-title { font-size: 1.4rem; font-weight: 700; color: #fca5a5; margin-bottom: 0.5rem; }
  .not-found-msg   { color: #94a3b8; font-size: 0.95rem; line-height: 1.7; }

  section[data-testid="stSidebar"] {
    background: rgba(10,14,26,0.95) !important;
    border-right: 1px solid rgba(99,102,241,0.2);
  }

  .stTextArea textarea {
    background: rgba(15,23,42,0.9) !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    border-radius: 12px !important; color: #e2e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important; font-size: 1rem !important;
  }
  .stTextArea textarea:focus {
    border-color: rgba(99,102,241,0.8) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
  }

  .stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important; padding: 0.6rem 2rem !important;
    transition: all 0.2s !important; width: 100%;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #818cf8, #6366f1) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 20px rgba(99,102,241,0.35) !important;
  }

  .stat-box {
    background: rgba(15,23,42,0.7); border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px; padding: 1rem; text-align: center;
  }
  .stat-num { font-size: 1.6rem; font-weight: 700; color: #818cf8; }
  .stat-lbl { font-size: 0.78rem; color: #64748b; margin-top: 0.2rem; }

  .section-hdr {
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #c7d2fe;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 8px; padding: 0.4rem 0.85rem;
    margin: 1.5rem 0 0.75rem; display: inline-flex; align-items: center; gap: 0.5rem;
  }

  .intent-box {
    background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px; padding: 1rem 1.25rem; margin: 1rem 0;
  }
  .keyword-tag {
    display: inline-block; padding: 0.15rem 0.6rem;
    background: rgba(99,102,241,0.15); color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 99px; font-size: 0.76rem; margin: 0.15rem;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════ KEYWORD INTENT MAPPING ════════════════════════
# This is the "brain" of the app — no API needed.
# Each category has a list of words a user might type.
# We count matches and route the query to the right category.

CATEGORY_KEYWORDS = {
    "Productivity AI": [
        "office", "automate", "automation", "task", "tasks", "schedule",
        "scheduling", "calendar", "meeting", "meetings", "workflow",
        "productivity", "organise", "organize", "planner", "reminder",
        "reminders", "to-do", "todo", "notes", "note-taking", "notetaking",
        "summarize", "summarise", "document", "documents", "file", "files",
        "email", "emails", "inbox", "collaborate", "collaboration", "team",
        "teams", "project", "management", "manager", "assistant", "daily",
        "work", "workplace", "business", "report", "reports", "presentation",
        "spreadsheet", "data entry", "transcribe", "transcription", "focus",
        "efficiency", "streamline", "admin", "administrative", "time management",
        "personal assistant", "virtual assistant", "organizer", "diary",
    ],
    "Generative Text / Chatbots": [
        "write", "writing", "writer", "text", "generate", "generation",
        "chatbot", "chat", "gpt", "language", "llm", "content", "blog",
        "article", "essay", "copywriting", "copy", "draft", "drafting",
        "creative writing", "story", "stories", "novel", "script",
        "paraphrase", "rewrite", "grammar", "proofread", "proofreading",
        "translate", "translation", "conversation", "conversational",
        "question answer", "summarise", "summarize", "poem", "poetry",
        "caption", "captions", "newsletter", "email writing", "headline",
        "text generation", "autocomplete text", "writing assistant",
        "ai writer", "language model",
    ],
    "Image Generation": [
        "image", "images", "photo", "photos", "picture", "pictures",
        "generate image", "art", "artwork", "design", "visual", "visuals",
        "illustration", "illustrations", "graphic", "graphics", "logo",
        "logos", "poster", "posters", "banner", "banners", "avatar",
        "draw", "drawing", "paint", "painting", "sketch", "render",
        "rendering", "3d", "animation", "anime", "realistic", "portrait",
        "landscape", "background", "wallpaper", "thumbnail", "ai art",
        "diffusion", "midjourney", "dalle", "stable diffusion",
        "image creator", "photo generator", "creative image",
    ],
    "AI Coding Assistants": [
        "code", "coding", "coder", "program", "programming", "developer",
        "development", "software", "debug", "debugging", "bug", "error",
        "python", "javascript", "java", "html", "css", "sql", "api",
        "function", "script", "scripting", "github", "git", "refactor",
        "autocomplete", "ide", "editor", "syntax", "compile", "compiler",
        "test", "testing", "code review", "pull request", "technical",
        "backend", "frontend", "fullstack", "web development",
        "mobile development", "app development", "algorithm",
        "data structure", "devops", "code generation", "ai coding",
    ],
    "Marketing AI": [
        "marketing", "market", "advertise", "advertising", "advertisement",
        "ads", "campaign", "campaigns", "social media", "instagram",
        "facebook", "twitter", "linkedin", "tiktok", "youtube", "seo",
        "search engine", "keyword research", "analytics", "traffic",
        "conversion", "lead", "leads", "sales", "sell", "brand",
        "branding", "engagement", "audience", "customer", "customers",
        "email marketing", "newsletter", "promotion", "promote", "viral",
        "influencer", "growth hacking", "funnel", "roi", "revenue",
        "ecommerce", "shopify", "product description", "tagline", "slogan",
        "ad copy", "marketing copy", "content marketing",
    ],
}

PRICING_KEYWORDS = {
    "Free":     ["free", "no cost", "zero cost", "without paying", "gratis", "no charge", "freeware", "open source"],
    "Freemium": ["freemium", "free trial", "trial", "limited free", "upgrade", "basic free", "try before"],
    "Paid":     ["paid", "premium", "pro ", "professional", "enterprise", "subscription", "best quality", "buy"],
}


def understand_intent(user_query: str) -> dict:
    """
    Pure Python keyword matching — no API, no internet needed.

    How it works:
    1. Lowercase and clean the query
    2. Check each category's keyword list for matches
    3. The category (or categories) with the most hits wins
    4. Also detect pricing preference from the text
    """
    query_clean = user_query.lower()
    query_clean = re.sub(r"[^\w\s]", " ", query_clean)

    # Score each category
    category_scores = {}
    matched_kws     = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        hits  = []
        for kw in keywords:
            if kw.lower() in query_clean:
                score += 1
                hits.append(kw.lower())
        category_scores[category] = score
        matched_kws[category]     = hits

    max_score = max(category_scores.values())

    # Nothing matched
    if max_score == 0:
        return {
            "categories": [],
            "keywords":   [],
            "pricing_pref": "Any",
            "summary":    user_query,
            "no_match":   True,
        }

    # Pick categories within 1 point of the winner (handles overlapping queries)
    winning_cats = [
        cat for cat, score in category_scores.items()
        if score >= max_score - 1 and score > 0
    ]

    all_hits = []
    for cat in winning_cats:
        all_hits.extend(matched_kws[cat])

    # Detect pricing
    pricing_pref = "Any"
    for pref, pref_kws in PRICING_KEYWORDS.items():
        for pk in pref_kws:
            if pk in query_clean:
                pricing_pref = pref
                break

    return {
        "categories":     winning_cats,
        "keywords":       list(set(all_hits))[:8],
        "pricing_pref":   pricing_pref,
        "summary":        user_query,
        "category_scores": category_scores,
        "no_match":       False,
    }


# ═══════════════════════════ DATA LOADING ══════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    meta    = pd.read_csv("cleaned_metadata.csv")
    reviews = pd.read_csv("cleaned_reviews.csv")
    return meta, reviews


@st.cache_data(show_spinner=False)
def compute_sentiment_scores(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    VADER sentiment on every review → averaged per app.
    compound score: -1.0 (very negative) to +1.0 (very positive)
    """
    analyser = SentimentIntensityAnalyzer()

    def safe_score(text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0.0
        return analyser.polarity_scores(text)["compound"]

    reviews_df = reviews_df.copy()
    reviews_df["sentiment"] = reviews_df["review_text"].apply(safe_score)

    return (
        reviews_df.groupby("app_id")
        .agg(avg_sentiment=("sentiment", "mean"),
             review_count=("sentiment", "count"))
        .reset_index()
    )


# ══════════════════════ HYBRID SCORING & FILTERING ════════════════════
def get_recommendations(meta_df, sent_df, intent, pricing_pref, top_n=10, alpha=0.55):

    cats = intent.get("categories", [])
    filtered = meta_df[meta_df["project_category"].isin(cats)].copy() if cats else meta_df.copy()

    if filtered.empty:
        return pd.DataFrame()

    # Keyword relevance count
    keywords = [kw.lower() for kw in intent.get("keywords", [])]
    def kw_count(row):
        text = " ".join([
            str(row.get("app_name", "")),
            str(row.get("description", "")),
            str(row.get("summary", "")),
        ]).lower()
        return sum(1 for kw in keywords if kw in text)

    filtered["kw_hits"] = filtered.apply(kw_count, axis=1)

    # Pricing filter
    if pricing_pref in ("Free", "Freemium", "Paid"):
        p = filtered[filtered["pricing_model"] == pricing_pref]
        if not p.empty:
            filtered = p

    # Merge sentiment
    filtered = filtered.merge(sent_df, on="app_id", how="left")
    filtered["avg_sentiment"] = filtered["avg_sentiment"].fillna(0.0)
    filtered["review_count"]  = filtered["review_count"].fillna(0)

    # Normalise to 0-1
    filtered["norm_rating"]    = (filtered["bayesian_avg"] - 1) / 4.0
    filtered["norm_sentiment"] = (filtered["avg_sentiment"] + 1) / 2.0

    # Keyword bonus (max +0.12)
    filtered["kw_bonus"] = (filtered["kw_hits"] * 0.03).clip(upper=0.12)

    # Hybrid score
    beta = 1.0 - alpha
    filtered["hybrid_score"] = (
        alpha * filtered["norm_rating"] +
        beta  * filtered["norm_sentiment"] +
        filtered["kw_bonus"]
    ).clip(upper=1.0)

    return filtered.sort_values("hybrid_score", ascending=False).head(top_n).reset_index(drop=True)


# ════════════════════════════ UI HELPERS ═════════════════════════════
def sentiment_pill(score):
    if score >= 0.05:
        return '<span class="sent-pill sent-positive">😊 Positive</span>'
    elif score <= -0.05:
        return '<span class="sent-pill sent-negative">😟 Negative</span>'
    return '<span class="sent-pill sent-neutral">😐 Neutral</span>'


def price_badge(model):
    model = str(model).strip()
    cls  = {"Free":"badge-free","Freemium":"badge-freemium","Paid":"badge-paid"}.get(model,"badge-category")
    icon = {"Free":"🆓","Freemium":"⚡","Paid":"💳"}.get(model,"💰")
    return f'<span class="badge {cls}">{icon} {model}</span>'


def rank_class(i):
    return {0:"card-rank-1",1:"card-rank-2",2:"card-rank-3"}.get(i,"card-rank-other")


def render_card(row, rank):
    name     = row.get("app_name","Unknown App")
    dev      = row.get("developer","Unknown Developer")
    desc     = str(row.get("summary", row.get("description","No description available.")))
    desc     = desc[:280]+"…" if len(desc)>280 else desc
    cat      = row.get("project_category","")
    pricing  = str(row.get("pricing_model",""))
    rating   = float(row.get("bayesian_avg",0))
    sent     = float(row.get("avg_sentiment",0))
    hybrid   = float(row.get("hybrid_score",0))
    installs = int(row.get("installs",0))
    medals   = {0:"🥇",1:"🥈",2:"🥉"}
    medal    = medals.get(rank, f"#{rank+1}")
    stars    = "⭐"*round(rating)+f" {rating:.2f}"

    st.markdown(f"""
    <div class="app-card {rank_class(rank)}">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:0.5rem;">
        <div>
          <div class="app-name">{medal} &nbsp;{name}</div>
          <div class="app-dev">by {dev}</div>
        </div>
        <div>{sentiment_pill(sent)}</div>
      </div>
      <div class="app-badges">
        <span class="badge badge-category">📂 {cat}</span>
        {price_badge(pricing)}
        <span class="badge badge-category">📦 {installs:,} installs</span>
        <span class="badge badge-category">{stars}</span>
      </div>
      <div class="app-desc">{desc}</div>
      <div class="section-hdr">Hybrid Score Breakdown</div>
      <div class="score-row">
        <span class="score-label">🏆 Hybrid Score</span>
        <div class="score-bar-bg"><div class="score-bar-fill fill-hybrid" style="width:{hybrid*100:.1f}%"></div></div>
        <span class="score-val">{hybrid:.2f}</span>
      </div>
      <div class="score-row">
        <span class="score-label">⭐ Rating Signal</span>
        <div class="score-bar-bg"><div class="score-bar-fill fill-rating" style="width:{((rating-1)/4)*100:.1f}%"></div></div>
        <span class="score-val">{rating:.2f}/5</span>
      </div>
      <div class="score-row">
        <span class="score-label">💬 Sentiment</span>
        <div class="score-bar-bg"><div class="score-bar-fill fill-sentiment" style="width:{((sent+1)/2)*100:.1f}%"></div></div>
        <span class="score-val">{sent:+.2f}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════ MAIN APP ══════════════════════════════
def main():

    st.markdown("""
    <div class="hero-header">
      <div class="hero-title">🤖 Senti-Recommend</div>
      <div class="hero-subtitle">
        AI Tool Discovery Engine &nbsp;·&nbsp; Hybrid Collaborative Filtering + Sentiment Analysis
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    try:
        with st.spinner("📦 Loading AI tool database…"):
            meta_df, reviews_df = load_data()
        with st.spinner("💬 Computing sentiment scores from 69,000+ reviews…"):
            sent_df = compute_sentiment_scores(reviews_df)
    except FileNotFoundError:
        st.error(
            "⚠️ **Data files not found.**\n\n"
            "Put `cleaned_metadata.csv` and `cleaned_reviews.csv` "
            "in the **same folder** as this script."
        )
        st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Search Settings")
        st.markdown("---")
        top_n = st.slider("📊 Number of results", 3, 20, 8)
        alpha = st.slider("⚖️ Rating vs Sentiment weight", 0.0, 1.0, 0.55, 0.05,
                          help="Higher = trust star ratings more. Lower = trust review text more.")
        pricing_override = st.selectbox("💰 Pricing preference",
                                        ["Auto-detect","Free","Freemium","Paid","Any"])
        st.markdown("---")
        st.markdown("### 📂 Categories in Dataset")
        for cat in sorted(meta_df["project_category"].unique()):
            count = len(meta_df[meta_df["project_category"]==cat])
            st.caption(f"• {cat}  ({count} apps)")
        st.markdown("---")
        st.markdown("### 📖 How It Works")
        st.caption(
            "1️⃣ Your query is matched by keywords\n\n"
            "2️⃣ VADER reads 69k+ reviews for real sentiment\n\n"
            "3️⃣ Hybrid Score = Rating + Sentiment combined\n\n"
            "4️⃣ Top tools ranked and displayed"
        )

    # Stats row
    c1,c2,c3,c4 = st.columns(4)
    for col,(num,lbl) in zip([c1,c2,c3,c4],[
        (len(meta_df),"Total AI Tools"),
        (f"{len(reviews_df):,}","Total Reviews"),
        (f"{meta_df['avg_rating'].mean():.2f}","Avg. Star Rating"),
        (len(meta_df["project_category"].unique()),"Categories"),
    ]):
        col.markdown(
            f'<div class="stat-box"><div class="stat-num">{num}</div>'
            f'<div class="stat-lbl">{lbl}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-hdr">🔍 Describe what you need</div>', unsafe_allow_html=True)

    EXAMPLES = [
        "I want to automate my daily office tasks and manage my schedule",
        "I need a free tool to write marketing copy and social media posts",
        "Find me an AI coding assistant for Python debugging",
        "I want an image generator for creative artwork on a small budget",
        "Help me generate professional business emails and reports",
    ]

    col_input, col_examples = st.columns([2,1])

    with col_input:
        user_query = st.text_area(
            label="query",
            placeholder="e.g.  I want to automate my office tasks and improve productivity…",
            height=120,
            label_visibility="collapsed",
        )
        search_clicked = st.button("🚀 Find AI Tools", use_container_width=True)

    with col_examples:
        st.markdown("**💡 Try an example:**")
        for ex in EXAMPLES:
            label = ex[:52]+"…" if len(ex)>52 else ex
            if st.button(label, key=f"ex_{ex[:15]}", use_container_width=True):
                user_query     = ex
                search_clicked = True

    # Run search
    if search_clicked and user_query and user_query.strip():

        intent = understand_intent(user_query.strip())

        if intent.get("no_match"):
            st.markdown(f"""
            <div class="not-found-box">
              <div class="not-found-icon">🔍</div>
              <div class="not-found-title">No Matching Tools Found</div>
              <div class="not-found-msg">
                Your request — <em>"{user_query[:140]}"</em> — did not match
                any category in our dataset.<br><br>
                Our database covers <strong>482 apps</strong> across 5 categories:<br><br>
                📝 Generative Text / Chatbots &nbsp;·&nbsp;
                🖼️ Image Generation &nbsp;·&nbsp;
                💻 AI Coding Assistants &nbsp;·&nbsp;
                📋 Productivity AI &nbsp;·&nbsp;
                📣 Marketing AI<br><br>
                <strong>Try rephrasing</strong> — for example:<br>
                <em>"automate tasks"</em>, <em>"write content"</em>,
                <em>"generate images"</em>, <em>"help me code"</em>,
                <em>"social media marketing"</em>
              </div>
            </div>
            """, unsafe_allow_html=True)
            return

        # Show what was detected
        kw_tags = "".join(f'<span class="keyword-tag">{kw}</span>' for kw in intent.get("keywords",[]))
        st.markdown(f"""
        <div class="intent-box">
          <span style="color:#818cf8;font-weight:600;">🎯 Matched category:</span>
          <span style="color:#e2e8f0;margin-left:0.5rem;">
            {" &nbsp;·&nbsp; ".join(intent.get("categories",[]))}
          </span>
          <br><br>
          <span style="color:#64748b;font-size:0.82rem;">Keywords detected → </span>
          {kw_tags if kw_tags else '<span style="color:#64748b;font-size:0.82rem;">none</span>'}
          &nbsp;&nbsp;
          <span style="color:#64748b;font-size:0.82rem;">
            | Pricing → {intent.get("pricing_pref","Any")}
          </span>
        </div>
        """, unsafe_allow_html=True)

        pricing_pref = (
            intent.get("pricing_pref","Any")
            if pricing_override == "Auto-detect"
            else (pricing_override if pricing_override != "Any" else "Any")
        )

        with st.spinner("⚙️ Ranking tools with hybrid scoring…"):
            results = get_recommendations(meta_df, sent_df, intent,
                                          pricing_pref, top_n, alpha)

        if results.empty:
            st.markdown(f"""
            <div class="not-found-box">
              <div class="not-found-icon">📭</div>
              <div class="not-found-title">No Tools Found for This Filter</div>
              <div class="not-found-msg">
                We matched your query to
                <strong>{", ".join(intent.get("categories",[]))}</strong>
                but no apps passed the pricing filter
                (<strong>{pricing_pref}</strong>).<br><br>
                Try setting <strong>Pricing preference → Any</strong> in the sidebar.
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            n_cats = len(intent.get("categories",[]))
            st.markdown(
                f'<div class="section-hdr">🏆 Top {len(results)} Recommendations</div>',
                unsafe_allow_html=True
            )
            st.caption(
                f"Ranked by Hybrid Score = {alpha:.0%} Rating + {1-alpha:.0%} Sentiment  "
                f"| Filtered from {len(meta_df)} apps"
            )
            for i, (_, row) in enumerate(results.iterrows()):
                render_card(row, i)

    elif search_clicked:
        st.warning("⚠️ Please type what you are looking for before searching.")
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;">
          <div style="font-size:3rem;margin-bottom:1rem;">✨</div>
          <div style="font-size:1.1rem;color:#64748b;">
            Describe your need above and click
            <strong style="color:#818cf8;">Find AI Tools</strong><br>
            to discover best-matched tools from our database of 482 apps.
          </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
