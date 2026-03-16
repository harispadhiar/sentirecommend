"""
╔══════════════════════════════════════════════════════════════════╗
║   SENTI-RECOMMEND — Hybrid AI Tool Discovery Engine             ║
║   PGD Data Science with AI — Final Project                      ║
║   Streamlit Application (app.py)                                ║
╚══════════════════════════════════════════════════════════════════╝

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
import json
import warnings
from collections import defaultdict

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Senti-Recommend",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — dark-teal data-lab aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --border:    #30363d;
    --teal:      #2dd4bf;
    --teal-dim:  #1a9e8f;
    --amber:     #fbbf24;
    --rose:      #fb7185;
    --muted:     #8b949e;
    --text:      #e6edf3;
    --radius:    12px;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

h1, h2, h3, h4 {
    font-family: 'Space Mono', monospace !important;
    color: var(--text) !important;
}

.stButton > button {
    background: var(--teal) !important;
    color: #0d1117 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 0.5rem 1.4rem !important;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: var(--teal-dim) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(45,212,191,0.3) !important;
}

.stSlider > div > div { accent-color: var(--teal); }

/* Card styles */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.9rem;
    transition: border-color 0.2s ease;
}
.card:hover { border-color: var(--teal); }

.card-rank {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--teal);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.0rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.25rem;
}
.card-meta {
    font-size: 0.82rem;
    color: var(--muted);
    margin-bottom: 0.6rem;
}
.score-bar-wrap { margin-top: 0.5rem; }
.score-label {
    font-size: 0.75rem;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    margin-bottom: 2px;
}
.score-bar-bg {
    background: var(--border);
    border-radius: 4px;
    height: 6px;
    margin-bottom: 5px;
    overflow: hidden;
}
.score-bar-fill { height: 100%; border-radius: 4px; }

.badge {
    display: inline-block;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    padding: 2px 8px;
    border-radius: 20px;
    margin-right: 4px;
    font-weight: 700;
}
.badge-cat  { background: rgba(45,212,191,0.15); color: var(--teal); }
.badge-free { background: rgba(251,191,36,0.15);  color: var(--amber); }
.badge-gem  { background: rgba(251,113,133,0.15); color: var(--rose); }

/* Metric tiles */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.4rem; }
.metric-tile {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--teal);
}
.metric-lbl { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }

/* Header banner */
.banner {
    background: linear-gradient(135deg, #0d2330 0%, #0d1117 50%, #1a0d20 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.banner::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        45deg, transparent, transparent 40px,
        rgba(45,212,191,0.02) 40px, rgba(45,212,191,0.02) 80px
    );
}
.banner h1 { font-size: 2rem; margin: 0 0 0.3rem 0; }
.banner p  { color: var(--muted); margin: 0; font-size: 0.9rem; }

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

/* Tab overrides */
[data-testid="stTab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* Input fields */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] select {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* Dividers */
hr { border-color: var(--border) !important; }

/* Star rating */
.stars { color: var(--amber); font-size: 0.9rem; letter-spacing: 1px; }

/* Expander */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def star_str(rating):
    if pd.isna(rating):
        return "N/A"
    full  = int(rating)
    empty = 5 - full
    return "★" * full + "☆" * empty + f"  {rating:.1f}"

def fmt_number(n):
    if pd.isna(n): return "—"
    n = int(n)
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.1f}K"
    return str(n)

def score_bar(label, value, color="#2dd4bf"):
    pct = max(0, min(1, float(value))) * 100
    return f"""
    <div class="score-bar-wrap">
      <div class="score-label"><span>{label}</span><span>{value:.3f}</span></div>
      <div class="score-bar-bg">
        <div class="score-bar-fill" style="width:{pct:.1f}%;background:{color}"></div>
      </div>
    </div>"""

def clean_text_fn(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+|\S+@\S+', ' ', text)
    text = re.sub(r"[^a-z\'\s]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# CACHED MODEL BUILDING  (runs once, cached for session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def build_all_models():
    """
    Loads raw CSVs and builds ALL models from scratch.
    Results are cached — this runs once per session.
    """
    # ── 1. Load raw data ──────────────────────────────────────────────────────
    reviews  = pd.read_csv("cleaned_reviews.csv")
    metadata = pd.read_csv("cleaned_metadata.csv")

    # ── 2. Sentiment proxy from star ratings (NB02 fallback) ──────────────────
    # Try loading NLP scores; if unavailable, compute from ratings
    try:
        sent_df = pd.read_csv("app_sentiment_scores.csv")
        metadata = metadata.merge(
            sent_df[['app_id','mean_sentiment_score','is_hidden_gem']],
            on='app_id', how='left'
        )
        nlp_source = "NLP Model (Notebook 02)"
    except FileNotFoundError:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        metadata['mean_sentiment_score'] = scaler.fit_transform(
            metadata[['bayesian_avg']].fillna(metadata['bayesian_avg'].median())
        )
        s75 = metadata['bayesian_avg'].quantile(0.55)
        r25 = metadata['total_reviews'].quantile(0.30)
        metadata['is_hidden_gem'] = (
            (metadata['bayesian_avg'] >= s75) &
            (metadata['total_reviews'] <= r25)
        ).astype(int)
        nlp_source = "Rating Proxy (run NB02 for full NLP scores)"

    metadata['mean_sentiment_score'] = metadata['mean_sentiment_score'].fillna(0)
    metadata['is_hidden_gem']        = metadata['is_hidden_gem'].fillna(0).astype(int)

    # ── 3. User-Item Matrix ───────────────────────────────────────────────────
    pivot = reviews.pivot_table(
        index='user_name', columns='app_id',
        values='star_rating', aggfunc='mean'
    )
    user_means    = pivot.mean(axis=1)
    pivot_centred = pivot.subtract(user_means, axis=0)

    app_ids    = list(pivot.columns)
    user_ids   = list(pivot.index)
    app_to_idx = {a: i for i, a in enumerate(app_ids)}

    item_sparse = csr_matrix(pivot_centred.fillna(0).values.T)

    # ── 4. Item-CF Similarity ─────────────────────────────────────────────────
    try:
        item_sim = np.load("item_similarity_matrix.npy")
    except FileNotFoundError:
        item_sim = cosine_similarity(item_sparse)
        np.fill_diagonal(item_sim, 0)

    # ── 5. SVD ────────────────────────────────────────────────────────────────
    try:
        app_factors = np.load("svd_app_factors.npy")
        svd_sim     = np.load("svd_sim_matrix.npy")
    except FileNotFoundError:
        svd = TruncatedSVD(n_components=50, random_state=42)
        app_factors = svd.fit_transform(item_sparse)
        svd_sim = cosine_similarity(app_factors)
        np.fill_diagonal(svd_sim, 0)

    # ── 6. Content-Based TF-IDF ───────────────────────────────────────────────
    metadata['content_text'] = (
        metadata['app_name'].fillna('') + ' ' +
        metadata['project_category'].fillna('') + ' ' +
        metadata['summary'].fillna('') + ' ' +
        metadata['description'].fillna('').str[:500]
    ).str.lower()

    content_tfidf = TfidfVectorizer(
        max_features=3000, ngram_range=(1, 2),
        min_df=1, stop_words='english', strip_accents='unicode'
    )
    content_mat = content_tfidf.fit_transform(metadata['content_text'])
    content_sim = cosine_similarity(content_mat)
    np.fill_diagonal(content_sim, 0)
    meta_app_to_idx = {row['app_id']: i for i, row in metadata.iterrows()}

    # ── 7. Sentiment Score Lookup (normalised 0–1) ────────────────────────────
    scaler_sent = MinMaxScaler(feature_range=(0, 1))
    metadata['sentiment_norm'] = scaler_sent.fit_transform(
        metadata[['mean_sentiment_score']]
    )
    sentiment_lookup  = dict(zip(metadata['app_id'], metadata['sentiment_norm']))
    hidden_gem_lookup = dict(zip(metadata['app_id'], metadata['is_hidden_gem']))

    # ── 8. Keyword-based ABSA per app (from review text) ─────────────────────
    ASPECT_KW = {
        'Ease of Use':       ['easy','simple','intuitive','user friendly','beginner','accessible',
                               'learning curve','complicated','difficult','confusing','smooth'],
        'Pricing':           ['price','expensive','cheap','affordable','cost','subscription',
                               'free','premium','overpriced','worth','value','refund'],
        'Customer Support':  ['support','customer service','response','helpful','ignored',
                               'contact','reply','team','resolved','agent'],
        'Performance':       ['fast','slow','crash','lag','speed','bug','glitch','reliable',
                               'stable','freeze','error','performance'],
        'Features':          ['feature','functionality','integration','plugin','option',
                               'missing','lacks','limited','powerful','update','api']
    }
    POS_WORDS = {'great','excellent','amazing','love','best','perfect','awesome','fast',
                 'helpful','good','smooth','reliable','easy','simple','impressed',
                 'satisfied','powerful','intuitive','quick','worth'}
    NEG_WORDS = {'bad','terrible','awful','worst','hate','useless','crash','bug','slow',
                 'disappointing','expensive','overpriced','waste','limited','missing',
                 'poor','frustrating','annoying','broken','ignored','error'}
    NEG_NEG   = {'not','no','never',"don't","doesn't","won't","can't","isn't"}

    def absa_app(text):
        if not text or len(str(text)) < 5:
            return {a: 0.0 for a in ASPECT_KW}
        t = str(text).lower()
        sentences = re.split(r'[.!?\n]', t)
        aspect_scores = defaultdict(list)
        for sent in sentences:
            if len(sent.strip()) < 4:
                continue
            words = sent.split()
            for aspect, kws in ASPECT_KW.items():
                if not any(k in sent for k in kws):
                    continue
                pos = neg = 0
                for j, w in enumerate(words):
                    cw = re.sub(r"[^a-z']", '', w)
                    neg_ctx = any(words[k].rstrip("',") in NEG_NEG
                                  for k in range(max(0, j-3), j))
                    if cw in POS_WORDS:
                        neg += 1 if neg_ctx else 0
                        pos += 0 if neg_ctx else 1
                    elif cw in NEG_WORDS:
                        pos += 1 if neg_ctx else 0
                        neg += 0 if neg_ctx else 1
                if pos + neg > 0:
                    aspect_scores[aspect].append((pos - neg) / (pos + neg))
        return {a: float(np.mean(v)) if v else 0.0 for a, v in aspect_scores.items()}

    # Aggregate reviews per app (sample max 300 reviews per app for speed)
    app_reviews_text = (reviews.groupby('app_id')['review_text']
                        .apply(lambda x: ' '.join(x.dropna().head(300).astype(str)))
                        .reset_index())
    app_reviews_text.columns = ['app_id', 'agg_text']
    app_absa = {}
    for _, row in app_reviews_text.iterrows():
        app_absa[row['app_id']] = absa_app(row['agg_text'])

    # ── Bundle everything ─────────────────────────────────────────────────────
    return {
        'reviews':          reviews,
        'metadata':         metadata,
        'app_ids':          app_ids,
        'user_ids':         user_ids,
        'app_to_idx':       app_to_idx,
        'item_sim':         item_sim,
        'app_factors':      app_factors,
        'svd_sim':          svd_sim,
        'content_tfidf':    content_tfidf,
        'content_mat':      content_mat,
        'content_sim':      content_sim,
        'meta_app_to_idx':  meta_app_to_idx,
        'sentiment_lookup': sentiment_lookup,
        'hidden_gem_lookup':hidden_gem_lookup,
        'app_absa':         app_absa,
        'nlp_source':       nlp_source,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_recommend(
    models, user_ratings=None, query_text=None,
    n=10, category=None, free_only=None, pricing_filter=None,
    alpha=0.50, beta=0.35, gamma=0.15, gem_boost=0.05
):
    """
    Main hybrid recommendation function.
    Returns a DataFrame of ranked recommendations.
    """
    meta         = models['metadata']
    app_ids      = models['app_ids']
    app_to_idx   = models['app_to_idx']
    item_sim     = models['item_sim']
    svd_sim      = models['svd_sim']
    app_factors  = models['app_factors']
    content_sim  = models['content_sim']
    content_tfidf= models['content_tfidf']
    content_mat  = models['content_mat']
    meta_idx     = models['meta_app_to_idx']
    sent_lkp     = models['sentiment_lookup']
    gem_lkp      = models['hidden_gem_lookup']

    # Normalise weights
    total = alpha + beta + gamma
    a, b, g = alpha/total, beta/total, gamma/total

    # Build candidate pool
    candidates = meta.copy()
    if category and category != "All Categories":
        candidates = candidates[candidates['project_category'] == category]
    if free_only:
        candidates = candidates[candidates['free'] == True]
    if pricing_filter and pricing_filter != "All":
        candidates = candidates[candidates['pricing_model'] == pricing_filter]
    if user_ratings:
        candidates = candidates[~candidates['app_id'].isin(user_ratings.keys())]

    cand_ids = candidates['app_id'].tolist()
    if not cand_ids:
        return pd.DataFrame()

    # ── CF scores ─────────────────────────────────────────────────────────────
    cf_raw = {}
    if user_ratings:
        for target_aid in cand_ids:
            if target_aid not in app_to_idx:
                cf_raw[target_aid] = 0.0; continue
            t_idx = app_to_idx[target_aid]
            num = den = 0.0
            for rated_aid, rating in user_ratings.items():
                if rated_aid not in app_to_idx: continue
                r_idx = app_to_idx[rated_aid]
                sim_v = item_sim[t_idx, r_idx]
                if sim_v > 0:
                    num += sim_v * rating
                    den += sim_v
            if den > 0:
                cf_raw[target_aid] = num / den
            else:
                # SVD fallback
                liked = [a for a, r in user_ratings.items() if r >= 4 and a in app_to_idx]
                if liked:
                    ideal = app_factors[[app_to_idx[a] for a in liked]].mean(axis=0, keepdims=True)
                    sims  = cosine_similarity(ideal, app_factors)[0]
                    cf_raw[target_aid] = float(sims[t_idx]) if t_idx < len(sims) else 0.0
                else:
                    cf_raw[target_aid] = 0.0
    else:
        # No user history — use Bayesian average as CF signal
        for aid in cand_ids:
            row = candidates[candidates['app_id'] == aid]
            cf_raw[aid] = float(row['bayesian_avg'].values[0]) if len(row) else 0.0

    # Normalise CF to 0–1
    cf_vals = list(cf_raw.values())
    mn, mx  = min(cf_vals), max(cf_vals)
    cf_norm = {aid: (v - mn) / (mx - mn) if mx > mn else 0.5 for aid, v in cf_raw.items()}

    # ── Content scores ────────────────────────────────────────────────────────
    content_norm = {}
    if query_text and query_text.strip():
        q_vec  = content_tfidf.transform([query_text.lower()])
        q_sims = cosine_similarity(q_vec, content_mat)[0]
        raw_c  = {aid: float(q_sims[meta_idx[aid]]) if aid in meta_idx else 0.0
                  for aid in cand_ids}
    elif user_ratings:
        liked_idxs = [meta_idx[a] for a in user_ratings if a in meta_idx and user_ratings[a] >= 4]
        if liked_idxs:
            raw_c = {}
            for aid in cand_ids:
                if aid in meta_idx:
                    sims = [content_sim[li, meta_idx[aid]] for li in liked_idxs]
                    raw_c[aid] = float(np.mean(sims))
                else:
                    raw_c[aid] = 0.0
        else:
            raw_c = {aid: 0.5 for aid in cand_ids}
    else:
        raw_c = {aid: 0.5 for aid in cand_ids}

    c_vals = list(raw_c.values())
    cmn, cmx = min(c_vals), max(c_vals)
    # When a query is present, skip normalisation — preserve raw cosine scores
    # so low-relevance queries score near 0, not 0.5
    if query_text and query_text.strip():
        content_norm = raw_c   # raw cosine similarity (0–1), no rescaling
    else:
        content_norm = {aid: (v - cmn)/(cmx - cmn) if cmx > cmn else 0.5
                        for aid, v in raw_c.items()}
                        
    # ── Fuse ──────────────────────────────────────────────────────────────────
    results = []
    for aid in cand_ids:
        cf_s   = cf_norm.get(aid, 0)
        nlp_s  = sent_lkp.get(aid, 0.5)
        cont_s = content_norm.get(aid, 0)
        is_gem = gem_lkp.get(aid, 0)

        hybrid = a * cf_s + b * nlp_s + g * cont_s + gem_boost * is_gem
        hybrid = min(1.0, hybrid)

        row = candidates[candidates['app_id'] == aid]
        if len(row) == 0: continue
        row = row.iloc[0]
        results.append({
            'app_id':       aid,
            'app_name':     str(row['app_name']),
            'category':     str(row['project_category']),
            'pricing':      str(row['pricing_model']),
            'avg_rating':   float(row['avg_rating']) if pd.notna(row['avg_rating']) else 0.0,
            'total_ratings':int(row['total_ratings']) if pd.notna(row['total_ratings']) else 0,
            'total_reviews':int(row['total_reviews']) if pd.notna(row['total_reviews']) else 0,
            'installs':     int(row['installs']) if pd.notna(row['installs']) else 0,
            'is_hidden_gem':int(is_gem),
            'hybrid_score': round(hybrid, 4),
            'cf_score':     round(cf_s, 4),
            'nlp_score':    round(nlp_s, 4),
            'content_score':round(cont_s, 4),
            'summary':      str(row.get('summary', '')) if pd.notna(row.get('summary', '')) else '',
        })

    df = pd.DataFrame(results).sort_values('hybrid_score', ascending=False).head(n)
    df.insert(0, 'rank', range(1, len(df)+1))
    return df.reset_index(drop=True)


def find_similar_apps(models, app_id, n=6, mode='hybrid'):
    """Find apps similar to a given app using item-CF or SVD."""
    meta      = models['metadata']
    app_ids   = models['app_ids']
    app_to_idx= models['app_to_idx']
    meta_idx  = models['meta_app_to_idx']

    if mode == 'svd':
        sim_mat = models['svd_sim']
    else:
        sim_mat = models['item_sim']

    if app_id not in app_to_idx:
        return pd.DataFrame()

    idx  = app_to_idx[app_id]
    sims = sim_mat[idx].copy()

    results = []
    top_idxs = np.argsort(sims)[::-1][:n*2]
    for i in top_idxs:
        if sims[i] <= 0: continue
        aid = app_ids[i]
        row = meta[meta['app_id'] == aid]
        if len(row) == 0: continue
        row = row.iloc[0]
        results.append({
            'app_id':       aid,
            'app_name':     str(row['app_name']),
            'category':     str(row['project_category']),
            'avg_rating':   float(row['avg_rating']) if pd.notna(row['avg_rating']) else 0.0,
            'similarity':   round(float(sims[i]), 4),
        })
        if len(results) >= n: break

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_score_chart(row_data):
    """Mini horizontal bar chart for score decomposition."""
    labels = ['CF Score', 'NLP Score', 'Content Score', 'Hybrid Score']
    values = [row_data['cf_score'], row_data['nlp_score'],
              row_data['content_score'], row_data['hybrid_score']]
    colors = ['#3b82f6', '#2dd4bf', '#f59e0b', '#a78bfa']

    fig, ax = plt.subplots(figsize=(5, 2.2))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')

    bars = ax.barh(labels, values, color=colors, edgecolor='none', height=0.55)
    ax.set_xlim(0, 1)
    ax.spines[['top','right','bottom','left']].set_visible(False)
    ax.tick_params(axis='both', colors='#8b949e', labelsize=9)
    ax.xaxis.set_visible(False)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color='#8b949e', fontsize=9)

    for bar, val in zip(bars, values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', color='#e6edf3', fontsize=9, fontweight='bold')

    plt.tight_layout(pad=0.3)
    return fig


def make_absa_radar(absa_scores):
    """Radar chart for ABSA aspect scores."""
    aspects = list(absa_scores.keys())
    values  = [max(-1, min(1, v)) for v in absa_scores.values()]
    values_norm = [(v + 1) / 2 for v in values]  # scale to 0–1

    N = len(aspects)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values_norm += values_norm[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')

    ax.plot(angles, values_norm, 'o-', linewidth=2, color='#2dd4bf')
    ax.fill(angles, values_norm, alpha=0.2, color='#2dd4bf')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(aspects, size=8, color='#8b949e')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['−0.5', '0', '+0.5'], size=7, color='#555')
    ax.spines['polar'].set_color('#30363d')
    ax.grid(color='#30363d', linewidth=0.7)

    plt.tight_layout()
    return fig


def make_rating_donut(row_data, metadata_full):
    """Donut chart for star rating breakdown."""
    meta = metadata_full[metadata_full['app_id'] == row_data['app_id']]
    if len(meta) == 0:
        return None
    meta = meta.iloc[0]
    vals = [
        meta.get('ratings_5_star', 0) or 0,
        meta.get('ratings_4_star', 0) or 0,
        meta.get('ratings_3_star', 0) or 0,
        meta.get('ratings_2_star', 0) or 0,
        meta.get('ratings_1_star', 0) or 0,
    ]
    labels = ['5★', '4★', '3★', '2★', '1★']
    colors = ['#27ae60','#2ecc71','#f1c40f','#e67e22','#e74c3c']

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')

    wedges, texts, autotexts = ax.pie(
        vals, labels=labels, colors=colors, autopct='%1.0f%%',
        startangle=90, pctdistance=0.78,
        wedgeprops={'edgecolor': '#161b22', 'linewidth': 2, 'width': 0.55}
    )
    for t in texts: t.set_color('#8b949e'); t.set_fontsize(8)
    for t in autotexts: t.set_color('white'); t.set_fontsize(7)

    ax.text(0, 0, f"{meta.get('avg_rating', 0):.1f}\n★",
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='#fbbf24', fontfamily='monospace')

    plt.tight_layout()
    return fig


def make_category_dist_chart(metadata_full):
    """Bar chart of apps per category."""
    counts = metadata_full['project_category'].value_counts()
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    palette = ['#2dd4bf','#3b82f6','#a78bfa','#f59e0b','#fb7185']
    bars = ax.barh(counts.index, counts.values,
                   color=palette[:len(counts)], edgecolor='none', height=0.6)
    ax.set_xlabel("Number of Apps", color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e', labelsize=9)
    ax.spines[['top','right','bottom','left']].set_color('#30363d')
    for bar, val in zip(bars, counts.values):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                str(val), va='center', color='#e6edf3', fontsize=9)
    plt.tight_layout(pad=0.5)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RENDER: RECOMMENDATION CARD
# ─────────────────────────────────────────────────────────────────────────────

def render_rec_card(row, show_scores=True):
    gem_badge  = '<span class="badge badge-gem">💎 Hidden Gem</span>' if row['is_hidden_gem'] else ''
    free_badge = '<span class="badge badge-free">Free</span>' if 'Free' in str(row['pricing']) else ''
    cat_badge  = f'<span class="badge badge-cat">{row["category"]}</span>'

    rating_disp = star_str(row['avg_rating'])
    reviews_disp = fmt_number(row['total_reviews'])
    installs_disp = fmt_number(row['installs'])

    scores_html = ""
    if show_scores:
        scores_html = (
            score_bar("CF Score",      row['cf_score'],      "#3b82f6") +
            score_bar("NLP Sentiment", row['nlp_score'],     "#2dd4bf") +
            score_bar("Content Score", row['content_score'], "#f59e0b") +
            score_bar("Hybrid Score",  row['hybrid_score'],  "#a78bfa")
        )

    summary_text = str(row.get('summary',''))[:180] + "..." if len(str(row.get('summary',''))) > 180 else str(row.get('summary',''))

    st.markdown(f"""
    <div class="card">
      <div class="card-rank">#{row['rank']:02d}</div>
      <div class="card-title">{row['app_name']}</div>
      <div class="card-meta">{cat_badge} {free_badge} {gem_badge}</div>
      <div class="card-meta">
        <span class="stars">{rating_disp}</span>
        &nbsp;·&nbsp; {reviews_disp} reviews
        &nbsp;·&nbsp; {installs_disp} installs
      </div>
      <div style="font-size:0.82rem;color:#8b949e;margin-bottom:0.6rem">{summary_text}</div>
      {scores_html}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(models):
    st.sidebar.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:1.1rem;
    font-weight:700;color:#2dd4bf;margin-bottom:0.2rem">⬡ SENTI-RECOMMEND</div>
    <div style="font-size:0.73rem;color:#8b949e;margin-bottom:1.5rem">
    Hybrid AI Tool Discovery Engine</div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown('<div class="section-title">🎚 Hybrid Weights</div>', unsafe_allow_html=True)

    st.sidebar.markdown("""
    <div style="font-size:0.78rem;color:#8b949e;margin-bottom:0.8rem">
    Adjust how much each signal contributes to recommendations.
    Weights are auto-normalised to sum to 1.
    </div>""", unsafe_allow_html=True)

    alpha = st.sidebar.slider("α — CF Weight",      0.0, 1.0, 0.50, 0.05,
                               help="Collaborative filtering: what similar users liked")
    beta  = st.sidebar.slider("β — NLP Weight",     0.0, 1.0, 0.35, 0.05,
                               help="Sentiment analysis: how users feel about the app")
    gamma = st.sidebar.slider("γ — Content Weight", 0.0, 1.0, 0.15, 0.05,
                               help="Content-based: app description similarity")

    total = alpha + beta + gamma
    if total > 0:
        an, bn, gn = alpha/total, beta/total, gamma/total
    else:
        an = bn = gn = 1/3
    st.sidebar.markdown(f"""
    <div style="font-size:0.75rem;color:#8b949e;
    background:#0d1117;padding:8px 10px;border-radius:8px;font-family:monospace">
    Normalised: α={an:.2f} · β={bn:.2f} · γ={gn:.2f}
    </div>""", unsafe_allow_html=True)

    st.sidebar.markdown('<br><div class="section-title">🔍 Filters</div>', unsafe_allow_html=True)

    cats = ["All Categories"] + sorted(models['metadata']['project_category'].dropna().unique().tolist())
    category = st.sidebar.selectbox("AI Category", cats)

    pricing_opts = ["All"] + sorted(models['metadata']['pricing_model'].dropna().unique().tolist())
    pricing = st.sidebar.selectbox("Pricing Model", pricing_opts)

    free_only = st.sidebar.toggle("Free Apps Only", value=False)
    gem_boost = st.sidebar.slider("💎 Hidden Gem Boost", 0.0, 0.20, 0.05, 0.01,
                                   help="Extra score added to hidden gem apps")
    n_results = st.sidebar.slider("Number of Results", 5, 20, 10, 1)

    st.sidebar.markdown('<br><div class="section-title">ℹ️ System Info</div>', unsafe_allow_html=True)
    meta = models['metadata']
    st.sidebar.markdown(f"""
    <div style="font-size:0.77rem;color:#8b949e;line-height:1.8">
    📱 <b>{len(meta)}</b> AI tools indexed<br>
    💬 <b>{len(models['reviews']):,}</b> reviews analysed<br>
    💎 <b>{meta['is_hidden_gem'].sum()}</b> hidden gems found<br>
    🧠 Sentiment: {models['nlp_source'][:28]}
    </div>""", unsafe_allow_html=True)

    return alpha, beta, gamma, category, pricing, free_only, gem_boost, n_results


# ─────────────────────────────────────────────────────────────────────────────
# TAB: DISCOVER
# ─────────────────────────────────────────────────────────────────────────────

def tab_discover(models, alpha, beta, gamma, category, pricing, free_only, gem_boost, n_results):
    st.markdown('<div class="section-title">🔭 Discover AI Tools</div>', unsafe_allow_html=True)

    col_q, col_btn = st.columns([5, 1])
    with col_q:
        query = st.text_input("", placeholder="Describe what you need  (e.g. 'easy image generator for marketing teams')",
                               label_visibility="collapsed")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("Search →")

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

    # User rating profile
    with st.expander("🧑 Add Your Rating History  (optional — improves CF recommendations)", expanded=False):
        st.markdown('<div style="font-size:0.82rem;color:#8b949e;margin-bottom:0.8rem">'
                    'Tell the engine which apps you\'ve already used and how you rated them.</div>',
                    unsafe_allow_html=True)
        all_app_names = sorted(models['metadata']['app_name'].dropna().unique().tolist())
        app_name_to_id = dict(zip(models['metadata']['app_name'], models['metadata']['app_id']))

        c1, c2, c3, c4 = st.columns(4)
        user_ratings = {}
        for i, col in enumerate([c1, c2, c3, c4], 1):
            with col:
                picked = st.selectbox(f"App {i}", ["— none —"] + all_app_names,
                                       key=f"urec_app_{i}", label_visibility="visible")
                rating = st.select_slider(f"Rating {i}", [1,2,3,4,5], value=4,
                                          key=f"urec_r_{i}", label_visibility="collapsed")
                if picked != "— none —":
                    aid = app_name_to_id.get(picked)
                    if aid:
                        user_ratings[aid] = float(rating)

    # Run recommendation
    if search_clicked or query or user_ratings:
        with st.spinner("Computing hybrid recommendations…"):
            recs = hybrid_recommend(
                models, user_ratings=user_ratings if user_ratings else None,
                query_text=query if query.strip() else None,
                n=n_results, category=category, free_only=free_only,
                pricing_filter=pricing if pricing != "All" else None,
                alpha=alpha, beta=beta, gamma=gamma, gem_boost=gem_boost
            )

        if len(recs) == 0:
            st.warning("No results matched your filters. Try broadening the search.")
            return
    # Out-of-scope query warning
    if query and query.strip():
        CATEGORIES = [
            'generative text', 'chatbot', 'image generation',
            'coding', 'productivity', 'marketing', 'writing',
            'photo', 'video', 'pdf', 'email', 'voice', 'chat',
            'ai tool', 'generator', 'assistant'
        ]
        query_lower = query.lower()
        in_scope = any(kw in query_lower for kw in CATEGORIES)
        if not in_scope and recs['content_score'].max() < 0.10:
            st.warning(
                "⚠️ **Query outside dataset scope.** "
                "No dedicated matches found for this tool type. "
                "Showing highest-rated apps from our catalogue instead.\n\n"
                "**Dataset covers:** Generative Text · Image Generation · "
                "AI Coding · Productivity AI · Marketing AI"
            )
        # Summary metrics
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-tile">
            <div class="metric-val">{len(recs)}</div>
            <div class="metric-lbl">Recommendations</div>
          </div>
          <div class="metric-tile">
            <div class="metric-val">{recs['avg_rating'].mean():.2f}</div>
            <div class="metric-lbl">Avg Star Rating</div>
          </div>
          <div class="metric-tile">
            <div class="metric-val">{recs['hybrid_score'].mean():.3f}</div>
            <div class="metric-lbl">Avg Hybrid Score</div>
          </div>
          <div class="metric-tile">
            <div class="metric-val">{recs['is_hidden_gem'].sum()}</div>
            <div class="metric-lbl">💎 Hidden Gems</div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Show toggle
        show_scores = st.toggle("Show score breakdown", value=True)

        col_a, col_b = st.columns(2)
        for i, row in recs.iterrows():
            with col_a if i % 2 == 0 else col_b:
                render_rec_card(row, show_scores=show_scores)

    else:
        # Landing state
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#8b949e">
          <div style="font-size:2.5rem;margin-bottom:1rem">🔍</div>
          <div style="font-family:'Space Mono',monospace;font-size:1rem;color:#e6edf3;margin-bottom:0.5rem">
            Start discovering AI tools
          </div>
          <div style="font-size:0.85rem">
            Type a query above, add your rating history, or just hit <b>Search →</b>
            to see top-ranked recommendations using the current filters.
          </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB: APP DETAIL
# ─────────────────────────────────────────────────────────────────────────────

def tab_app_detail(models):
    st.markdown('<div class="section-title">📱 App Deep-Dive</div>', unsafe_allow_html=True)

    meta = models['metadata']
    app_names = sorted(meta['app_name'].dropna().unique().tolist())
    app_name_to_id = dict(zip(meta['app_name'], meta['app_id']))

    selected_name = st.selectbox("Choose an AI tool to inspect", app_names)
    if not selected_name:
        return

    app_id = app_name_to_id.get(selected_name)
    row    = meta[meta['app_id'] == app_id].iloc[0]
    absa   = models['app_absa'].get(app_id, {a: 0.0 for a in ['Ease of Use','Pricing','Customer Support','Performance','Features']})

    # ── Header ────────────────────────────────────────────────────────────────
    gem_label = "  💎 Hidden Gem" if row.get('is_hidden_gem', 0) else ""
    st.markdown(f"""
    <div class="banner">
      <h1>{''.join(c for c in str(row['app_name']) if ord(c) < 65536)}{gem_label}</h1>
      <p>{str(row.get('developer',''))[:60]} &nbsp;·&nbsp; {row.get('project_category','')}
         &nbsp;·&nbsp; {row.get('pricing_model','')}</p>
    </div>""", unsafe_allow_html=True)

    # ── Metrics row ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-tile">
        <div class="metric-val" style="font-size:1.3rem">{row.get('avg_rating',0):.2f} ★</div>
        <div class="metric-lbl">Avg Rating</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val">{fmt_number(row.get('total_ratings',0))}</div>
        <div class="metric-lbl">Total Ratings</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val">{fmt_number(row.get('total_reviews',0))}</div>
        <div class="metric-lbl">Written Reviews</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val">{fmt_number(row.get('installs',0))}</div>
        <div class="metric-lbl">Installs</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val">{row.get('bayesian_avg',0):.2f}</div>
        <div class="metric-lbl">Bayesian Avg</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([1.3, 1.4, 1.3])

    with c1:
        st.markdown("**Rating Distribution**")
        donut_fig = make_rating_donut(row, meta)
        if donut_fig:
            st.pyplot(donut_fig, use_container_width=False)
        plt.close('all')

    with c2:
        st.markdown("**Aspect-Based Sentiment (ABSA)**")
        if any(v != 0 for v in absa.values()):
            radar_fig = make_absa_radar(absa)
            st.pyplot(radar_fig, use_container_width=False)
            plt.close('all')
        else:
            st.info("Not enough review data for ABSA analysis.")

    with c3:
        st.markdown("**Sentiment Scores**")
        score_row = {
            'cf_score':      float(models['metadata'][models['metadata']['app_id']==app_id]['sentiment_norm'].values[0]) if 'sentiment_norm' in models['metadata'].columns else 0.5,
            'nlp_score':     float(models['sentiment_lookup'].get(app_id, 0.5)),
            'content_score': 0.5,
            'hybrid_score':  float(models['sentiment_lookup'].get(app_id, 0.5)),
        }
        score_fig = make_score_chart(score_row)
        st.pyplot(score_fig, use_container_width=False)
        plt.close('all')

    # ── Description ───────────────────────────────────────────────────────────
    with st.expander("📝 Full Description", expanded=False):
        desc = str(row.get('description', 'No description available.'))
        st.markdown(f'<div style="font-size:0.87rem;color:#c9d1d9;line-height:1.7">{desc[:2000]}</div>',
                    unsafe_allow_html=True)

    # ── ABSA breakdown table ──────────────────────────────────────────────────
    with st.expander("🎯 ABSA Breakdown — Aspect Sentiment Details", expanded=True):
        absa_df = pd.DataFrame([
            {
                'Aspect': aspect,
                'Score':  round(score, 3),
                'Sentiment': '🟢 Positive' if score > 0.15 else ('🔴 Negative' if score < -0.15 else '🟡 Neutral'),
                'Strength': '▓▓▓▓' if abs(score) > 0.6 else ('▓▓▓░' if abs(score) > 0.35 else ('▓▓░░' if abs(score) > 0.15 else '▓░░░'))
            }
            for aspect, score in absa.items()
        ])
        st.dataframe(absa_df, use_container_width=True, hide_index=True)

    # ── Similar apps ─────────────────────────────────────────────────────────
    st.markdown('<div style="margin-top:1.5rem"><b>🔗 Similar Apps</b></div>', unsafe_allow_html=True)
    sim_df = find_similar_apps(models, app_id, n=4)
    if len(sim_df) > 0:
        cols = st.columns(4)
        for i, (_, sim_row) in enumerate(sim_df.iterrows()):
            with cols[i % 4]:
                st.markdown(f"""
                <div class="card" style="padding:0.8rem">
                  <div style="font-size:0.78rem;font-weight:700;
                  color:#e6edf3;margin-bottom:0.2rem">{sim_row['app_name'][:28]}</div>
                  <div style="font-size:0.72rem;color:#8b949e">{sim_row['category']}</div>
                  <div style="font-size:0.75rem;color:#2dd4bf;margin-top:0.3rem">
                  sim: {sim_row['similarity']:.3f}</div>
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB: HIDDEN GEMS
# ─────────────────────────────────────────────────────────────────────────────

def tab_hidden_gems(models, alpha, beta, gamma, gem_boost, n_results):
    st.markdown('<div class="section-title">💎 Hidden Gems — Discover Underrated AI Tools</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.87rem;color:#8b949e;
    background:#161b22;border:1px solid #30363d;
    border-radius:12px;padding:1rem 1.4rem;margin-bottom:1.5rem">
    <b style="color:#2dd4bf">What is a Hidden Gem?</b><br>
    An app with a <b>high sentiment score</b> (users genuinely love it)
    but a <b>low total review count</b> (not yet discovered by the mainstream).
    These are the underrated tools your project proposal specifically targets.
    </div>""", unsafe_allow_html=True)

    # Use NLP-heavy weights to surface gems
    recs = hybrid_recommend(
        models,
        n=n_results,
        alpha=max(0.1, alpha * 0.5),
        beta=min(0.9, beta * 1.5),
        gamma=gamma,
        gem_boost=max(gem_boost, 0.10),  # Extra gem boost
    )

    if len(recs) == 0:
        st.warning("No gems found. Try adjusting filters.")
        return

    gems = recs[recs['is_hidden_gem'] == 1]
    all_results = recs

    # Metrics
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-tile">
        <div class="metric-val">{len(gems)}</div>
        <div class="metric-lbl">Hidden Gems in Results</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val">{models['metadata']['is_hidden_gem'].sum()}</div>
        <div class="metric-lbl">Total Gems in Catalogue</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val">{f"{gems['avg_rating'].mean():.2f}" if len(gems) > 0 else 'N/A'}</div>
        <div class="metric-lbl">Avg Rating of Gems</div>
      </div>
    </div>""", unsafe_allow_html=True)

    if len(gems) > 0:
        st.markdown("### 💎 Confirmed Hidden Gems")
        c1, c2 = st.columns(2)
        for i, (_, row) in enumerate(gems.iterrows()):
            with c1 if i % 2 == 0 else c2:
                render_rec_card(row, show_scores=True)

    st.markdown("### All Results (Gem-Weighted)")
    non_gems = all_results[all_results['is_hidden_gem'] == 0]
    if len(non_gems) > 0:
        c1, c2 = st.columns(2)
        for i, (_, row) in enumerate(non_gems.iterrows()):
            with c1 if i % 2 == 0 else c2:
                render_rec_card(row, show_scores=False)


# ─────────────────────────────────────────────────────────────────────────────
# TAB: EXPLORE
# ─────────────────────────────────────────────────────────────────────────────

def tab_explore(models):
    st.markdown('<div class="section-title">🗺 Explore the AI Tool Landscape</div>',
                unsafe_allow_html=True)

    meta    = models['metadata']
    reviews = models['reviews']

    # ── Overview metrics ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-tile">
        <div class="metric-val">{len(meta)}</div>
        <div class="metric-lbl">AI Tools Indexed</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val">{len(reviews):,}</div>
        <div class="metric-lbl">User Reviews</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val">{meta['project_category'].nunique()}</div>
        <div class="metric-lbl">AI Categories</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val">{meta['avg_rating'].mean():.2f}★</div>
        <div class="metric-lbl">Avg Rating (Global)</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val">{meta['is_hidden_gem'].sum()}</div>
        <div class="metric-lbl">Hidden Gems</div>
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Apps per Category**")
        cat_fig = make_category_dist_chart(meta)
        st.pyplot(cat_fig, use_container_width=True)
        plt.close('all')

    with c2:
        st.markdown("**Rating Distribution Across All Apps**")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        fig2.patch.set_facecolor('#0d1117')
        ax2.set_facecolor('#0d1117')
        ax2.hist(meta['avg_rating'].dropna(), bins=30,
                 color='#2dd4bf', edgecolor='#0d1117', alpha=0.85)
        ax2.set_xlabel("Average Rating", color='#8b949e', fontsize=9)
        ax2.set_ylabel("Number of Apps", color='#8b949e', fontsize=9)
        ax2.tick_params(colors='#8b949e', labelsize=9)
        ax2.spines[['top','right','bottom','left']].set_color('#30363d')
        ax2.axvline(meta['avg_rating'].mean(), color='#fbbf24',
                    linestyle='--', linewidth=1.5,
                    label=f"Mean: {meta['avg_rating'].mean():.2f}")
        ax2.legend(fontsize=8, facecolor='#161b22', labelcolor='#8b949e')
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close('all')

    # ── Sentiment vs Rating Scatter ────────────────────────────────────────────
    st.markdown("**Sentiment Score vs Star Rating  (each dot = one AI tool)**")
    fig3, ax3 = plt.subplots(figsize=(12, 4.5))
    fig3.patch.set_facecolor('#0d1117')
    ax3.set_facecolor('#0d1117')

    cats     = meta['project_category'].dropna().unique()
    palette  = {'Generative Text / Chatbots': '#2dd4bf',
                 'Image Generation':          '#3b82f6',
                 'AI Coding Assistants':      '#a78bfa',
                 'Productivity AI':           '#f59e0b',
                 'Marketing AI':              '#fb7185'}

    for cat in cats:
        sub = meta[meta['project_category'] == cat]
        ax3.scatter(sub['avg_rating'], sub['mean_sentiment_score'],
                    color=palette.get(cat, '#8b949e'),
                    s=80, alpha=0.7, edgecolors='none', label=cat)

    # Mark hidden gems
    gems = meta[meta['is_hidden_gem'] == 1]
    if len(gems) > 0:
        ax3.scatter(gems['avg_rating'], gems['mean_sentiment_score'],
                    color='none', edgecolors='#fbbf24', s=120, linewidths=1.5,
                    label='💎 Hidden Gem', zorder=5)

    ax3.set_xlabel("Average Star Rating", color='#8b949e', fontsize=10)
    ax3.set_ylabel("NLP Sentiment Score", color='#8b949e', fontsize=10)
    ax3.tick_params(colors='#8b949e', labelsize=9)
    ax3.spines[['top','right','bottom','left']].set_color('#30363d')
    ax3.legend(fontsize=8, facecolor='#161b22', labelcolor='#e6edf3',
               loc='lower right', framealpha=0.9)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close('all')

    # ── Top 10 by category ────────────────────────────────────────────────────
    st.markdown('<div style="margin-top:1.5rem"><b>Top 3 Apps per Category (by Bayesian Average)</b></div>',
                unsafe_allow_html=True)
    cols = st.columns(5)
    for col_idx, cat in enumerate(sorted(cats)):
        sub = meta[meta['project_category'] == cat].nlargest(3, 'bayesian_avg')
        with cols[col_idx % 5]:
            st.markdown(f'<div style="font-size:0.72rem;color:#2dd4bf;'
                        f'font-family:monospace;margin-bottom:0.4rem">{cat[:20]}</div>',
                        unsafe_allow_html=True)
            for _, r in sub.iterrows():
                st.markdown(f"""
                <div style="font-size:0.78rem;padding:4px 0;
                border-bottom:1px solid #30363d;color:#c9d1d9">
                {str(r['app_name'])[:26]}<br>
                <span style="color:#fbbf24;font-size:0.72rem">
                ★ {r['bayesian_avg']:.2f}</span>
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB: ABOUT
# ─────────────────────────────────────────────────────────────────────────────

def tab_about(models):
    st.markdown('<div class="section-title">📖 About This System</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("""
## Senti-Recommend

A hybrid AI tool discovery engine combining **Collaborative Filtering**
and **Sentiment-Driven Review Analysis** — built as a PGD Data Science with AI final project.

### System Architecture

```
User Input (query / ratings / filters)
         │
         ├── Item-Based CF ──────────┐
         │   Cosine similarity on    │
         │   user-item matrix        │
         │                           │
         ├── SVD Latent Factors ─────┤──▶ Weighted Fusion ──▶ Ranked Results
         │   Matrix factorisation    │       α·CF + β·NLP + γ·Content
         │                           │
         ├── NLP Sentiment Score ────┤
         │   Logistic Regression     │
         │   + TF-IDF (NB02)         │
         │                           │
         └── Content-Based TF-IDF ───┘
             App description similarity
             (cold-start safety net)
```

### Hybrid Scoring Formula
```
Hybrid Score = α × CF_Score
             + β × Sentiment_Score
             + γ × Content_Score
             + GemBoost × is_hidden_gem
```

### Hidden Gem Definition
An app qualifies as a Hidden Gem if it has:
- A **Bayesian average** ≥ 55th percentile (users who DO rate it, love it)
- A **total review count** ≤ 30th percentile (not yet widely discovered)

These are the "underrated tools" the system is specifically designed to surface.

### Aspect-Based Sentiment Analysis (ABSA)
Beyond a single positive/negative score, ABSA detects sentiment  
across **5 specific dimensions** per app:
- 🟢 Ease of Use · 💰 Pricing · 🛠️ Customer Support
- ⚡ Performance · 🔧 Features
""")

    with c2:
        st.markdown("### Project Notebooks")
        for nb_num, nb_title, nb_desc in [
            ("01", "Data Acquisition & Cleaning", "Raw data ingestion, quality checks, feature engineering"),
            ("02", "NLP Sentiment Analysis",       "TF-IDF + Logistic Regression, ABSA, sentiment scoring"),
            ("03", "CF + Hybrid Recommender",      "Item-CF, SVD, content-based, hybrid fusion, evaluation"),
            ("04", "Streamlit Application",        "This interactive web dashboard"),
        ]:
            st.markdown(f"""
            <div class="card" style="padding:0.9rem">
              <div style="font-family:'Space Mono',monospace;
              font-size:0.7rem;color:#2dd4bf">NB{nb_num}</div>
              <div style="font-size:0.88rem;font-weight:600;
              color:#e6edf3;margin:0.2rem 0">{nb_title}</div>
              <div style="font-size:0.78rem;color:#8b949e">{nb_desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("### Key Libraries")
        for lib, purpose in [
            ("scikit-learn", "TF-IDF, SVD, Logistic Regression, cosine similarity"),
            ("pandas / numpy", "Data manipulation and matrix operations"),
            ("scipy.sparse", "Efficient sparse matrix storage"),
            ("Streamlit", "Interactive web dashboard"),
            ("matplotlib / seaborn", "Data visualisation"),
        ]:
            st.markdown(f"""
            <div style="font-size:0.8rem;padding:5px 0;
            border-bottom:1px solid #30363d;color:#c9d1d9">
            <b style="color:#2dd4bf">{lib}</b><br>
            <span style="color:#8b949e">{purpose}</span>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load models ───────────────────────────────────────────────────────────
    with st.spinner("⚙️  Building recommendation models from data…  (first load only — ~30 sec)"):
        models = build_all_models()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    alpha, beta, gamma, category, pricing, free_only, gem_boost, n_results = render_sidebar(models)

    # ── Main banner ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="banner">
      <h1>🔍 Senti-Recommend</h1>
      <p>Hybrid AI Tool Discovery Engine &nbsp;·&nbsp;
         Collaborative Filtering + Sentiment Analysis &nbsp;·&nbsp;
         PGD Data Science with AI</p>
    </div>""", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔭  Discover",
        "📱  App Detail",
        "💎  Hidden Gems",
        "🗺  Explore",
        "📖  About",
    ])

    with tab1:
        tab_discover(models, alpha, beta, gamma, category, pricing,
                     free_only, gem_boost, n_results)

    with tab2:
        tab_app_detail(models)

    with tab3:
        tab_hidden_gems(models, alpha, beta, gamma, gem_boost, n_results)

    with tab4:
        tab_explore(models)

    with tab5:
        tab_about(models)


if __name__ == "__main__":
    main()
