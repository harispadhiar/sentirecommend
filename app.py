"""
╔══════════════════════════════════════════════════════════════════╗
║   SENTI-RECOMMEND — Hybrid AI Tool Discovery Engine             ║
║   PGD Data Science with AI — Final Project                      ║
║   app.py  (src-based entry point)                               ║
╚══════════════════════════════════════════════════════════════════╝

Run with:
    streamlit run app.py

Project structure:
    app.py          ← this file (UI only)
    src/
        config.py       ← all constants and settings
        utils.py        ← small helper functions
        data_loader.py  ← CSV loading and preprocessing
        models.py       ← CF, SVD, TF-IDF model building
        recommender.py  ← hybrid fusion engine
        absa.py         ← aspect-based sentiment analysis
        charts.py       ← all matplotlib chart functions
"""

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SRC IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
from src.config import (
    APP_TITLE, APP_ICON, APP_SUBTITLE,
    DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_GAMMA, DEFAULT_GEM_BOOST,
    PROJECT_CATEGORIES,
)
from src.utils import star_str, fmt_number, score_bar_html, truncate
from src.data_loader import load_all_data, build_sentiment_lookup, build_hidden_gem_lookup
from src.models import build_all_models
from src.recommender import hybrid_recommend, find_similar_apps, is_out_of_scope
from src.absa import compute_app_absa, absa_to_dataframe
from src.charts import (
    make_score_chart,
    make_absa_radar,
    make_rating_donut,
    make_category_dist_chart,
    make_rating_histogram,
    make_sentiment_scatter,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
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
.card-meta { font-size: 0.82rem; color: var(--muted); margin-bottom: 0.6rem; }

.score-bar-wrap { margin-top: 0.5rem; }
.score-label {
    font-size: 0.75rem; color: var(--muted);
    display: flex; justify-content: space-between; margin-bottom: 2px;
}
.score-bar-bg {
    background: var(--border); border-radius: 4px;
    height: 6px; margin-bottom: 5px; overflow: hidden;
}
.score-bar-fill { height: 100%; border-radius: 4px; }

.badge {
    display: inline-block; font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    padding: 2px 8px; border-radius: 20px;
    margin-right: 4px; font-weight: 700;
}
.badge-cat  { background: rgba(45,212,191,0.15); color: var(--teal); }
.badge-free { background: rgba(251,191,36,0.15);  color: var(--amber); }
.badge-gem  { background: rgba(251,113,133,0.15); color: var(--rose); }

.metric-row { display: flex; gap: 1rem; margin-bottom: 1.4rem; }
.metric-tile {
    flex: 1; background: var(--surface);
    border: 1px solid var(--border); border-radius: var(--radius);
    padding: 1rem 1.2rem; text-align: center;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem; font-weight: 700; color: var(--teal);
}
.metric-lbl { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }

.banner {
    background: linear-gradient(135deg, #0d2330 0%, #0d1117 50%, #1a0d20 100%);
    border: 1px solid var(--border); border-radius: var(--radius);
    padding: 2rem 2.5rem; margin-bottom: 2rem;
    position: relative; overflow: hidden;
}
.banner::before {
    content: ''; position: absolute; top:0; left:0; right:0; bottom:0;
    background: repeating-linear-gradient(
        45deg, transparent, transparent 40px,
        rgba(45,212,191,0.02) 40px, rgba(45,212,191,0.02) 80px
    );
}
.banner h1 { font-size: 2rem; margin: 0 0 0.3rem 0; }
.banner p  { color: var(--muted); margin: 0; font-size: 0.9rem; }

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem; letter-spacing: 0.15em;
    text-transform: uppercase; color: var(--teal);
    margin-bottom: 1rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}
[data-testid="stTab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}
hr { border-color: var(--border) !important; }
.stars { color: var(--amber); font-size: 0.9rem; letter-spacing: 1px; }
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (cached — runs once per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_everything():
    """
    Full startup pipeline — loads data, builds all models, computes ABSA.
    Cached by Streamlit so this only runs once per session.
    """
    # 1. Load and preprocess data
    reviews, metadata, nlp_source = load_all_data()

    # 2. Build ML models
    model_components = build_all_models(reviews, metadata)

    # 3. Build lookup dicts
    sentiment_lookup   = build_sentiment_lookup(metadata)
    hidden_gem_lookup  = build_hidden_gem_lookup(metadata)

    # 4. ABSA per app
    app_absa = compute_app_absa(reviews)

    return {
        "reviews":          reviews,
        "metadata":         metadata,
        "nlp_source":       nlp_source,
        "sentiment_lookup": sentiment_lookup,
        "hidden_gem_lookup":hidden_gem_lookup,
        "app_absa":         app_absa,
        **model_components,   # app_ids, user_ids, app_to_idx,
                               # item_sim, app_factors, svd_sim,
                               # content_tfidf, content_mat, content_sim,
                               # meta_app_to_idx
    }


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION WRAPPER  (convenience shim for UI tabs)
# ─────────────────────────────────────────────────────────────────────────────

def recommend(state, user_ratings=None, query_text=None, n=10,
              category=None, free_only=False, pricing_filter=None,
              alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA,
              gamma=DEFAULT_GAMMA, gem_boost=DEFAULT_GEM_BOOST):
    """Thin wrapper that passes state dict to src.recommender.hybrid_recommend."""
    return hybrid_recommend(
        models           = state,
        metadata_df      = state["metadata"],
        sentiment_lookup = state["sentiment_lookup"],
        hidden_gem_lookup= state["hidden_gem_lookup"],
        user_ratings     = user_ratings,
        query_text       = query_text,
        n                = n,
        category         = category,
        free_only        = free_only,
        pricing_filter   = pricing_filter,
        alpha            = alpha,
        beta             = beta,
        gamma            = gamma,
        gem_boost        = gem_boost,
    )


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def render_rec_card(row, show_scores=True):
    """Render a single recommendation result card."""
    gem_badge  = '<span class="badge badge-gem">💎 Hidden Gem</span>' if row["is_hidden_gem"] else ""
    free_badge = '<span class="badge badge-free">Free</span>' if "Free" in str(row["pricing"]) else ""
    cat_badge  = f'<span class="badge badge-cat">{row["category"]}</span>'

    scores_html = ""
    if show_scores:
        scores_html = (
            score_bar_html("CF Score",      row["cf_score"],      "#3b82f6") +
            score_bar_html("NLP Sentiment", row["nlp_score"],     "#2dd4bf") +
            score_bar_html("Content Score", row["content_score"], "#f59e0b") +
            score_bar_html("Hybrid Score",  row["hybrid_score"],  "#a78bfa")
        )

    st.markdown(f"""
    <div class="card">
      <div class="card-rank">#{row['rank']:02d}</div>
      <div class="card-title">{row['app_name']}</div>
      <div class="card-meta">{cat_badge} {free_badge} {gem_badge}</div>
      <div class="card-meta">
        <span class="stars">{star_str(row['avg_rating'])}</span>
        &nbsp;·&nbsp; {fmt_number(row['total_reviews'])} reviews
        &nbsp;·&nbsp; {fmt_number(row['installs'])} installs
      </div>
      <div style="font-size:0.82rem;color:#8b949e;margin-bottom:0.6rem">
        {truncate(row.get('summary',''), 180)}
      </div>
      {scores_html}
    </div>
    """, unsafe_allow_html=True)


def render_two_col_cards(df, show_scores=True):
    """Render a DataFrame of recommendations in a two-column card grid."""
    col_a, col_b = st.columns(2)
    for i, row in df.iterrows():
        with col_a if i % 2 == 0 else col_b:
            render_rec_card(row, show_scores=show_scores)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(state):
    """Render the sidebar controls and return selected values."""
    st.sidebar.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:1.1rem;
    font-weight:700;color:#2dd4bf;margin-bottom:0.2rem">⬡ SENTI-RECOMMEND</div>
    <div style="font-size:0.73rem;color:#8b949e;margin-bottom:1.5rem">
    Hybrid AI Tool Discovery Engine</div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown('<div class="section-title">🎚 Hybrid Weights</div>',
                        unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="font-size:0.78rem;color:#8b949e;margin-bottom:0.8rem">
    Adjust how much each signal contributes.
    Weights are auto-normalised to sum to 1.
    </div>""", unsafe_allow_html=True)

    alpha = st.sidebar.slider("α — CF Weight",      0.0, 1.0, DEFAULT_ALPHA, 0.05,
                               help="Collaborative filtering: what similar users liked")
    beta  = st.sidebar.slider("β — NLP Weight",     0.0, 1.0, DEFAULT_BETA,  0.05,
                               help="Sentiment analysis: how users feel about the app")
    gamma = st.sidebar.slider("γ — Content Weight", 0.0, 1.0, DEFAULT_GAMMA, 0.05,
                               help="Content-based: app description similarity")

    total = alpha + beta + gamma
    an, bn, gn = (alpha/total, beta/total, gamma/total) if total > 0 else (1/3, 1/3, 1/3)
    st.sidebar.markdown(f"""
    <div style="font-size:0.75rem;color:#8b949e;
    background:#0d1117;padding:8px 10px;border-radius:8px;font-family:monospace">
    Normalised: α={an:.2f} · β={bn:.2f} · γ={gn:.2f}
    </div>""", unsafe_allow_html=True)

    st.sidebar.markdown('<br><div class="section-title">🔍 Filters</div>',
                        unsafe_allow_html=True)

    meta = state["metadata"]
    cats = ["All Categories"] + sorted(meta["project_category"].dropna().unique().tolist())
    category = st.sidebar.selectbox("AI Category", cats)

    pricing_opts = ["All"] + sorted(meta["pricing_model"].dropna().unique().tolist())
    pricing  = st.sidebar.selectbox("Pricing Model", pricing_opts)
    free_only = st.sidebar.toggle("Free Apps Only", value=False)

    gem_boost = st.sidebar.slider("💎 Hidden Gem Boost", 0.0, 0.20,
                                   DEFAULT_GEM_BOOST, 0.01,
                                   help="Extra score bonus added to hidden gem apps")
    n_results = st.sidebar.slider("Number of Results", 5, 20, 10, 1)

    st.sidebar.markdown('<br><div class="section-title">ℹ️ System Info</div>',
                        unsafe_allow_html=True)
    st.sidebar.markdown(f"""
    <div style="font-size:0.77rem;color:#8b949e;line-height:1.8">
    📱 <b>{len(meta)}</b> AI tools indexed<br>
    💬 <b>{len(state['reviews']):,}</b> reviews analysed<br>
    💎 <b>{meta['is_hidden_gem'].sum()}</b> hidden gems found<br>
    🧠 {state['nlp_source'][:35]}
    </div>""", unsafe_allow_html=True)

    return alpha, beta, gamma, category, pricing, free_only, gem_boost, n_results


# ─────────────────────────────────────────────────────────────────────────────
# TAB: DISCOVER
# ─────────────────────────────────────────────────────────────────────────────

def tab_discover(state, alpha, beta, gamma, category, pricing,
                 free_only, gem_boost, n_results):
    st.markdown('<div class="section-title">🔭 Discover AI Tools</div>',
                unsafe_allow_html=True)

    col_q, col_btn = st.columns([5, 1])
    with col_q:
        query = st.text_input(
            "", label_visibility="collapsed",
            placeholder="Describe what you need  (e.g. 'easy image generator for marketing teams')"
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("Search →")

    # Optional user rating history
    with st.expander("🧑 Add Your Rating History  (optional — improves CF recommendations)",
                     expanded=False):
        st.markdown('<div style="font-size:0.82rem;color:#8b949e;margin-bottom:0.8rem">'
                    "Tell the engine which apps you've already used and how you rated them."
                    "</div>", unsafe_allow_html=True)
        meta = state["metadata"]
        all_app_names  = sorted(meta["app_name"].dropna().unique().tolist())
        app_name_to_id = dict(zip(meta["app_name"], meta["app_id"]))

        c1, c2, c3, c4 = st.columns(4)
        user_ratings = {}
        for i, col in enumerate([c1, c2, c3, c4], 1):
            with col:
                picked = st.selectbox(f"App {i}", ["— none —"] + all_app_names,
                                      key=f"urec_app_{i}")
                rating = st.select_slider(f"Rating {i}", [1, 2, 3, 4, 5], value=4,
                                          key=f"urec_r_{i}", label_visibility="collapsed")
                if picked != "— none —":
                    aid = app_name_to_id.get(picked)
                    if aid:
                        user_ratings[aid] = float(rating)

    # Run recommendations
    if search_clicked or query or user_ratings:
        with st.spinner("Computing hybrid recommendations…"):
            recs = recommend(
                state,
                user_ratings   = user_ratings if user_ratings else None,
                query_text     = query if query.strip() else None,
                n              = n_results,
                category       = category,
                free_only      = free_only,
                pricing_filter = pricing if pricing != "All" else None,
                alpha=alpha, beta=beta, gamma=gamma, gem_boost=gem_boost,
            )

        if len(recs) == 0:
            st.warning("No results matched your filters. Try broadening the search.")
            return

        # Out-of-scope warning
        if is_out_of_scope(query, recs):
            st.warning(
                "⚠️ **Query outside dataset scope.** "
                "No close matches found for this tool type. "
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

        show_scores = st.toggle("Show score breakdown", value=True)
        render_two_col_cards(recs, show_scores=show_scores)

    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#8b949e">
          <div style="font-size:2.5rem;margin-bottom:1rem">🔍</div>
          <div style="font-family:'Space Mono',monospace;font-size:1rem;
          color:#e6edf3;margin-bottom:0.5rem">Start discovering AI tools</div>
          <div style="font-size:0.85rem">
            Type a query above, add your rating history, or just hit
            <b>Search →</b> to see top-ranked recommendations.
          </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB: APP DETAIL
# ─────────────────────────────────────────────────────────────────────────────

def tab_app_detail(state):
    st.markdown('<div class="section-title">📱 App Deep-Dive</div>',
                unsafe_allow_html=True)

    meta = state["metadata"]
    app_names      = sorted(meta["app_name"].dropna().unique().tolist())
    app_name_to_id = dict(zip(meta["app_name"], meta["app_id"]))

    selected_name = st.selectbox("Choose an AI tool to inspect", app_names)
    if not selected_name:
        return

    app_id  = app_name_to_id.get(selected_name)
    app_row = meta[meta["app_id"] == app_id].iloc[0]
    absa    = state["app_absa"].get(
        app_id,
        {a: 0.0 for a in ["Ease of Use", "Pricing", "Customer Support", "Performance", "Features"]}
    )

    # Header banner
    gem_label = "  💎 Hidden Gem" if app_row.get("is_hidden_gem", 0) else ""
    st.markdown(f"""
    <div class="banner">
      <h1>{''.join(c for c in str(app_row['app_name']) if ord(c) < 65536)}{gem_label}</h1>
      <p>{str(app_row.get('developer',''))[:60]} &nbsp;·&nbsp;
         {app_row.get('project_category','')} &nbsp;·&nbsp;
         {app_row.get('pricing_model','')}</p>
    </div>""", unsafe_allow_html=True)

    # Key metrics
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-tile">
        <div class="metric-val" style="font-size:1.3rem">{app_row.get('avg_rating',0):.2f} ★</div>
        <div class="metric-lbl">Avg Rating</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val">{fmt_number(app_row.get('total_ratings',0))}</div>
        <div class="metric-lbl">Total Ratings</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val">{fmt_number(app_row.get('total_reviews',0))}</div>
        <div class="metric-lbl">Written Reviews</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val">{fmt_number(app_row.get('installs',0))}</div>
        <div class="metric-lbl">Installs</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val">{app_row.get('bayesian_avg',0):.2f}</div>
        <div class="metric-lbl">Bayesian Avg</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Charts
    c1, c2, c3 = st.columns([1.3, 1.4, 1.3])

    with c1:
        st.markdown("**Rating Distribution**")
        donut_fig = make_rating_donut(app_row)
        if donut_fig:
            st.pyplot(donut_fig, use_container_width=False)
        plt.close("all")

    with c2:
        st.markdown("**Aspect-Based Sentiment (ABSA)**")
        if any(v != 0 for v in absa.values()):
            st.pyplot(make_absa_radar(absa), use_container_width=False)
        else:
            st.info("Not enough review data for ABSA analysis.")
        plt.close("all")

    with c3:
        st.markdown("**Sentiment Score Breakdown**")
        nlp_s = float(state["sentiment_lookup"].get(app_id, 0.5))
        st.pyplot(
            make_score_chart(
                cf_score      = nlp_s,
                nlp_score     = nlp_s,
                content_score = 0.5,
                hybrid_score  = nlp_s,
            ),
            use_container_width=False,
        )
        plt.close("all")

    # Description
    with st.expander("📝 Full Description", expanded=False):
        desc = str(app_row.get("description", "No description available."))
        st.markdown(
            f'<div style="font-size:0.87rem;color:#c9d1d9;line-height:1.7">{desc[:2000]}</div>',
            unsafe_allow_html=True,
        )

    # ABSA table
    with st.expander("🎯 ABSA Breakdown — Aspect Sentiment Details", expanded=True):
        st.dataframe(absa_to_dataframe(absa), use_container_width=True, hide_index=True)

    # Similar apps
    st.markdown('<div style="margin-top:1.5rem"><b>🔗 Similar Apps</b></div>',
                unsafe_allow_html=True)
    sim_df = find_similar_apps(state, state["metadata"], app_id, n=4)
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

def tab_hidden_gems(state, alpha, beta, gamma, gem_boost, n_results):
    st.markdown('<div class="section-title">💎 Hidden Gems — Discover Underrated AI Tools</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.87rem;color:#8b949e;background:#161b22;
    border:1px solid #30363d;border-radius:12px;padding:1rem 1.4rem;margin-bottom:1.5rem">
    <b style="color:#2dd4bf">What is a Hidden Gem?</b><br>
    An app with a <b>high sentiment score</b> (users genuinely love it) but a
    <b>low total review count</b> (not yet discovered by the mainstream).
    These are the underrated tools this project specifically targets.
    </div>""", unsafe_allow_html=True)

    recs = recommend(
        state,
        n         = n_results,
        alpha     = max(0.10, alpha * 0.5),
        beta      = min(0.90, beta  * 1.5),
        gamma     = gamma,
        gem_boost = max(gem_boost, 0.10),
    )

    if len(recs) == 0:
        st.warning("No gems found. Try adjusting filters.")
        return

    gems     = recs[recs["is_hidden_gem"] == 1]
    non_gems = recs[recs["is_hidden_gem"] == 0]

    gem_avg = f"{gems['avg_rating'].mean():.2f}" if len(gems) > 0 else "N/A"

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-tile">
        <div class="metric-val">{len(gems)}</div>
        <div class="metric-lbl">Hidden Gems in Results</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val">{state['metadata']['is_hidden_gem'].sum()}</div>
        <div class="metric-lbl">Total Gems in Catalogue</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val">{gem_avg}</div>
        <div class="metric-lbl">Avg Rating of Gems</div>
      </div>
    </div>""", unsafe_allow_html=True)

    if len(gems) > 0:
        st.markdown("### 💎 Confirmed Hidden Gems")
        render_two_col_cards(gems.reset_index(drop=True), show_scores=True)

    if len(non_gems) > 0:
        st.markdown("### All Results (Gem-Weighted)")
        render_two_col_cards(non_gems.reset_index(drop=True), show_scores=False)


# ─────────────────────────────────────────────────────────────────────────────
# TAB: EXPLORE
# ─────────────────────────────────────────────────────────────────────────────

def tab_explore(state):
    st.markdown('<div class="section-title">🗺 Explore the AI Tool Landscape</div>',
                unsafe_allow_html=True)

    meta    = state["metadata"]
    reviews = state["reviews"]

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
        st.pyplot(make_category_dist_chart(meta), use_container_width=True)
        plt.close("all")

    with c2:
        st.markdown("**Rating Distribution Across All Apps**")
        st.pyplot(make_rating_histogram(meta), use_container_width=True)
        plt.close("all")

    st.markdown("**Sentiment Score vs Star Rating  (each dot = one AI tool)**")
    st.pyplot(make_sentiment_scatter(meta), use_container_width=True)
    plt.close("all")

    # Top 3 per category
    st.markdown('<div style="margin-top:1.5rem"><b>Top 3 Apps per Category (by Bayesian Average)</b></div>',
                unsafe_allow_html=True)
    cats = sorted(meta["project_category"].dropna().unique().tolist())
    cols = st.columns(len(cats))
    for col_idx, cat in enumerate(cats):
        sub = meta[meta["project_category"] == cat].nlargest(3, "bayesian_avg")
        with cols[col_idx]:
            st.markdown(f'<div style="font-size:0.72rem;color:#2dd4bf;'
                        f'font-family:monospace;margin-bottom:0.4rem">{cat[:20]}</div>',
                        unsafe_allow_html=True)
            for _, r in sub.iterrows():
                st.markdown(f"""
                <div style="font-size:0.78rem;padding:4px 0;
                border-bottom:1px solid #30363d;color:#c9d1d9">
                {str(r['app_name'])[:26]}<br>
                <span style="color:#fbbf24;font-size:0.72rem">★ {r['bayesian_avg']:.2f}</span>
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB: ABOUT
# ─────────────────────────────────────────────────────────────────────────────

def tab_about(state):
    st.markdown('<div class="section-title">📖 About This System</div>',
                unsafe_allow_html=True)

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
         │   (cosine similarity)     │
         ├── SVD Latent Factors ─────┤──▶ α·CF + β·NLP + γ·Content ──▶ Ranked Results
         │   (matrix factorisation)  │
         ├── NLP Sentiment Score ────┤
         │   (Notebook 02 / proxy)   │
         └── Content TF-IDF ─────────┘
             (cold-start fallback)
```

### Hybrid Scoring Formula
```
Hybrid Score = α × CF_Score
             + β × Sentiment_Score
             + γ × Content_Score
             + GemBoost × is_hidden_gem
```

### src/ Module Structure
| Module | Responsibility |
|--------|---------------|
| `config.py` | All constants, weights, lexicons |
| `data_loader.py` | CSV loading, sentiment proxy, hidden gems |
| `models.py` | Item-CF, SVD, Content TF-IDF |
| `recommender.py` | Hybrid fusion engine |
| `absa.py` | Aspect-Based Sentiment Analysis |
| `charts.py` | All matplotlib visualisations |
| `utils.py` | Small helper functions |
""")

    with c2:
        st.markdown("### Project Notebooks")
        for nb_num, nb_title, nb_desc in [
            ("01", "Data Acquisition & Cleaning", "Raw data ingestion, quality checks"),
            ("02", "NLP Sentiment Analysis",       "TF-IDF + Logistic Regression, ABSA"),
            ("03", "CF + Hybrid Recommender",      "Item-CF, SVD, evaluation"),
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
            ("scikit-learn",      "SVD, TF-IDF, cosine similarity"),
            ("pandas / numpy",    "Data manipulation, matrix ops"),
            ("scipy.sparse",      "Efficient sparse matrix storage"),
            ("Streamlit",         "Interactive web dashboard"),
            ("matplotlib",        "Data visualisation"),
        ]:
            st.markdown(f"""
            <div style="font-size:0.8rem;padding:5px 0;
            border-bottom:1px solid #30363d;color:#c9d1d9">
            <b style="color:#2dd4bf">{lib}</b><br>
            <span style="color:#8b949e">{purpose}</span>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    with st.spinner("⚙️  Building recommendation models…  (first load only — ~30 sec)"):
        state = load_everything()

    alpha, beta, gamma, category, pricing, free_only, gem_boost, n_results = \
        render_sidebar(state)

    st.markdown(f"""
    <div class="banner">
      <h1>🔍 {APP_TITLE}</h1>
      <p>{APP_SUBTITLE}</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔭  Discover",
        "📱  App Detail",
        "💎  Hidden Gems",
        "🗺  Explore",
        "📖  About",
    ])

    with tab1:
        tab_discover(state, alpha, beta, gamma, category, pricing,
                     free_only, gem_boost, n_results)
    with tab2:
        tab_app_detail(state)
    with tab3:
        tab_hidden_gems(state, alpha, beta, gamma, gem_boost, n_results)
    with tab4:
        tab_explore(state)
    with tab5:
        tab_about(state)


if __name__ == "__main__":
    main()
