"""
recommender.py — Hybrid Recommendation Engine
The core of Senti-Recommend.

Implements:
  - hybrid_recommend()   — main recommendation function
  - find_similar_apps()  — item-CF or SVD based app similarity
  - _get_cf_scores()     — item-CF scoring with SVD fallback
  - _get_content_scores()— TF-IDF query or liked-app scoring

Hybrid Fusion Formula:
  score = α × CF_score + β × NLP_score + γ × content_score + gem_boost × is_gem

  α + β + γ are auto-normalised to sum to 1.0
  When query is present but no user history: switches to query-dominant weights
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_GAMMA,
    DEFAULT_GEM_BOOST, QUERY_ONLY_WEIGHTS,
    OUT_OF_SCOPE_THRESHOLD,
    IN_SCOPE_KEYWORDS,
)
from src.utils import is_query_in_scope


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL SCORING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_cf_scores(models, user_ratings, cand_ids):
    """
    Compute Item-CF scores for all candidate apps.

    Falls back to SVD latent similarity when item-CF has no signal
    (i.e. no shared users between the target and rated apps).

    Returns
    -------
    dict {app_id: raw_cf_score}  — NOT yet normalised to 0–1
    """
    app_to_idx  = models["app_to_idx"]
    item_sim    = models["item_sim"]
    app_factors = models["app_factors"]

    cf_raw = {}
    for target_aid in cand_ids:
        if target_aid not in app_to_idx:
            cf_raw[target_aid] = 0.0
            continue

        t_idx = app_to_idx[target_aid]
        num = den = 0.0

        for rated_aid, rating in user_ratings.items():
            if rated_aid not in app_to_idx:
                continue
            r_idx = app_to_idx[rated_aid]
            sim_v = item_sim[t_idx, r_idx]
            if sim_v > 0:
                num += sim_v * rating
                den += sim_v

        if den > 0:
            cf_raw[target_aid] = num / den
        else:
            # SVD fallback: average of liked apps' latent vectors
            liked = [
                a for a, r in user_ratings.items()
                if r >= 4 and a in app_to_idx
            ]
            if liked:
                ideal = app_factors[
                    [app_to_idx[a] for a in liked]
                ].mean(axis=0, keepdims=True)
                sims = cosine_similarity(ideal, app_factors)[0]
                cf_raw[target_aid] = float(sims[t_idx]) if t_idx < len(sims) else 0.0
            else:
                cf_raw[target_aid] = 0.0

    return cf_raw


def _get_content_scores(models, user_ratings, cand_ids, query_text=None):
    """
    Compute content-based scores for candidate apps.

    Two modes:
      - Query mode  : cosine similarity between query vector and app TF-IDF
      - History mode: average content similarity to liked apps

    When a query is present, raw cosine scores are returned WITHOUT
    normalisation so that out-of-scope queries score near 0 (not 0.5).

    Returns
    -------
    dict {app_id: content_score (0–1)}
    """
    content_tfidf = models["content_tfidf"]
    content_mat   = models["content_mat"]
    content_sim   = models["content_sim"]
    meta_idx      = models["meta_app_to_idx"]

    if query_text and query_text.strip():
        # Query mode — raw cosine, no normalisation
        q_vec  = content_tfidf.transform([query_text.lower()])
        q_sims = cosine_similarity(q_vec, content_mat)[0]
        return {
            aid: float(q_sims[meta_idx[aid]]) if aid in meta_idx else 0.0
            for aid in cand_ids
        }

    elif user_ratings:
        # History mode — average similarity to liked apps
        liked_idxs = [
            meta_idx[a]
            for a, r in user_ratings.items()
            if a in meta_idx and r >= 4
        ]
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

        # Normalise history scores to 0–1
        vals = list(raw_c.values())
        mn, mx = min(vals), max(vals)
        return {
            aid: (v - mn) / (mx - mn) if mx > mn else 0.5
            for aid, v in raw_c.items()
        }

    else:
        # No signal — return neutral 0.5
        return {aid: 0.5 for aid in cand_ids}


def _normalise(score_dict):
    """Min-max normalise a dict of scores to the 0–1 range."""
    vals = list(score_dict.values())
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return {k: 0.5 for k in score_dict}
    return {k: (v - mn) / (mx - mn) for k, v in score_dict.items()}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN HYBRID RECOMMENDER
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_recommend(
    models,
    metadata_df,
    sentiment_lookup,
    hidden_gem_lookup,
    user_ratings=None,
    query_text=None,
    n=10,
    category=None,
    free_only=False,
    pricing_filter=None,
    alpha=DEFAULT_ALPHA,
    beta=DEFAULT_BETA,
    gamma=DEFAULT_GAMMA,
    gem_boost=DEFAULT_GEM_BOOST,
):
    """
    Main hybrid recommendation function.

    Parameters
    ----------
    models           : dict  (output of models.build_all_models)
    metadata_df      : pd.DataFrame
    sentiment_lookup : dict {app_id: float 0–1}
    hidden_gem_lookup: dict {app_id: 0 or 1}
    user_ratings     : dict {app_id: float 1–5} or None
    query_text       : str or None
    n                : int  — number of results
    category         : str or None  — filter by project_category
    free_only        : bool
    pricing_filter   : str or None  — filter by pricing_model
    alpha, beta, gamma: float  — fusion weights (auto-normalised)
    gem_boost        : float  — bonus added to hidden gem scores

    Returns
    -------
    pd.DataFrame with columns:
      rank, app_name, category, pricing, avg_rating, total_ratings,
      total_reviews, installs, is_hidden_gem, hybrid_score,
      cf_score, nlp_score, content_score, summary
    """
    # ── Build candidate pool ──────────────────────────────────────────────────
    candidates = metadata_df.copy()
    if category and category != "All Categories":
        candidates = candidates[candidates["project_category"] == category]
    if free_only:
        candidates = candidates[candidates["free"] == True]
    if pricing_filter and pricing_filter != "All":
        candidates = candidates[candidates["pricing_model"] == pricing_filter]
    if user_ratings:
        candidates = candidates[~candidates["app_id"].isin(user_ratings.keys())]

    cand_ids = candidates["app_id"].tolist()
    if not cand_ids:
        return pd.DataFrame()

    # ── Determine fusion weights ──────────────────────────────────────────────
    total = alpha + beta + gamma
    a, b, g = alpha / total, beta / total, gamma / total

    # When query is typed but no user history: let content dominate
    if query_text and query_text.strip() and not user_ratings:
        a, b, g = QUERY_ONLY_WEIGHTS

    # ── Compute individual scores ─────────────────────────────────────────────
    # CF scores
    if user_ratings:
        cf_raw  = _get_cf_scores(models, user_ratings, cand_ids)
        cf_norm = _normalise(cf_raw)
    else:
        # No history — use Bayesian average as CF signal
        bayesian = candidates.set_index("app_id")["bayesian_avg"].fillna(0).to_dict()
        cf_norm  = _normalise(bayesian)

    # Content scores (raw cosine when query present — no normalisation)
    content_scores = _get_content_scores(models, user_ratings, cand_ids, query_text)

    # ── Fuse into hybrid score ────────────────────────────────────────────────
    results = []
    candidates_idx = candidates.set_index("app_id")

    for aid in cand_ids:
        cf_s   = cf_norm.get(aid, 0.0)
        nlp_s  = sentiment_lookup.get(aid, 0.5)
        cont_s = content_scores.get(aid, 0.0)
        is_gem = hidden_gem_lookup.get(aid, 0)

        # Apply gem boost only when content score shows some relevance
        gem_bonus = gem_boost * is_gem * (1 if cont_s > 0.05 else 0)

        hybrid = a * cf_s + b * nlp_s + g * cont_s + gem_bonus
        hybrid = min(1.0, hybrid)

        if aid not in candidates_idx.index:
            continue
        row = candidates_idx.loc[aid]

        results.append({
            "app_id":        aid,
            "app_name":      str(row.get("app_name", aid)),
            "category":      str(row.get("project_category", "")),
            "pricing":       str(row.get("pricing_model", "")),
            "avg_rating":    float(row["avg_rating"])    if pd.notna(row.get("avg_rating"))    else 0.0,
            "total_ratings": int(row["total_ratings"])   if pd.notna(row.get("total_ratings")) else 0,
            "total_reviews": int(row["total_reviews"])   if pd.notna(row.get("total_reviews")) else 0,
            "installs":      int(row["installs"])        if pd.notna(row.get("installs"))      else 0,
            "is_hidden_gem": int(is_gem),
            "hybrid_score":  round(hybrid, 4),
            "cf_score":      round(cf_s,   4),
            "nlp_score":     round(nlp_s,  4),
            "content_score": round(cont_s, 4),
            "summary":       str(row.get("summary", "")) if pd.notna(row.get("summary")) else "",
        })

    if not results:
        return pd.DataFrame()

    df = (pd.DataFrame(results)
          .sort_values("hybrid_score", ascending=False)
          .head(n)
          .reset_index(drop=True))
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SIMILAR APP LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def find_similar_apps(models, metadata_df, app_id, n=6, mode="cf"):
    """
    Find apps with similar user rating patterns (CF) or latent factors (SVD).

    Parameters
    ----------
    models      : dict  (output of models.build_all_models)
    metadata_df : pd.DataFrame
    app_id      : str
    n           : int
    mode        : 'cf' or 'svd'

    Returns
    -------
    pd.DataFrame with columns: app_id, app_name, category, avg_rating, similarity
    """
    app_to_idx = models["app_to_idx"]
    app_ids    = models["app_ids"]
    sim_mat    = models["svd_sim"] if mode == "svd" else models["item_sim"]
    meta_idx   = metadata_df.set_index("app_id")

    if app_id not in app_to_idx:
        return pd.DataFrame()

    idx      = app_to_idx[app_id]
    sims     = sim_mat[idx].copy()
    top_idxs = np.argsort(sims)[::-1][: n * 2]

    results = []
    for i in top_idxs:
        if sims[i] <= 0:
            continue
        aid = app_ids[i]
        if aid not in meta_idx.index:
            continue
        row = meta_idx.loc[aid]
        results.append({
            "app_id":     aid,
            "app_name":   str(row.get("app_name", aid)),
            "category":   str(row.get("project_category", "")),
            "avg_rating": float(row["avg_rating"]) if pd.notna(row.get("avg_rating")) else 0.0,
            "similarity": round(float(sims[i]), 4),
        })
        if len(results) >= n:
            break

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# OUT-OF-SCOPE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def is_out_of_scope(query_text, recs_df):
    """
    Determine if a query is outside the dataset's scope.

    Returns True (out of scope) when:
      - A query was provided
      - The query doesn't match known in-scope keywords
      - The max content_score in results is below OUT_OF_SCOPE_THRESHOLD

    Parameters
    ----------
    query_text : str
    recs_df    : pd.DataFrame  (output of hybrid_recommend)

    Returns
    -------
    bool
    """
    if not query_text or not query_text.strip():
        return False
    if len(recs_df) == 0:
        return True
    if "content_score" not in recs_df.columns:
        return False

    in_scope = is_query_in_scope(query_text, IN_SCOPE_KEYWORDS)
    low_confidence = recs_df["content_score"].max() < OUT_OF_SCOPE_THRESHOLD

    return not in_scope or low_confidence
