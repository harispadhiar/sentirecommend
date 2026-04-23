"""
models.py — Model Building for Senti-Recommend
Builds and returns all ML models used by the recommender:
  1. User-Item Matrix (mean-centred)
  2. Item-Based CF  (cosine similarity)
  3. SVD Matrix Factorisation  (latent factors)
  4. Content-Based TF-IDF  (app description similarity)

Each function is independent so models can be rebuilt selectively.
Pre-computed .npy files are loaded if available (from Notebook 03),
otherwise models are built from scratch.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    DATA_FILES,
    SVD_N_COMPONENTS,
    SVD_RANDOM_STATE,
    CONTENT_TFIDF_MAX_FEATURES,
    CONTENT_TFIDF_NGRAM_RANGE,
)


# ─────────────────────────────────────────────────────────────────────────────
# USER-ITEM MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def build_user_item_matrix(reviews_df):
    """
    Build a mean-centred User-Item matrix from the reviews DataFrame.

    Mean-centering removes the "generous rater" vs "harsh rater" bias:
    each user's mean rating is subtracted from all their ratings.

    Parameters
    ----------
    reviews_df : pd.DataFrame  with columns [user_name, app_id, star_rating]

    Returns
    -------
    pivot_full    : pd.DataFrame  — raw (not centred) pivot
    item_sparse   : csr_matrix    — centred, transposed (apps × users)
    app_ids       : list[str]
    user_ids      : list[str]
    app_to_idx    : dict {app_id: int}
    """
    pivot_full = reviews_df.pivot_table(
        index="user_name",
        columns="app_id",
        values="star_rating",
        aggfunc="mean",
    )

    # Mean-centre per user
    user_means    = pivot_full.mean(axis=1)
    pivot_centred = pivot_full.subtract(user_means, axis=0)

    app_ids    = list(pivot_full.columns)
    user_ids   = list(pivot_full.index)
    app_to_idx = {a: i for i, a in enumerate(app_ids)}

    # Sparse matrix: apps × users (transposed for item-item similarity)
    item_sparse = csr_matrix(pivot_centred.fillna(0).values.T)

    return pivot_full, item_sparse, app_ids, user_ids, app_to_idx


# ─────────────────────────────────────────────────────────────────────────────
# ITEM-BASED COLLABORATIVE FILTERING
# ─────────────────────────────────────────────────────────────────────────────

def build_item_cf(item_sparse, app_ids):
    """
    Build the Item-Item cosine similarity matrix.

    Attempts to load a pre-computed matrix from disk first.
    Falls back to computing from scratch if not found.

    Parameters
    ----------
    item_sparse : csr_matrix  (apps × users, mean-centred)
    app_ids     : list[str]

    Returns
    -------
    item_sim : np.ndarray  shape (n_apps, n_apps)
    """
    try:
        item_sim = np.load(DATA_FILES["item_sim"])
        print("  ✅ Item-CF similarity loaded from file.")
    except FileNotFoundError:
        print("  ⚙️  Computing Item-CF similarity matrix...")
        item_sim = cosine_similarity(item_sparse)
        np.fill_diagonal(item_sim, 0)
        print(f"  ✅ Item-CF similarity computed: {item_sim.shape}")

    return item_sim


def predict_cf_rating(item_sim, app_to_idx, user_ratings, target_app_id):
    """
    Predict the rating a user would give to target_app_id
    using weighted Item-CF (similarity-weighted average of known ratings).

    Parameters
    ----------
    item_sim       : np.ndarray   — item similarity matrix
    app_to_idx     : dict
    user_ratings   : dict {app_id: float}
    target_app_id  : str

    Returns
    -------
    float or None  — predicted rating, or None if no signal available
    """
    if target_app_id not in app_to_idx:
        return None

    t_idx = app_to_idx[target_app_id]
    num = den = 0.0

    for rated_app, rating in user_ratings.items():
        if rated_app == target_app_id or rated_app not in app_to_idx:
            continue
        r_idx = app_to_idx[rated_app]
        sim_v = item_sim[t_idx, r_idx]
        if sim_v > 0:
            num += sim_v * rating
            den += sim_v

    return num / den if den > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# SVD MATRIX FACTORISATION
# ─────────────────────────────────────────────────────────────────────────────

def build_svd(item_sparse):
    """
    Build SVD latent factor model.

    Attempts to load pre-computed factors from disk first.
    Falls back to computing with TruncatedSVD if not found.

    Parameters
    ----------
    item_sparse : csr_matrix  (apps × users, mean-centred)

    Returns
    -------
    app_factors : np.ndarray  shape (n_apps, SVD_N_COMPONENTS)
    svd_sim     : np.ndarray  shape (n_apps, n_apps)
    """
    try:
        app_factors = np.load(DATA_FILES["svd_factors"])
        svd_sim     = np.load(DATA_FILES["svd_sim"])
        print("  ✅ SVD factors loaded from file.")
    except FileNotFoundError:
        print(f"  ⚙️  Training SVD (k={SVD_N_COMPONENTS})...")
        svd = TruncatedSVD(
            n_components=SVD_N_COMPONENTS,
            random_state=SVD_RANDOM_STATE,
            n_iter=10,
        )
        app_factors = svd.fit_transform(item_sparse)
        svd_sim = cosine_similarity(app_factors)
        np.fill_diagonal(svd_sim, 0)
        print(f"  ✅ SVD trained. Explained variance: "
              f"{svd.explained_variance_ratio_.sum()*100:.1f}%")

    return app_factors, svd_sim


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT-BASED TF-IDF
# ─────────────────────────────────────────────────────────────────────────────

def build_content_model(metadata_df):
    """
    Build the TF-IDF content-based similarity model.

    Uses the 'content_text' column (built by data_loader.build_content_text).

    Parameters
    ----------
    metadata_df : pd.DataFrame  — must contain 'content_text' column

    Returns
    -------
    content_tfidf   : TfidfVectorizer  (fitted)
    content_mat     : sparse matrix    (n_apps × n_features)
    content_sim     : np.ndarray       (n_apps × n_apps)
    meta_app_to_idx : dict {app_id: row_index_in_metadata}
    """
    if "content_text" not in metadata_df.columns:
        raise ValueError(
            "metadata_df must have a 'content_text' column. "
            "Run data_loader.build_content_text() first."
        )

    content_tfidf = TfidfVectorizer(
        max_features=CONTENT_TFIDF_MAX_FEATURES,
        ngram_range=CONTENT_TFIDF_NGRAM_RANGE,
        min_df=1,
        stop_words="english",
        strip_accents="unicode",
        lowercase=True
    )

    content_mat = content_tfidf.fit_transform(metadata_df["content_text"])
    content_sim = cosine_similarity(content_mat)
    np.fill_diagonal(content_sim, 0)

    meta_app_to_idx = {
        row["app_id"]: i for i, row in metadata_df.iterrows()
    }

    print(f"  ✅ Content TF-IDF: {content_mat.shape} matrix")
    return content_tfidf, content_mat, content_sim, meta_app_to_idx


# ─────────────────────────────────────────────────────────────────────────────
# FULL MODEL BUILD PIPELINE  (called once by app.py on startup)
# ─────────────────────────────────────────────────────────────────────────────

def build_all_models(reviews_df, metadata_df):
    """
    Build all models in the correct order.
    Loads from disk where possible, computes from scratch otherwise.

    Parameters
    ----------
    reviews_df  : pd.DataFrame  (from data_loader.load_reviews)
    metadata_df : pd.DataFrame  (from data_loader.load_metadata, with content_text)

    Returns
    -------
    dict containing all model components needed by the recommender.
    """
    print("Building models...")

    # 1. User-Item Matrix
    print("  [1/4] User-Item Matrix...")
    pivot_full, item_sparse, app_ids, user_ids, app_to_idx = \
        build_user_item_matrix(reviews_df)

    # 2. Item-CF
    print("  [2/4] Item-Based CF...")
    item_sim = build_item_cf(item_sparse, app_ids)

    # 3. SVD
    print("  [3/4] SVD Matrix Factorisation...")
    app_factors, svd_sim = build_svd(item_sparse)

    # 4. Content TF-IDF
    print("  [4/4] Content-Based TF-IDF...")
    content_tfidf, content_mat, content_sim, meta_app_to_idx = \
        build_content_model(metadata_df)

    print("  ✅ All models ready.")

    return {
        "app_ids":          app_ids,
        "user_ids":         user_ids,
        "app_to_idx":       app_to_idx,
        "item_sim":         item_sim,
        "app_factors":      app_factors,
        "svd_sim":          svd_sim,
        "content_tfidf":    content_tfidf,
        "content_mat":      content_mat,
        "content_sim":      content_sim,
        "meta_app_to_idx":  meta_app_to_idx,
    }
