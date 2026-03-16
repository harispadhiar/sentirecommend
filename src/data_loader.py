"""
data_loader.py — Data Loading and Preprocessing
Handles all raw data ingestion, sentiment proxy computation,
hidden gem flagging, and content text construction.

All functions return clean, merged DataFrames ready for model building.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    DATA_FILES,
    GEM_RATING_PERCENTILE,
    GEM_REVIEWS_PERCENTILE,
)


# ─────────────────────────────────────────────────────────────────────────────
# RAW DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_reviews(path=None):
    """
    Load the cleaned reviews CSV.

    Returns
    -------
    pd.DataFrame with columns including:
      app_id, review_id, user_name, review_text, star_rating,
      review_date, thumbs_up_count, review_length, has_reply
    """
    path = path or DATA_FILES["reviews"]
    df = pd.read_csv(path)
    df["star_rating"] = pd.to_numeric(df["star_rating"], errors="coerce")
    df = df.dropna(subset=["star_rating", "review_text"])
    df["review_text"] = df["review_text"].astype(str)
    return df.reset_index(drop=True)


def load_metadata(path=None):
    """
    Load the cleaned app metadata CSV.

    Returns
    -------
    pd.DataFrame with columns including:
      app_id, app_name, developer, project_category, avg_rating,
      total_ratings, total_reviews, installs, bayesian_avg,
      pricing_model, free, description, summary
    """
    path = path or DATA_FILES["metadata"]
    df = pd.read_csv(path)
    df["avg_rating"]    = pd.to_numeric(df["avg_rating"],    errors="coerce")
    df["bayesian_avg"]  = pd.to_numeric(df["bayesian_avg"],  errors="coerce")
    df["total_ratings"] = pd.to_numeric(df["total_ratings"], errors="coerce").fillna(0).astype(int)
    df["total_reviews"] = pd.to_numeric(df["total_reviews"], errors="coerce").fillna(0).astype(int)
    df["installs"]      = pd.to_numeric(df["installs"],      errors="coerce").fillna(0).astype(int)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT LOADING / PROXY
# ─────────────────────────────────────────────────────────────────────────────

def load_or_compute_sentiment(metadata_df, path=None):
    """
    Attempt to load full NLP sentiment scores from Notebook 02 output.
    If the file is not found, compute a lightweight proxy from star ratings.

    Parameters
    ----------
    metadata_df : pd.DataFrame  (from load_metadata)
    path        : str or None   (path to app_sentiment_scores.csv)

    Returns
    -------
    metadata_df : pd.DataFrame  with new columns:
                    mean_sentiment_score, is_hidden_gem, sentiment_norm
    nlp_source  : str  — describes which source was used
    """
    path = path or DATA_FILES["app_sentiment"]

    try:
        sent_df = pd.read_csv(path)
        metadata_df = metadata_df.merge(
            sent_df[["app_id", "mean_sentiment_score", "is_hidden_gem"]],
            on="app_id", how="left",
        )
        nlp_source = "NLP Model (Notebook 02)"

    except FileNotFoundError:
        # Fallback: scale bayesian_avg to (-1, +1) as a sentiment proxy
        scaler = MinMaxScaler(feature_range=(-1, 1))
        fill_val = metadata_df["bayesian_avg"].median()
        metadata_df["mean_sentiment_score"] = scaler.fit_transform(
            metadata_df[["bayesian_avg"]].fillna(fill_val)
        )

        # Flag hidden gems using percentile thresholds from config
        s_thresh = metadata_df["bayesian_avg"].quantile(GEM_RATING_PERCENTILE)
        r_thresh = metadata_df["total_reviews"].quantile(GEM_REVIEWS_PERCENTILE)
        metadata_df["is_hidden_gem"] = (
            (metadata_df["bayesian_avg"]  >= s_thresh) &
            (metadata_df["total_reviews"] <= r_thresh)
        ).astype(int)

        nlp_source = "Rating Proxy (run NB02 for full NLP scores)"

    # Fill any remaining nulls
    metadata_df["mean_sentiment_score"] = metadata_df["mean_sentiment_score"].fillna(0)
    metadata_df["is_hidden_gem"]        = metadata_df["is_hidden_gem"].fillna(0).astype(int)

    # Normalise sentiment score to (0, 1) for hybrid formula
    scaler_norm = MinMaxScaler(feature_range=(0, 1))
    metadata_df["sentiment_norm"] = scaler_norm.fit_transform(
        metadata_df[["mean_sentiment_score"]]
    )

    return metadata_df, nlp_source


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT TEXT CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_content_text(metadata_df, desc_max_chars=500):
    """
    Construct the 'content_text' column used by the TF-IDF content-based model.

    Concatenates: app_name + category + summary + description (truncated)

    Parameters
    ----------
    metadata_df   : pd.DataFrame
    desc_max_chars: int — max characters to take from description

    Returns
    -------
    pd.DataFrame with new 'content_text' column
    """
    metadata_df["content_text"] = (
        metadata_df["app_name"].fillna("") + " " +
        metadata_df["project_category"].fillna("") + " " +
        metadata_df["summary"].fillna("") + " " +
        metadata_df["description"].fillna("").str[:desc_max_chars]
    ).str.lower()

    return metadata_df


# ─────────────────────────────────────────────────────────────────────────────
# FULL LOAD PIPELINE  (convenience wrapper used by app.py and notebooks)
# ─────────────────────────────────────────────────────────────────────────────

def load_all_data():
    """
    Full data loading pipeline — single call to get everything ready.

    Returns
    -------
    reviews     : pd.DataFrame
    metadata    : pd.DataFrame  (with sentiment + content_text columns)
    nlp_source  : str
    """
    reviews  = load_reviews()
    metadata = load_metadata()
    metadata, nlp_source = load_or_compute_sentiment(metadata)
    metadata = build_content_text(metadata)
    return reviews, metadata, nlp_source


# ─────────────────────────────────────────────────────────────────────────────
# LOOKUP DICT BUILDERS  (used by recommender)
# ─────────────────────────────────────────────────────────────────────────────

def build_sentiment_lookup(metadata_df):
    """Return {app_id: normalised_sentiment_score (0–1)}"""
    return dict(zip(metadata_df["app_id"], metadata_df["sentiment_norm"]))


def build_hidden_gem_lookup(metadata_df):
    """Return {app_id: 0 or 1}"""
    return dict(zip(metadata_df["app_id"], metadata_df["is_hidden_gem"]))
