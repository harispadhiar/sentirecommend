"""
absa.py — Aspect-Based Sentiment Analysis (ABSA)
Detects how users feel about specific dimensions of each AI tool:
  Ease of Use · Pricing · Customer Support · Performance · Features

Algorithm per review sentence:
  1. Check if any aspect keyword appears in the sentence
  2. Count positive/negative sentiment words in that sentence
  3. Apply negation flipping (e.g. "not good" → negative)
  4. Return a score per aspect: +1 = positive, -1 = negative

Aggregation per app:
  - Concatenate up to ABSA_MAX_REVIEWS_PER_APP reviews per app
  - Run sentence-level ABSA on the combined text
  - Average sentence scores per aspect
"""

import re
import numpy as np
from collections import defaultdict

from src.config import (
    ASPECT_KEYWORDS,
    POSITIVE_WORDS,
    NEGATIVE_WORDS,
    NEGATION_WORDS,
    ABSA_MAX_REVIEWS_PER_APP,
)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-TEXT ABSA
# ─────────────────────────────────────────────────────────────────────────────

def analyse_aspects(text):
    """
    Run ABSA on a single text string (review or aggregated review blob).

    Parameters
    ----------
    text : str
        Raw review text.

    Returns
    -------
    dict : {aspect_name: float}
        Sentiment score per aspect, ranging from -1.0 (very negative)
        to +1.0 (very positive). Returns 0.0 for aspects not mentioned.
    """
    if not text or len(str(text)) < 5:
        return {aspect: 0.0 for aspect in ASPECT_KEYWORDS}

    text_lower  = str(text).lower()
    sentences   = re.split(r"[.!?\n]", text_lower)
    aspect_scores = defaultdict(list)

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 4:
            continue
        words = sentence.split()

        for aspect, keywords in ASPECT_KEYWORDS.items():
            # Loosen matching: allow partial keyword matches (substring matching)
            # e.g., "confusing" matches "confus", "user-friendly" matches "friendly"
            matched = any(kw in sentence for kw in keywords)
            if not matched:
                # Also try matching keyword prefixes (first 4+ characters)
                # to catch variations like "confus" for "confusing"
                matched = any(
                    len(kw) > 4 and kw[:4] in sentence
                    for kw in keywords
                )
            if not matched:
                continue

            pos_count = neg_count = 0

            for j, word in enumerate(words):
                clean_word = re.sub(r"[^a-z']", "", word)

                # Check for negation context (within 3 preceding words)
                is_negated = any(
                    words[k].rstrip("',") in NEGATION_WORDS
                    for k in range(max(0, j - 3), j)
                )

                if clean_word in POSITIVE_WORDS:
                    if is_negated:
                        neg_count += 1
                    else:
                        pos_count += 1
                elif clean_word in NEGATIVE_WORDS:
                    if is_negated:
                        pos_count += 1
                    else:
                        neg_count += 1

            if pos_count + neg_count > 0:
                score = (pos_count - neg_count) / (pos_count + neg_count)
                aspect_scores[aspect].append(score)

    return {
        aspect: round(float(np.mean(scores)), 4) if scores else 0.0
        for aspect, scores in aspect_scores.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# APP-LEVEL ABSA  (aggregated over all reviews for one app)
# ─────────────────────────────────────────────────────────────────────────────

def compute_app_absa(reviews_df):
    """
    Compute ABSA scores for every app in the reviews DataFrame.

    Parameters
    ----------
    reviews_df : pd.DataFrame
        Must contain columns: ['app_id', 'review_text']

    Returns
    -------
    dict : {app_id: {aspect_name: float}}
        Nested dict of aspect sentiment scores per app.
    """
    # Aggregate reviews per app (cap at ABSA_MAX_REVIEWS_PER_APP for speed)
    app_text = (
        reviews_df
        .groupby("app_id")["review_text"]
        .apply(
            lambda x: " ".join(
                x.dropna().head(ABSA_MAX_REVIEWS_PER_APP).astype(str)
            )
        )
        .reset_index()
    )
    app_text.columns = ["app_id", "agg_text"]

    app_absa = {}
    for _, row in app_text.iterrows():
        app_absa[row["app_id"]] = analyse_aspects(row["agg_text"])

    return app_absa


# ─────────────────────────────────────────────────────────────────────────────
# ABSA SUMMARY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def absa_to_dataframe(absa_scores):
    """
    Convert an ABSA score dict to a display-ready DataFrame.

    Returns columns: Aspect, Score, Sentiment, Strength
    """
    import pandas as pd

    rows = []
    for aspect, score in absa_scores.items():
        if score > 0.15:
            sentiment = "🟢 Positive"
        elif score < -0.15:
            sentiment = "🔴 Negative"
        else:
            sentiment = "🟡 Neutral"

        abs_score = abs(score)
        if abs_score > 0.6:
            strength = "▓▓▓▓ Strong"
        elif abs_score > 0.35:
            strength = "▓▓▓░ Moderate"
        elif abs_score > 0.15:
            strength = "▓▓░░ Mild"
        else:
            strength = "▓░░░ Weak"

        rows.append({
            "Aspect":    aspect,
            "Score":     round(score, 3),
            "Sentiment": sentiment,
            "Strength":  strength,
        })

    return pd.DataFrame(rows)


def absa_overall_score(absa_scores):
    """
    Compute a single overall ABSA score as the mean of all aspect scores.
    Returns 0.0 if no aspects have data.
    """
    scores = list(absa_scores.values())
    return round(float(np.mean(scores)), 4) if scores else 0.0
