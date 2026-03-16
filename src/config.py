"""
config.py — Central configuration for Senti-Recommend
All constants, file paths, model parameters, and lexicons live here.
Change values here to affect the entire system.
"""

# ─────────────────────────────────────────────────────────────────────────────
# DATA FILE PATHS
# ─────────────────────────────────────────────────────────────────────────────

import os

# Path to processed data folder
DATA_DIR = os.path.join("data", "processed")

DATA_FILES = {
    "reviews":  os.path.join(DATA_DIR, "cleaned_reviews.csv"),
    "metadata": os.path.join(DATA_DIR, "cleaned_metadata.csv"),
    "app_sentiment":    os.path.join(DATA_DIR, "app_sentiment_scores.csv"),
    "review_sentiment": os.path.join(DATA_DIR, "review_sentiments.csv"),
    "item_sim":    os.path.join(DATA_DIR, "item_similarity_matrix.npy"),
    "svd_factors": os.path.join(DATA_DIR, "svd_app_factors.npy"),
    "svd_sim":     os.path.join(DATA_DIR, "svd_sim_matrix.npy"),
    "app_ids":     os.path.join(DATA_DIR, "app_ids_ordered.json"),
}


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID MODEL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# Default fusion weights (must sum to 1.0 — normalised automatically)
DEFAULT_ALPHA = 0.50   # α — Collaborative Filtering weight
DEFAULT_BETA  = 0.35   # β — NLP Sentiment weight
DEFAULT_GAMMA = 0.15   # γ — Content-Based weight

# When a query is typed but no user history exists, content should dominate
QUERY_ONLY_WEIGHTS = (0.10, 0.20, 0.70)   # (alpha, beta, gamma)

# Hidden gem bonus added to hybrid score
DEFAULT_GEM_BOOST = 0.05

# SVD latent factors
SVD_N_COMPONENTS = 50
SVD_RANDOM_STATE = 42

# Hidden gem thresholds (percentiles of bayesian_avg and total_reviews)
GEM_RATING_PERCENTILE  = 0.55   # App must be above this bayesian_avg percentile
GEM_REVIEWS_PERCENTILE = 0.30   # App must be below this total_reviews percentile

# Content TF-IDF settings
CONTENT_TFIDF_MAX_FEATURES = 3000
CONTENT_TFIDF_NGRAM_RANGE  = (1, 2)

# Max reviews per app aggregated for ABSA (trade-off: quality vs speed)
ABSA_MAX_REVIEWS_PER_APP = 300

# Content score threshold — below this we flag query as out-of-scope
OUT_OF_SCOPE_THRESHOLD = 0.10


# ─────────────────────────────────────────────────────────────────────────────
# ASPECT-BASED SENTIMENT ANALYSIS LEXICONS
# ─────────────────────────────────────────────────────────────────────────────

ASPECT_KEYWORDS = {
    "Ease of Use": [
        "easy", "simple", "intuitive", "user friendly", "user-friendly",
        "beginner", "accessible", "learning curve", "complicated",
        "difficult", "confusing", "hard to use", "smooth", "seamless",
        "overwhelming", "straightforward", "clunky",
    ],
    "Pricing": [
        "price", "pricing", "expensive", "cheap", "affordable", "cost",
        "subscription", "paid", "free", "premium", "value for money",
        "worth", "overpriced", "budget", "plan", "tier", "billing",
        "trial", "money", "fee", "charge", "refund",
    ],
    "Customer Support": [
        "support", "customer service", "help desk", "response", "team",
        "staff", "agent", "contact", "ticket", "reply", "chat support",
        "helpful", "unhelpful", "ignored", "resolved", "quick response",
        "responsive", "unresponsive",
    ],
    "Performance": [
        "fast", "slow", "speed", "quick", "lag", "crash", "bug", "glitch",
        "freeze", "reliable", "unreliable", "stable", "performance",
        "loading", "responsive", "snappy", "delay", "timeout", "error",
        "accurate", "accuracy",
    ],
    "Features": [
        "feature", "features", "functionality", "function", "tool", "tools",
        "option", "options", "capability", "integration", "plugin", "api",
        "update", "missing", "lacks", "limited", "powerful", "robust",
        "versatile", "customise", "customize",
    ],
}

POSITIVE_WORDS = {
    "great", "excellent", "amazing", "fantastic", "wonderful", "love", "loved",
    "best", "perfect", "awesome", "brilliant", "incredible", "outstanding",
    "superb", "terrific", "easy", "simple", "fast", "helpful", "useful",
    "good", "nice", "smooth", "clean", "efficient", "reliable", "impressed",
    "happy", "satisfied", "worth", "recommend", "powerful", "intuitive",
    "accurate", "quick", "responsive", "stable", "seamless", "convenient",
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "worst", "hate", "hated",
    "useless", "broken", "fail", "failed", "disappointing", "disappointed",
    "slow", "crash", "bug", "glitch", "error", "problem", "issue",
    "expensive", "overpriced", "waste", "cancel", "refund", "scam",
    "difficult", "confusing", "limited", "missing", "lacks", "poor",
    "frustrating", "annoying", "mediocre", "unreliable", "ignored",
    "unresponsive", "freeze", "lag", "delay", "inaccurate", "complicated",
}

NEGATION_WORDS = {
    "not", "no", "never", "cannot", "can't", "won't", "doesn't",
    "isn't", "wasn't", "weren't", "don't", "didn't", "hardly", "barely",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATASET SCOPE — used for out-of-scope query detection
# ─────────────────────────────────────────────────────────────────────────────

# Keywords that indicate an in-scope query
IN_SCOPE_KEYWORDS = [
    "generative", "text", "chatbot", "chat", "image", "generation", "photo",
    "coding", "code", "programming", "developer", "productivity", "marketing",
    "writing", "email", "voice", "pdf", "video", "ai tool", "generator",
    "assistant", "automation", "summarize", "transcribe", "design",
]

# The 5 dataset categories
PROJECT_CATEGORIES = [
    "Generative Text / Chatbots",
    "Image Generation",
    "AI Coding Assistants",
    "Productivity AI",
    "Marketing AI",
]


# ─────────────────────────────────────────────────────────────────────────────
# UI / DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

APP_TITLE    = "Senti-Recommend"
APP_ICON     = "🔍"
APP_SUBTITLE = "Hybrid AI Tool Discovery Engine · Collaborative Filtering + Sentiment Analysis"

# Colour palette (matches CSS variables in app.py)
COLORS = {
    "teal":    "#2dd4bf",
    "blue":    "#3b82f6",
    "purple":  "#a78bfa",
    "amber":   "#fbbf24",
    "rose":    "#fb7185",
    "muted":   "#8b949e",
    "surface": "#161b22",
    "bg":      "#0d1117",
    "text":    "#e6edf3",
    "border":  "#30363d"
}

# Chart colour palette per category
CATEGORY_COLORS = {
    "Generative Text / Chatbots": "#2dd4bf",
    "Image Generation":           "#3b82f6",
    "AI Coding Assistants":       "#a78bfa",
    "Productivity AI":            "#f59e0b",
    "Marketing AI":               "#fb7185",
}
