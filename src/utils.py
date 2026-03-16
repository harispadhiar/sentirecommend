"""
utils.py — Shared helper functions for Senti-Recommend
Small, reusable utilities used across the app and notebooks.
"""

import re
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# TEXT FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def star_str(rating):
    """
    Convert a numeric rating to a star string.
    Example: 4.5 → '★★★★☆  4.5'
    """
    if pd.isna(rating):
        return "N/A"
    full  = int(rating)
    empty = 5 - full
    return "★" * full + "☆" * empty + f"  {rating:.1f}"


def fmt_number(n):
    """
    Format large numbers into readable shorthand.
    Example: 1_500_000 → '1.5M', 3200 → '3.2K'
    """
    if pd.isna(n):
        return "—"
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def truncate(text, max_len=180):
    """Truncate text to max_len characters, appending '...' if cut."""
    text = str(text) if pd.notna(text) else ""
    return text[:max_len] + "..." if len(text) > max_len else text


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text):
    """
    Clean a raw review string for NLP processing.

    Steps:
      1. Lowercase
      2. Remove URLs and email addresses
      3. Remove special characters (keep apostrophes for contractions)
      4. Normalise whitespace

    Returns an empty string for null input.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+|\S+@\S+", " ", text)   # URLs / emails
    text = re.sub(r"[^a-z\'\s]", " ", text)                  # special chars
    text = re.sub(r"\s+", " ", text).strip()                  # whitespace
    return text


# ─────────────────────────────────────────────────────────────────────────────
# HTML COMPONENT HELPERS  (used in Streamlit markdown blocks)
# ─────────────────────────────────────────────────────────────────────────────

def score_bar_html(label, value, color="#2dd4bf"):
    """
    Generate HTML for a labelled horizontal score bar.
    Value should be between 0.0 and 1.0.
    """
    pct = max(0.0, min(1.0, float(value))) * 100
    return f"""
    <div class="score-bar-wrap">
      <div class="score-label">
        <span>{label}</span><span>{value:.3f}</span>
      </div>
      <div class="score-bar-bg">
        <div class="score-bar-fill"
             style="width:{pct:.1f}%;background:{color}">
        </div>
      </div>
    </div>"""


def badge_html(text, badge_class="badge-cat"):
    """Generate an inline HTML badge."""
    return f'<span class="badge {badge_class}">{text}</span>'


def metric_tile_html(value, label):
    """Generate a metric tile HTML block."""
    return f"""
    <div class="metric-tile">
      <div class="metric-val">{value}</div>
      <div class="metric-lbl">{label}</div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# QUERY SCOPE CHECK
# ─────────────────────────────────────────────────────────────────────────────

def is_query_in_scope(query_text, in_scope_keywords):
    """
    Returns True if the query mentions any keyword that suggests it is
    within the dataset's scope (AI tools in the 5 project categories).

    Used to decide whether to show an out-of-scope warning.
    """
    if not query_text:
        return True
    query_lower = query_text.lower()
    return any(kw in query_lower for kw in in_scope_keywords)
