"""
charts.py — All Visualisation Functions for Senti-Recommend
Each function returns a matplotlib Figure that Streamlit renders with st.pyplot().

Dark theme matches the app's CSS colour palette.
All figures use facecolor '#0d1117' (bg) or '#161b22' (surface).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from src.config import COLORS, CATEGORY_COLORS


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STYLE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _dark_ax(ax, fig=None):
    """Apply dark theme styling to a matplotlib Axes (and optionally Figure)."""
    ax.set_facecolor(COLORS["surface"])
    ax.tick_params(colors=COLORS["muted"], labelsize=9)
    ax.spines[["top", "right", "bottom", "left"]].set_color(COLORS["border"] if hasattr(ax, 'spines') else "#30363d")
    ax.xaxis.label.set_color(COLORS["muted"])
    ax.yaxis.label.set_color(COLORS["muted"])
    ax.title.set_color(COLORS["text"])
    if fig:
        fig.patch.set_facecolor(COLORS["bg"])
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# SCORE DECOMPOSITION BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def make_score_chart(cf_score, nlp_score, content_score, hybrid_score):
    """
    Horizontal bar chart showing the breakdown of a hybrid score.

    Parameters
    ----------
    cf_score, nlp_score, content_score, hybrid_score : float (0–1)

    Returns matplotlib Figure.
    """
    labels = ["CF Score", "NLP Score", "Content Score", "Hybrid Score"]
    values = [cf_score, nlp_score, content_score, hybrid_score]
    colors = [COLORS["blue"], COLORS["teal"], COLORS["amber"], COLORS["purple"]]

    fig, ax = plt.subplots(figsize=(5, 2.2))
    fig.patch.set_facecolor(COLORS["surface"])
    ax.set_facecolor(COLORS["surface"])

    bars = ax.barh(labels, values, color=colors, edgecolor="none", height=0.55)
    ax.set_xlim(0, 1)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.tick_params(colors=COLORS["muted"], labelsize=9)
    ax.xaxis.set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(
            val + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center",
            color=COLORS["text"], fontsize=9, fontweight="bold",
        )

    plt.tight_layout(pad=0.3)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ABSA RADAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def make_absa_radar(absa_scores):
    """
    Radar (spider) chart showing ABSA aspect scores for one app.

    Parameters
    ----------
    absa_scores : dict {aspect_name: float (-1 to +1)}

    Returns matplotlib Figure.
    """
    aspects = list(absa_scores.keys())
    values  = [max(-1.0, min(1.0, v)) for v in absa_scores.values()]
    # Normalise from (-1, +1) to (0, 1) for radar display
    values_norm = [(v + 1) / 2 for v in values]

    N      = len(aspects)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values_norm_closed = values_norm + values_norm[:1]
    angles_closed      = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(COLORS["surface"])
    ax.set_facecolor(COLORS["surface"])

    ax.plot(angles_closed, values_norm_closed, "o-",
            linewidth=2, color=COLORS["teal"])
    ax.fill(angles_closed, values_norm_closed,
            alpha=0.2, color=COLORS["teal"])

    ax.set_xticks(angles)
    ax.set_xticklabels(aspects, size=8, color=COLORS["muted"])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["−0.5", "0", "+0.5"], size=7, color="#555")
    ax.spines["polar"].set_color(COLORS["border"])
    ax.grid(color=COLORS["border"], linewidth=0.7)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RATING DONUT CHART
# ─────────────────────────────────────────────────────────────────────────────

def make_rating_donut(app_row):
    """
    Donut chart showing star rating breakdown for one app.

    Parameters
    ----------
    app_row : pd.Series or dict  — must contain ratings_1_star … ratings_5_star
                                   and avg_rating

    Returns matplotlib Figure, or None if no data.
    """
    vals = [
        app_row.get("ratings_5_star", 0) or 0,
        app_row.get("ratings_4_star", 0) or 0,
        app_row.get("ratings_3_star", 0) or 0,
        app_row.get("ratings_2_star", 0) or 0,
        app_row.get("ratings_1_star", 0) or 0,
    ]
    if sum(vals) == 0:
        return None

    labels = ["5★", "4★", "3★", "2★", "1★"]
    colors = ["#27ae60", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    fig.patch.set_facecolor(COLORS["surface"])
    ax.set_facecolor(COLORS["surface"])

    wedges, texts, autotexts = ax.pie(
        vals, labels=labels, colors=colors, autopct="%1.0f%%",
        startangle=90, pctdistance=0.78,
        wedgeprops={"edgecolor": COLORS["surface"], "linewidth": 2, "width": 0.55},
    )
    for t in texts:
        t.set_color(COLORS["muted"]); t.set_fontsize(8)
    for t in autotexts:
        t.set_color("white"); t.set_fontsize(7)

    avg = app_row.get("avg_rating", 0) or 0
    ax.text(0, 0, f"{float(avg):.1f}\n★",
            ha="center", va="center", fontsize=14,
            fontweight="bold", color=COLORS["amber"],
            fontfamily="monospace")

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY DISTRIBUTION BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def make_category_dist_chart(metadata_df):
    """
    Horizontal bar chart: number of apps per AI category.

    Returns matplotlib Figure.
    """
    counts  = metadata_df["project_category"].value_counts()
    palette = list(CATEGORY_COLORS.values())[:len(counts)]

    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    bars = ax.barh(counts.index, counts.values,
                   color=palette, edgecolor="none", height=0.6)
    ax.set_xlabel("Number of Apps", color=COLORS["muted"], fontsize=9)
    ax.tick_params(colors=COLORS["muted"], labelsize=9)
    ax.spines[["top", "right", "bottom", "left"]].set_color("#30363d")

    for bar, val in zip(bars, counts.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", color=COLORS["text"], fontsize=9)

    plt.tight_layout(pad=0.5)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT vs RATING SCATTER PLOT
# ─────────────────────────────────────────────────────────────────────────────

def make_sentiment_scatter(metadata_df):
    """
    Scatter plot: avg_rating (x) vs mean_sentiment_score (y),
    coloured by category, with hidden gems highlighted.

    Returns matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    for cat, color in CATEGORY_COLORS.items():
        sub = metadata_df[metadata_df["project_category"] == cat]
        ax.scatter(sub["avg_rating"], sub["mean_sentiment_score"],
                   color=color, s=80, alpha=0.7,
                   edgecolors="none", label=cat)

    gems = metadata_df[metadata_df["is_hidden_gem"] == 1]
    if len(gems) > 0:
        ax.scatter(gems["avg_rating"], gems["mean_sentiment_score"],
                   color="none", edgecolors=COLORS["amber"],
                   s=130, linewidths=1.5, label="💎 Hidden Gem", zorder=5)

    ax.set_xlabel("Average Star Rating", color=COLORS["muted"], fontsize=10)
    ax.set_ylabel("NLP Sentiment Score", color=COLORS["muted"], fontsize=10)
    ax.tick_params(colors=COLORS["muted"], labelsize=9)
    ax.spines[["top", "right", "bottom", "left"]].set_color("#30363d")
    ax.legend(fontsize=8, facecolor=COLORS["surface"],
              labelcolor=COLORS["text"], loc="lower right", framealpha=0.9)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RATING DISTRIBUTION HISTOGRAM
# ─────────────────────────────────────────────────────────────────────────────

def make_rating_histogram(metadata_df):
    """
    Histogram of avg_rating across all apps with mean line.

    Returns matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    ax.hist(metadata_df["avg_rating"].dropna(), bins=30,
            color=COLORS["teal"], edgecolor=COLORS["bg"], alpha=0.85)

    mean_val = metadata_df["avg_rating"].mean()
    ax.axvline(mean_val, color=COLORS["amber"], linestyle="--",
               linewidth=1.5, label=f"Mean: {mean_val:.2f}")

    ax.set_xlabel("Average Rating", color=COLORS["muted"], fontsize=9)
    ax.set_ylabel("Number of Apps", color=COLORS["muted"], fontsize=9)
    ax.tick_params(colors=COLORS["muted"], labelsize=9)
    ax.spines[["top", "right", "bottom", "left"]].set_color("#30363d")
    ax.legend(fontsize=8, facecolor=COLORS["surface"], labelcolor=COLORS["muted"])

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PRE-LAUNCH OVERVIEW DASHBOARD  (6-panel summary chart)
# ─────────────────────────────────────────────────────────────────────────────

def make_prelaunch_dashboard(metadata_df):
    """
    6-panel overview dashboard for Notebook 04 Step 6.
    Saved as 'prelaunch_dashboard.png'.

    Returns matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("Senti-Recommend — Pre-Launch Data Overview",
                 fontsize=14, fontweight="bold", color=COLORS["text"], y=0.98)

    for ax in axes.flat:
        ax.set_facecolor(COLORS["surface"])
        ax.spines[["top", "right", "bottom", "left"]].set_color("#30363d")
        ax.tick_params(colors=COLORS["muted"])

    palette = list(CATEGORY_COLORS.values())
    cat_counts = metadata_df["project_category"].value_counts()

    # 1 — Apps per Category
    axes[0, 0].barh(cat_counts.index, cat_counts.values,
                    color=palette, edgecolor="none")
    axes[0, 0].set_title("Apps per Category", color=COLORS["text"], fontweight="bold")
    axes[0, 0].set_xlabel("Count", color=COLORS["muted"])
    for i, (_, val) in enumerate(cat_counts.items()):
        axes[0, 0].text(val + 0.3, i, str(val), va="center",
                        color=COLORS["text"], fontsize=9)

    # 2 — Rating Distribution
    axes[0, 1].hist(metadata_df["avg_rating"].dropna(), bins=30,
                    color=COLORS["teal"], edgecolor=COLORS["bg"], alpha=0.85)
    axes[0, 1].set_title("App Rating Distribution", color=COLORS["text"], fontweight="bold")
    axes[0, 1].set_xlabel("Average Rating", color=COLORS["muted"])
    axes[0, 1].axvline(metadata_df["avg_rating"].mean(),
                       color=COLORS["amber"], linestyle="--", linewidth=1.5)

    # 3 — Sentiment Score Distribution
    if "mean_sentiment_score" in metadata_df.columns:
        axes[0, 2].hist(metadata_df["mean_sentiment_score"].dropna(), bins=30,
                        color=COLORS["purple"], edgecolor=COLORS["bg"], alpha=0.85)
        axes[0, 2].axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    axes[0, 2].set_title("Sentiment Score Distribution", color=COLORS["text"], fontweight="bold")
    axes[0, 2].set_xlabel("NLP Sentiment Score", color=COLORS["muted"])

    # 4 — Reviews vs Rating scatter
    cap = metadata_df["total_reviews"].quantile(0.95)
    gem_colors = metadata_df["is_hidden_gem"].map({0: COLORS["blue"], 1: COLORS["amber"]})
    axes[1, 0].scatter(metadata_df["total_reviews"].clip(upper=cap),
                       metadata_df["avg_rating"],
                       c=gem_colors, alpha=0.55, s=40, edgecolors="none")
    axes[1, 0].set_title("Reviews vs Avg Rating", color=COLORS["text"], fontweight="bold")
    axes[1, 0].set_xlabel("Total Reviews (95th pct cap)", color=COLORS["muted"])
    axes[1, 0].set_ylabel("Avg Rating", color=COLORS["muted"])
    blue_patch = mpatches.Patch(color=COLORS["blue"],  label="Normal")
    gold_patch = mpatches.Patch(color=COLORS["amber"], label="Hidden Gem 💎")
    axes[1, 0].legend(handles=[blue_patch, gold_patch], fontsize=8,
                      facecolor=COLORS["surface"], labelcolor=COLORS["text"])

    # 5 — Bayesian Avg by Category (box plot)
    cat_order = metadata_df.groupby("project_category")["bayesian_avg"].mean().sort_values().index
    data_box   = [
        metadata_df[metadata_df["project_category"] == c]["bayesian_avg"].dropna().values
        for c in cat_order
    ]
    bp = axes[1, 1].boxplot(
        data_box,
        labels=[c.split("/")[0].strip()[:18] for c in cat_order],
        patch_artist=True,
        medianprops={"color": COLORS["bg"], "linewidth": 2},
    )
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color); patch.set_alpha(0.8)
    axes[1, 1].set_title("Bayesian Avg by Category", color=COLORS["text"], fontweight="bold")
    axes[1, 1].tick_params(axis="x", rotation=15)

    # 6 — Normal vs Hidden Gem per Category
    gem_cats = metadata_df[metadata_df["is_hidden_gem"] == 1]["project_category"].value_counts()
    non_cats = metadata_df[metadata_df["is_hidden_gem"] == 0]["project_category"].value_counts()
    x_pos = range(len(cat_counts))
    w = 0.4
    axes[1, 2].bar([x - w / 2 for x in x_pos],
                   [non_cats.get(c, 0) for c in cat_counts.index],
                   w, label="Normal",      color=COLORS["blue"],  edgecolor="none", alpha=0.8)
    axes[1, 2].bar([x + w / 2 for x in x_pos],
                   [gem_cats.get(c, 0) for c in cat_counts.index],
                   w, label="Hidden Gem",  color=COLORS["amber"], edgecolor="none", alpha=0.8)
    axes[1, 2].set_xticks(list(x_pos))
    axes[1, 2].set_xticklabels(
        [c.split("/")[0].strip()[:12] for c in cat_counts.index],
        rotation=20, ha="right", fontsize=8, color=COLORS["muted"],
    )
    axes[1, 2].set_title("Normal vs Hidden Gem per Category",
                          color=COLORS["text"], fontweight="bold")
    axes[1, 2].legend(fontsize=8, facecolor=COLORS["surface"], labelcolor=COLORS["text"])

    plt.tight_layout()
    return fig
