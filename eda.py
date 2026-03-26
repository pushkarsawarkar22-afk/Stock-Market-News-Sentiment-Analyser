"""
eda.py — Exploratory Data Analysis for Financial News Headlines
---------------------------------------------------------------
Phase 2 of the NLP project pipeline.

Performs:
  1. Text preprocessing (cleaning, stopword removal, lemmatisation)
  2. Distribution analysis (headline lengths per ticker)
  3. Frequency analysis (top N-grams, word clouds)
  4. VADER baseline scoring + sentiment distribution plots
  5. Temporal sentiment trend charts

Run after scraper.py has produced data/raw_headlines.csv

Usage:
    python eda.py --input data/raw_headlines.csv --out_dir outputs/eda
"""

import os
import re
import argparse
import warnings
import logging
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Download NLTK assets (safe to call multiple times)
for pkg in ["punkt", "stopwords", "vader_lexicon", "wordnet", "averaged_perceptron_tagger"]:
    nltk.download(pkg, quiet=True)

# ── Colour palette (one per ticker for consistent styling) ─────────────────
TICKER_COLORS = {
    "AAPL": "#4C72B0",
    "TSLA": "#DD8452",
    "JPM": "#55A868",
    "PFE": "#C44E52",
    "AMZN": "#8172B3",
}
DEFAULT_COLOR = "#999999"

# ── Preprocessing ──────────────────────────────────────────────────────────

FINANCIAL_STOP_WORDS = {
    "stock", "share", "shares", "market", "markets", "company", "companies",
    "quarter", "report", "reports", "reported", "earnings", "revenue",
    "billion", "million", "year", "q1", "q2", "q3", "q4", "say", "says",
    "said", "new", "also", "would", "could", "may", "one", "two", "three",
}


def clean_text(text: str) -> str:
    """Lower-case, remove non-alpha characters, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatise(text: str, stop_words: set) -> list[str]:
    """Tokenise → remove stopwords → lemmatise."""
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t.isalpha() and t not in stop_words and len(t) > 2
    ]
    return tokens


def build_stop_words() -> set:
    english_stops = set(stopwords.words("english"))
    return english_stops | FINANCIAL_STOP_WORDS


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add cleaned text columns to the DataFrame."""
    stop_words = build_stop_words()
    df = df.copy()
    df["clean_title"] = df["title"].apply(clean_text)
    df["tokens"] = df["clean_title"].apply(lambda x: tokenize_and_lemmatise(x, stop_words))
    df["word_count"] = df["tokens"].apply(len)
    df["char_count"] = df["title"].apply(len)
    return df


# ── VADER scoring ──────────────────────────────────────────────────────────

def apply_vader(df: pd.DataFrame) -> pd.DataFrame:
    """Add VADER sentiment columns: compound, pos, neu, neg, label."""
    sia = SentimentIntensityAnalyzer()
    scores = df["title"].apply(lambda t: sia.polarity_scores(t))
    df = df.copy()
    df["vader_compound"] = scores.apply(lambda s: s["compound"])
    df["vader_pos"] = scores.apply(lambda s: s["pos"])
    df["vader_neu"] = scores.apply(lambda s: s["neu"])
    df["vader_neg"] = scores.apply(lambda s: s["neg"])
    df["vader_label"] = df["vader_compound"].apply(
        lambda c: "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
    )
    return df


# ── Plot helpers ───────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {path}")


# ── EDA plots ──────────────────────────────────────────────────────────────

def plot_headline_length_distribution(df: pd.DataFrame, out_dir: str) -> None:
    """Box + violin plot of character lengths per ticker."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Headline Length Distribution by Ticker", fontsize=14, fontweight="bold")

    tickers = sorted(df["ticker"].unique())
    palette = [TICKER_COLORS.get(t, DEFAULT_COLOR) for t in tickers]

    # Violin
    ax = axes[0]
    for i, ticker in enumerate(tickers):
        data = df[df["ticker"] == ticker]["char_count"]
        parts = ax.violinplot([data], positions=[i], showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(palette[i])
            pc.set_alpha(0.7)
        parts["cmedians"].set_color(palette[i])

    ax.set_xticks(range(len(tickers)))
    ax.set_xticklabels(tickers)
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Character count")
    ax.set_title("Violin: character count")
    ax.grid(axis="y", alpha=0.3)

    # Word count box
    ax = axes[1]
    groups = [df[df["ticker"] == t]["word_count"].values for t in tickers]
    bp = ax.boxplot(groups, patch_artist=True, labels=tickers)
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Token count (after cleaning)")
    ax.set_title("Box: cleaned token count")
    ax.grid(axis="y", alpha=0.3)

    save_fig(fig, os.path.join(out_dir, "01_headline_lengths.png"))


def plot_top_ngrams(df: pd.DataFrame, out_dir: str, n: int = 20) -> None:
    """Bar charts of top unigrams and bigrams across all headlines."""
    all_tokens = [t for tokens in df["tokens"] for t in tokens]

    # Unigrams
    unigram_freq = Counter(all_tokens).most_common(n)
    words_u, counts_u = zip(*unigram_freq)

    # Bigrams
    bigrams = [f"{all_tokens[i]} {all_tokens[i+1]}" for i in range(len(all_tokens) - 1)]
    bigram_freq = Counter(bigrams).most_common(n)
    words_b, counts_b = zip(*bigram_freq)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Top {n} Most Frequent Terms Across All Headlines", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.barh(words_u[::-1], counts_u[::-1], color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.set_title("Unigrams")
    ax.set_xlabel("Frequency")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    ax.barh(words_b[::-1], counts_b[::-1], color="#DD8452", alpha=0.8, edgecolor="white")
    ax.set_title("Bigrams")
    ax.set_xlabel("Frequency")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, "02_top_ngrams.png"))


def plot_vader_distribution(df: pd.DataFrame, out_dir: str) -> None:
    """Histogram + pie of VADER sentiment scores and labels."""
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig)
    fig.suptitle("VADER Baseline Sentiment Distribution", fontsize=14, fontweight="bold")

    # Histogram of compound scores
    ax1 = fig.add_subplot(gs[0])
    ax1.hist(df["vader_compound"], bins=40, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax1.axvline(0.05, color="green", linestyle="--", label="Positive threshold")
    ax1.axvline(-0.05, color="red", linestyle="--", label="Negative threshold")
    ax1.axvline(df["vader_compound"].mean(), color="orange", linestyle="-",
                label=f"Mean={df['vader_compound'].mean():.3f}")
    ax1.set_title("Compound score distribution")
    ax1.set_xlabel("VADER compound score")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Pie chart
    ax2 = fig.add_subplot(gs[1])
    label_counts = df["vader_label"].value_counts()
    colors = {"positive": "#55A868", "neutral": "#999999", "negative": "#C44E52"}
    wedge_colors = [colors.get(l, "#aaaaaa") for l in label_counts.index]
    ax2.pie(label_counts, labels=label_counts.index, autopct="%1.1f%%",
            colors=wedge_colors, startangle=140)
    ax2.set_title("Label proportions (all tickers)")

    # Per-ticker stacked bar
    ax3 = fig.add_subplot(gs[2])
    pivot = (df.groupby(["ticker", "vader_label"])
               .size()
               .unstack(fill_value=0)
               .reindex(columns=["positive", "neutral", "negative"], fill_value=0))
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_pct.plot(kind="bar", stacked=True, ax=ax3,
                   color=["#55A868", "#999999", "#C44E52"], edgecolor="white")
    ax3.set_title("Sentiment % per ticker")
    ax3.set_xlabel("")
    ax3.set_ylabel("Percentage")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(axis="y", alpha=0.3)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)

    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, "03_vader_distribution.png"))


def plot_temporal_sentiment(df: pd.DataFrame, out_dir: str) -> None:
    """Rolling average VADER compound score per ticker over time."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title("7-Day Rolling Average VADER Sentiment by Ticker", fontsize=13, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    for ticker in sorted(df["ticker"].unique()):
        sub = (df[df["ticker"] == ticker]
               .groupby("date")["vader_compound"]
               .mean()
               .sort_index())
        if len(sub) < 3:
            continue
        rolling = sub.rolling(window=7, min_periods=1).mean()
        color = TICKER_COLORS.get(ticker, DEFAULT_COLOR)
        ax.plot(rolling.index, rolling.values, label=ticker, color=color, linewidth=2)
        ax.fill_between(rolling.index, 0, rolling.values,
                        alpha=0.08, color=color)

    ax.set_xlabel("Date")
    ax.set_ylabel("Average compound score")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_facecolor("#fafafa")
    save_fig(fig, os.path.join(out_dir, "04_temporal_sentiment.png"))


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    """Heatmap of correlations between numeric EDA features."""
    num_cols = ["char_count", "word_count", "vader_compound",
                "vader_pos", "vader_neu", "vader_neg"]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=-1, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    save_fig(fig, os.path.join(out_dir, "05_correlation_heatmap.png"))


# ── Main ───────────────────────────────────────────────────────────────────

def run_eda(input_path: str, out_dir: str) -> pd.DataFrame:
    logger.info("Loading dataset...")
    df = pd.read_csv(input_path, parse_dates=["date"])
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Tickers: {sorted(df['ticker'].unique())}")
    logger.info(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    logger.info(f"  Missing titles: {df['title'].isna().sum()}")

    logger.info("Preprocessing text...")
    df = preprocess_dataframe(df)

    logger.info("Applying VADER baseline...")
    df = apply_vader(df)

    logger.info("Generating EDA plots...")
    plot_headline_length_distribution(df, out_dir)
    plot_top_ngrams(df, out_dir)
    plot_vader_distribution(df, out_dir)
    plot_temporal_sentiment(df, out_dir)
    plot_correlation_heatmap(df, out_dir)

    # Save enriched CSV
    enriched_path = os.path.join(out_dir, "headlines_enriched.csv")
    df.to_csv(enriched_path, index=False)
    logger.info(f"\nEnriched dataset saved → {enriched_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("EDA SUMMARY")
    print("=" * 60)
    print(f"Total headlines   : {len(df)}")
    print(f"Tickers           : {', '.join(sorted(df['ticker'].unique()))}")
    print(f"Avg headline len  : {df['char_count'].mean():.0f} chars")
    print(f"Avg VADER compound: {df['vader_compound'].mean():.4f}")
    print("\nVADER label distribution:")
    print(df["vader_label"].value_counts(normalize=True).mul(100).round(1).to_string())
    print("\nPer-ticker mean compound score:")
    print(df.groupby("ticker")["vader_compound"].mean().round(4).to_string())

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA for financial news headlines")
    parser.add_argument("--input", default="data/raw_headlines.csv")
    parser.add_argument("--out_dir", default="outputs/eda")
    args = parser.parse_args()

    run_eda(args.input, args.out_dir)
