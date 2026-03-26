"""
compare.py — Real-Time Model Comparison: VADER vs Custom NLP
-------------------------------------------------------------
Phase 4: The live test.

Scrapes a fresh batch of headlines from FinViz, runs both VADER and the
trained custom model on the same data, computes agreement rate, and
surfaces the most interesting disagreement cases for manual analysis.

Usage:
    python compare.py --tickers AAPL TSLA JPM PFE AMZN \\
                      --model_path models/best_model.pkl \\
                      --out_dir outputs/comparison
"""

import os
import argparse
import logging
import warnings
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Local modules
import sys
sys.path.insert(0, os.path.dirname(__file__))
from scraper import scrape_tickers

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

import nltk
nltk.download("vader_lexicon", quiet=True)

# ── Inference helpers ──────────────────────────────────────────────────────

def vader_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    return "neutral"


def run_vader(texts: pd.Series) -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()
    scores = texts.apply(lambda t: sia.polarity_scores(t))
    return pd.DataFrame({
        "vader_compound": scores.apply(lambda s: s["compound"]),
        "vader_label": scores.apply(lambda s: vader_label(s["compound"])),
    })


def run_custom_model(texts: pd.Series, pipeline) -> pd.DataFrame:
    predictions = pipeline.predict(texts)
    # Get probabilities if available
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        probs = pipeline.predict_proba(texts).max(axis=1)
    else:
        # LinearSVC: use decision function distance as confidence proxy
        dec = pipeline.decision_function(texts)
        probs = np.abs(dec).max(axis=1)
        # Normalise to 0-1
        probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)

    return pd.DataFrame({
        "custom_label": predictions,
        "custom_confidence": probs,
    })


# ── Agreement analysis ─────────────────────────────────────────────────────

def compute_agreement(df: pd.DataFrame) -> dict:
    agree_mask = df["vader_label"] == df["custom_label"]
    agree_rate = agree_mask.mean()

    # Cohen's Kappa
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(df["vader_label"], df["custom_label"])

    # Per-class agreement
    per_class = {}
    for label in ["positive", "neutral", "negative"]:
        mask = df["vader_label"] == label
        if mask.sum() == 0:
            continue
        per_class[label] = (df.loc[mask, "custom_label"] == label).mean()

    return {
        "overall_agreement": agree_rate,
        "cohen_kappa": kappa,
        "per_class_agreement": per_class,
        "n_agree": agree_mask.sum(),
        "n_disagree": (~agree_mask).sum(),
        "n_total": len(df),
    }


def extract_disagreements(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Return headlines where VADER and custom model disagree most clearly."""
    disagree = df[df["vader_label"] != df["custom_label"]].copy()
    # Sort by: high-confidence custom predictions first (most interesting)
    disagree = disagree.sort_values("custom_confidence", ascending=False)
    cols = ["ticker", "title", "vader_compound", "vader_label",
            "custom_label", "custom_confidence"]
    return disagree[cols].head(n)


# ── Plots ──────────────────────────────────────────────────────────────────

def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {path}")


def plot_agreement_overview(agreement: dict, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("VADER vs Custom Model — Agreement Analysis", fontsize=14, fontweight="bold")

    # Pie: agree vs disagree
    ax = axes[0]
    n_agree = agreement["n_agree"]
    n_disagree = agreement["n_disagree"]
    ax.pie(
        [n_agree, n_disagree],
        labels=[f"Agree ({n_agree})", f"Disagree ({n_disagree})"],
        colors=["#55A868", "#C44E52"],
        autopct="%1.1f%%",
        startangle=90,
    )
    kappa = agreement["cohen_kappa"]
    ax.set_title(f"Overall agreement rate\n(Cohen's κ = {kappa:.3f})")

    # Bar: per-class agreement
    ax = axes[1]
    per_class = agreement["per_class_agreement"]
    labels = list(per_class.keys())
    vals = [per_class[l] * 100 for l in labels]
    colors = {"positive": "#55A868", "neutral": "#999999", "negative": "#C44E52"}
    bar_colors = [colors.get(l, "#aaaaaa") for l in labels]
    bars = ax.bar(labels, vals, color=bar_colors, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Agreement %")
    ax.set_title("Per-class agreement rate\n(where VADER predicted that class)")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(50, color="gray", linestyle="--", alpha=0.4)

    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, "09_agreement_overview.png"))


def plot_prediction_matrix(df: pd.DataFrame, out_dir: str) -> None:
    """Cross-tabulation heatmap of VADER labels vs custom model labels."""
    ct = pd.crosstab(df["vader_label"], df["custom_label"],
                     rownames=["VADER"], colnames=["Custom Model"])
    # Ensure consistent column order
    for col in ["positive", "neutral", "negative"]:
        if col not in ct.columns:
            ct[col] = 0
    ct = ct.reindex(index=["positive", "neutral", "negative"],
                    columns=["positive", "neutral", "negative"],
                    fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "count"})
    ax.set_title("Prediction Matrix: VADER → Custom Model", fontsize=13, fontweight="bold")
    save_fig(fig, os.path.join(out_dir, "10_prediction_matrix.png"))


def plot_compound_by_custom_label(df: pd.DataFrame, out_dir: str) -> None:
    """Box plot: VADER compound score grouped by custom model prediction."""
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["positive", "neutral", "negative"]
    palette = {"positive": "#55A868", "neutral": "#999999", "negative": "#C44E52"}

    data_to_plot = [
        df[df["custom_label"] == label]["vader_compound"].values
        for label in order
    ]
    bp = ax.boxplot(data_to_plot, labels=order, patch_artist=True)
    for patch, label in zip(bp["boxes"], order):
        patch.set_facecolor(palette[label])
        patch.set_alpha(0.7)

    ax.axhline(0.05, color="green", linestyle="--", alpha=0.5, label="VADER pos threshold")
    ax.axhline(-0.05, color="red", linestyle="--", alpha=0.5, label="VADER neg threshold")
    ax.set_ylabel("VADER compound score")
    ax.set_xlabel("Custom model prediction")
    ax.set_title("VADER Score Distribution by Custom Model Label", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, os.path.join(out_dir, "11_compound_by_custom.png"))


def plot_daily_dual_trend(df: pd.DataFrame, out_dir: str) -> None:
    """Dual-line chart: daily mean VADER compound vs custom model positive rate."""
    df2 = df.copy()
    df2["custom_pos_flag"] = (df2["custom_label"] == "positive").astype(float)

    daily = df2.groupby("date").agg(
        vader_mean=("vader_compound", "mean"),
        custom_pos_rate=("custom_pos_flag", "mean"),
    ).sort_index()

    fig, ax1 = plt.subplots(figsize=(14, 5))
    fig.suptitle("Daily Sentiment Trend: VADER vs Custom Model", fontsize=14, fontweight="bold")

    color_vader = "#4C72B0"
    color_custom = "#DD8452"

    ax1.plot(daily.index, daily["vader_mean"], color=color_vader, linewidth=2,
             label="VADER mean compound", marker="o", markersize=4)
    ax1.fill_between(daily.index, 0, daily["vader_mean"], alpha=0.1, color=color_vader)
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("VADER compound score", color=color_vader)
    ax1.tick_params(axis="y", labelcolor=color_vader)
    ax1.set_xlabel("Date")

    ax2 = ax1.twinx()
    ax2.plot(daily.index, daily["custom_pos_rate"] * 100, color=color_custom,
             linewidth=2, label="Custom: % positive", linestyle="--", marker="s", markersize=4)
    ax2.set_ylabel("Custom model: % positive headlines", color=color_custom)
    ax2.tick_params(axis="y", labelcolor=color_custom)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    save_fig(fig, os.path.join(out_dir, "12_daily_dual_trend.png"))


# ── Main ───────────────────────────────────────────────────────────────────

def run_comparison(tickers: list, model_path: str, out_dir: str, use_cached: str | None) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 1. Get live data
    if use_cached and os.path.exists(use_cached):
        logger.info(f"Using cached data: {use_cached}")
        df = pd.read_csv(use_cached, parse_dates=["date"])
    else:
        logger.info("Scraping live headlines from FinViz...")
        df = scrape_tickers(tickers)
        cache_path = os.path.join(out_dir, "live_headlines.csv")
        df.to_csv(cache_path, index=False)
        logger.info(f"Cached to: {cache_path}")

    logger.info(f"Dataset: {len(df)} headlines across {df['ticker'].nunique()} tickers")

    # 2. Load trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run train_model.py first."
        )
    pipeline = joblib.load(model_path)
    logger.info(f"Model loaded: {model_path}")

    # 3. Run both models
    logger.info("Running VADER...")
    vader_df = run_vader(df["title"])
    logger.info("Running custom model...")
    custom_df = run_custom_model(df["title"], pipeline)

    df = pd.concat([df, vader_df, custom_df], axis=1)

    # 4. Agreement analysis
    agreement = compute_agreement(df)
    logger.info(
        f"\nAgreement: {agreement['overall_agreement']*100:.1f}% "
        f"| Cohen's κ = {agreement['cohen_kappa']:.3f}"
    )

    # Save JSON summary
    agreement_copy = {k: v for k, v in agreement.items()}
    with open(os.path.join(out_dir, "agreement_summary.json"), "w") as f:
        json.dump(agreement_copy, f, indent=2)

    # 5. Disagreement cases
    disagreements = extract_disagreements(df, n=15)
    disagreements_path = os.path.join(out_dir, "disagreement_cases.csv")
    disagreements.to_csv(disagreements_path, index=False)
    logger.info(f"Saved {len(disagreements)} disagreement cases → {disagreements_path}")

    # 6. Plots
    plot_agreement_overview(agreement, out_dir)
    plot_prediction_matrix(df, out_dir)
    plot_compound_by_custom_label(df, out_dir)
    if "date" in df.columns and df["date"].notna().sum() > 2:
        plot_daily_dual_trend(df, out_dir)

    # 7. Save enriched output
    final_path = os.path.join(out_dir, "comparison_results.csv")
    df.to_csv(final_path, index=False)

    # 8. Console summary
    print("\n" + "=" * 70)
    print("LIVE COMPARISON RESULTS")
    print("=" * 70)
    print(f"Total headlines analysed : {agreement['n_total']}")
    print(f"Models agree             : {agreement['n_agree']} ({agreement['overall_agreement']*100:.1f}%)")
    print(f"Models disagree          : {agreement['n_disagree']}")
    print(f"Cohen's Kappa            : {agreement['cohen_kappa']:.4f}")
    print("\nPer-class agreement:")
    for label, rate in agreement["per_class_agreement"].items():
        print(f"  {label:<10} : {rate*100:.1f}%")

    print("\nTop 10 Disagreement Cases (VADER vs Custom Model):")
    print("-" * 70)
    for _, row in disagreements.head(10).iterrows():
        print(f"[{row['ticker']}] {row['title'][:60]}...")
        print(f"  VADER={row['vader_label']:>8} ({row['vader_compound']:+.3f}) | "
              f"Custom={row['custom_label']:>8} (conf={row['custom_confidence']:.2f})")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare VADER vs custom NLP model")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "TSLA", "JPM", "PFE", "AMZN"])
    parser.add_argument("--model_path", default="models/best_model.pkl")
    parser.add_argument("--out_dir", default="outputs/comparison")
    parser.add_argument("--use_cached", default=None,
                        help="Path to pre-scraped CSV to skip live scraping")
    args = parser.parse_args()

    run_comparison(args.tickers, args.model_path, args.out_dir, args.use_cached)
