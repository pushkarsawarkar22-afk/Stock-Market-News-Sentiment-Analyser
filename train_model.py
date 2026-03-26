"""
train_model.py — Custom NLP Sentiment Classifier
--------------------------------------------------
Phase 3: Trains a custom ML model on the Financial PhraseBank dataset
and evaluates it against VADER on real-time FinViz headlines.

Training data source:
  Financial PhraseBank (Malo et al., 2014)
  https://huggingface.co/datasets/financial_phrasebank
  OR local CSV with columns: sentence, label (positive/negative/neutral)

Pipeline:
  1. Load Financial PhraseBank (75% agreement split for clean labels)
  2. TF-IDF vectorisation
  3. Train Logistic Regression, Naive Bayes, SVM
  4. Evaluate with classification_report + confusion matrix
  5. Save best model + vectoriser with joblib

Usage:
    python train_model.py --phrasebank data/financial_phrasebank.csv \\
                          --out_dir models/
"""

import os
import argparse
import logging
import warnings
import joblib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download("vader_lexicon", quiet=True)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

LABEL_MAP = {"positive": 2, "neutral": 1, "negative": 0}
INV_LABEL = {v: k for k, v in LABEL_MAP.items()}
RANDOM_STATE = 42


# ── Data loading ───────────────────────────────────────────────────────────

def load_phrasebank(path: str) -> pd.DataFrame:
    """
    Load the Financial PhraseBank CSV.

    Expected columns: sentence, label
    Label values: positive / negative / neutral (strings)

    If using the HuggingFace dataset, run this first:
        from datasets import load_dataset
        ds = load_dataset("financial_phrasebank", "sentences_75agree")
        df = pd.DataFrame(ds["train"])
        df.to_csv("data/financial_phrasebank.csv", index=False)
    """
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    # Handle numeric labels (HuggingFace format: 0=negative, 1=neutral, 2=positive)
    if df["label"].dtype in [int, float, np.int64]:
        df["label"] = df["label"].map({0: "negative", 1: "neutral", 2: "positive"})

    df = df[df["label"].isin(["positive", "negative", "neutral"])].copy()
    df.dropna(subset=["sentence", "label"], inplace=True)
    df.rename(columns={"sentence": "text"}, inplace=True)
    logger.info(f"Phrasebank loaded: {len(df)} rows")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    return df


def generate_synthetic_data(n: int = 1000) -> pd.DataFrame:
    """
    Generate a minimal synthetic dataset for demonstration when
    the real Financial PhraseBank is unavailable.

    ⚠️  For a real project, use the actual Financial PhraseBank.
    """
    import random
    random.seed(42)

    pos_templates = [
        "{ticker} beats earnings expectations by {pct}%",
        "{ticker} reports record quarterly revenue",
        "{ticker} raises full-year guidance",
        "{ticker} shares surge after strong results",
        "Analysts upgrade {ticker} to buy",
        "{ticker} announces major share buyback program",
        "{ticker} dividend increased by {pct}%",
        "{ticker} signs landmark partnership deal",
    ]
    neg_templates = [
        "{ticker} misses earnings estimates",
        "{ticker} cuts guidance amid weak demand",
        "{ticker} shares fall after disappointing results",
        "Analysts downgrade {ticker} on growth concerns",
        "{ticker} reports wider than expected loss",
        "{ticker} faces regulatory scrutiny",
        "{ticker} lays off {pct}% of workforce",
        "{ticker} warns of margin pressure",
    ]
    neu_templates = [
        "{ticker} announces quarterly earnings date",
        "{ticker} files 10-K with the SEC",
        "{ticker} appoints new chief financial officer",
        "{ticker} to present at investor conference",
        "{ticker} updates investor relations website",
        "Trading update: {ticker} volume unchanged",
        "{ticker} board meeting scheduled for next month",
    ]

    tickers = ["AAPL", "TSLA", "JPM", "PFE", "AMZN", "MSFT", "GOOG", "META"]
    rows = []
    for _ in range(n // 3):
        t = random.choice(tickers)
        p = random.randint(2, 20)
        rows.append({"text": random.choice(pos_templates).format(ticker=t, pct=p),
                     "label": "positive"})
        rows.append({"text": random.choice(neg_templates).format(ticker=t, pct=p),
                     "label": "negative"})
        rows.append({"text": random.choice(neu_templates).format(ticker=t, pct=p),
                     "label": "neutral"})

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.warning("Using synthetic data — for real results, supply the Financial PhraseBank CSV.")
    return df


# ── Feature engineering ────────────────────────────────────────────────────

def build_tfidf_pipeline(clf, max_features: int = 10000) -> Pipeline:
    """Wrap a classifier in a TF-IDF pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
        )),
        ("clf", clf),
    ])


def add_vader_features(texts: pd.Series) -> np.ndarray:
    """Return VADER score array as supplementary features."""
    sia = SentimentIntensityAnalyzer()
    rows = [list(sia.polarity_scores(t).values()) for t in texts]
    return np.array(rows)  # shape: (n, 4) → compound, neg, neu, pos


# ── Training & evaluation ──────────────────────────────────────────────────

def train_all_models(df: pd.DataFrame) -> dict:
    """
    Train three classifiers, evaluate, return the best pipeline and metrics.
    """
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Linear SVM": LinearSVC(
            max_iter=2000, C=0.5, random_state=RANDOM_STATE, class_weight="balanced"
        ),
    }

    results = {}
    best_f1 = -1
    best_name = None
    best_pipeline = None

    for name, clf in classifiers.items():
        logger.info(f"Training {name}...")
        pipeline = build_tfidf_pipeline(clf)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=["positive", "neutral", "negative"])

        # Cross-validated F1
        cv_scores = cross_val_score(
            pipeline, X, y, cv=5, scoring="f1_weighted", n_jobs=-1
        )

        results[name] = {
            "pipeline": pipeline,
            "accuracy": acc,
            "f1_weighted": f1,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "report": report,
            "confusion_matrix": cm,
            "y_test": y_test,
            "y_pred": y_pred,
        }

        logger.info(
            f"  Accuracy={acc:.4f} | Weighted F1={f1:.4f} | "
            f"CV F1={cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_pipeline = pipeline

    logger.info(f"\n✓ Best model: {best_name} (weighted F1 = {best_f1:.4f})")
    results["_best"] = best_name
    results["_best_pipeline"] = best_pipeline
    return results


# ── Plots ──────────────────────────────────────────────────────────────────

def plot_confusion_matrices(results: dict, out_dir: str) -> None:
    model_names = [k for k in results if not k.startswith("_")]
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5))
    if len(model_names) == 1:
        axes = [axes]

    fig.suptitle("Confusion Matrices (Test Set)", fontsize=14, fontweight="bold")
    labels = ["positive", "neutral", "negative"]

    for ax, name in zip(axes, model_names):
        cm = results[name]["confusion_matrix"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            ax=ax, linewidths=0.5
        )
        f1 = results[name]["f1_weighted"]
        ax.set_title(f"{name}\nWeighted F1={f1:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    path = os.path.join(out_dir, "06_confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {path}")


def plot_model_comparison(results: dict, out_dir: str) -> None:
    model_names = [k for k in results if not k.startswith("_")]
    metrics = ["accuracy", "f1_weighted", "cv_f1_mean"]
    labels = ["Accuracy", "Weighted F1", "CV F1 (mean)"]

    x = np.arange(len(model_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [results[name][metric] for name in model_names]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Accuracy, F1, CV F1", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "07_model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {path}")


def plot_top_features(pipeline: Pipeline, n: int = 20, out_dir: str = ".") -> None:
    """Display top positive/negative TF-IDF features for Logistic Regression."""
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        logger.info("Top features plot only available for Logistic Regression / Linear SVM.")
        return

    vectoriser = pipeline.named_steps["tfidf"]
    feature_names = vectoriser.get_feature_names_out()
    classes = clf.classes_

    fig, axes = plt.subplots(1, len(classes), figsize=(6 * len(classes), 5))
    fig.suptitle("Top TF-IDF Features per Class", fontsize=14, fontweight="bold")
    colors = {"positive": "#55A868", "neutral": "#999999", "negative": "#C44E52"}

    for ax, (coef_row, label) in zip(axes, zip(clf.coef_, classes)):
        top_idx = np.argsort(coef_row)[-n:]
        top_words = feature_names[top_idx]
        top_scores = coef_row[top_idx]
        ax.barh(top_words, top_scores, color=colors.get(label, "#aaaaaa"), alpha=0.8)
        ax.set_title(f"{label.capitalize()}")
        ax.set_xlabel("Coefficient weight")
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "08_top_features.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {path}")


# ── Persist model ──────────────────────────────────────────────────────────

def save_model(pipeline: Pipeline, name: str, metrics: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "best_model.pkl")
    meta_path = os.path.join(out_dir, "model_metadata.json")

    joblib.dump(pipeline, model_path)
    meta = {
        "model_name": name,
        "accuracy": metrics["accuracy"],
        "f1_weighted": metrics["f1_weighted"],
        "cv_f1_mean": metrics["cv_f1_mean"],
        "cv_f1_std": metrics["cv_f1_std"],
        "classes": list(pipeline.classes_) if hasattr(pipeline, "classes_") else
                   list(pipeline.named_steps["clf"].classes_),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Model saved → {model_path}")
    logger.info(f"Metadata  → {meta_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def run_training(phrasebank_path: str | None, out_dir: str, model_out: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    if phrasebank_path and os.path.exists(phrasebank_path):
        df = load_phrasebank(phrasebank_path)
    else:
        logger.warning("Financial PhraseBank not found — using synthetic demo data.")
        df = generate_synthetic_data(n=1200)

    results = train_all_models(df)
    best_name = results["_best"]
    best_pipeline = results["_best_pipeline"]

    plot_confusion_matrices(results, out_dir)
    plot_model_comparison(results, out_dir)
    plot_top_features(best_pipeline, n=15, out_dir=out_dir)
    save_model(best_pipeline, best_name, results[best_name], model_out)

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for name in [k for k in results if not k.startswith("_")]:
        r = results[name]
        print(f"\n{name}")
        print(f"  Accuracy    : {r['accuracy']:.4f}")
        print(f"  Weighted F1 : {r['f1_weighted']:.4f}")
        print(f"  CV F1       : {r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.4f}")
    print(f"\nBest model: {best_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NLP sentiment classifier")
    parser.add_argument("--phrasebank", default=None,
                        help="Path to Financial PhraseBank CSV (sentence,label)")
    parser.add_argument("--out_dir", default="outputs/models",
                        help="Directory for evaluation plots")
    parser.add_argument("--model_out", default="models",
                        help="Directory to save trained model files")
    args = parser.parse_args()

    run_training(args.phrasebank, args.out_dir, args.model_out)
