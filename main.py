"""
main.py — Full NLP Sentiment Pipeline Orchestrator
----------------------------------------------------
Runs all four phases of the project in sequence:
  Phase 1 → Scrape FinViz headlines
  Phase 2 → Exploratory Data Analysis
  Phase 3 → Train custom NLP model
  Phase 4 → Live comparison: VADER vs Custom Model

Usage (recommended — full pipeline):
    python main.py --tickers AAPL TSLA JPM PFE AMZN

Skip already-done phases:
    python main.py --skip_scrape --skip_eda     # jump straight to compare
    python main.py --use_phrasebank data/financial_phrasebank.csv

All outputs land in outputs/ and models/
"""

import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Make utils importable regardless of CWD ────────────────────────────────
SRC_DIR = Path(__file__).parent / "utils"
sys.path.insert(0, str(SRC_DIR))

from scraper import scrape_tickers
from eda import run_eda
from train_model import run_training
from compare import run_comparison


def banner(title: str) -> None:
    width = 60
    logger.info("=" * width)
    logger.info(f"  {title}")
    logger.info("=" * width)


def main(args: argparse.Namespace) -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # ── Phase 1: Scrape ─────────────────────────────────────────────────────
    raw_path = "data/raw_headlines.csv"
    if args.skip_scrape:
        logger.info("Phase 1 skipped (--skip_scrape). Expecting: data/raw_headlines.csv")
    else:
        banner("PHASE 1 — Data Acquisition (FinViz Scraper)")
        df_raw = scrape_tickers(args.tickers, delay=args.delay)
        if df_raw.empty:
            logger.error("Scraping returned no data. Check your internet connection.")
            sys.exit(1)
        df_raw.to_csv(raw_path, index=False)
        logger.info(f"Phase 1 complete → {raw_path}")

    # ── Phase 2: EDA ────────────────────────────────────────────────────────
    if args.skip_eda:
        logger.info("Phase 2 skipped (--skip_eda).")
    else:
        banner("PHASE 2 — Exploratory Data Analysis")
        run_eda(input_path=raw_path, out_dir="outputs/eda")
        logger.info("Phase 2 complete → outputs/eda/")

    # ── Phase 3: Train model ────────────────────────────────────────────────
    if args.skip_train:
        logger.info("Phase 3 skipped (--skip_train).")
    else:
        banner("PHASE 3 — Custom NLP Model Training")
        phrasebank = args.use_phrasebank if args.use_phrasebank else None
        run_training(
            phrasebank_path=phrasebank,
            out_dir="outputs/models",
            model_out="models",
        )
        logger.info("Phase 3 complete → models/best_model.pkl")

    # ── Phase 4: Live comparison ────────────────────────────────────────────
    if args.skip_compare:
        logger.info("Phase 4 skipped (--skip_compare).")
    else:
        banner("PHASE 4 — Live Comparison: VADER vs Custom Model")
        run_comparison(
            tickers=args.tickers,
            model_path="models/best_model.pkl",
            out_dir="outputs/comparison",
            use_cached=raw_path if args.skip_scrape else None,
        )
        logger.info("Phase 4 complete → outputs/comparison/")

    logger.info("\n" + "=" * 60)
    logger.info("ALL PHASES COMPLETE")
    logger.info("=" * 60)
    logger.info("Output structure:")
    logger.info("  outputs/eda/            → EDA plots (Phases 1-2)")
    logger.info("  outputs/models/         → Model evaluation plots (Phase 3)")
    logger.info("  outputs/comparison/     → Live comparison results (Phase 4)")
    logger.info("  models/best_model.pkl   → Trained classifier")
    logger.info("  data/raw_headlines.csv  → Scraped headlines")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stock Market News Sentiment NLP Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core arguments
    parser.add_argument(
        "--tickers", nargs="+",
        default=["AAPL", "TSLA", "JPM", "PFE", "AMZN"],
        help="Stock tickers to analyse"
    )
    parser.add_argument(
        "--use_phrasebank",
        default=None,
        help="Path to Financial PhraseBank CSV for training (optional)"
    )
    parser.add_argument(
        "--delay", type=float, default=1.5,
        help="Seconds between FinViz requests"
    )

    # Phase skip flags
    parser.add_argument("--skip_scrape",  action="store_true", help="Skip Phase 1 (scraping)")
    parser.add_argument("--skip_eda",     action="store_true", help="Skip Phase 2 (EDA)")
    parser.add_argument("--skip_train",   action="store_true", help="Skip Phase 3 (training)")
    parser.add_argument("--skip_compare", action="store_true", help="Skip Phase 4 (comparison)")

    args = parser.parse_args()
    main(args)
