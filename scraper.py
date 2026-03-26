"""
scraper.py — FinViz News Scraper
---------------------------------
Scrapes financial news headlines from FinViz for one or more stock tickers.
Saves results to CSV for downstream EDA and modelling.

Usage:
    python scraper.py --tickers AAPL TSLA JPM PFE AMZN --output data/raw_headlines.csv
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import date, datetime
import time
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_news_table(ticker: str) -> BeautifulSoup | None:
    """Fetch the raw FinViz news table for a given ticker."""
    url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find(id="news-table")
        if table is None:
            logger.warning(f"No news table found for {ticker}")
        return table
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {ticker}: {e}")
        return None


def parse_news_table(table: BeautifulSoup, ticker: str) -> list[dict]:
    """
    Parse a FinViz news table and return structured records.

    FinViz quirk: date is printed once per day group; subsequent rows
    in the same day only show the time. We track current_date to fill forward.
    """
    records = []
    current_date = date.today().strftime("%b-%d-%y")

    for row in table.find_all("tr"):
        try:
            anchor = row.find("a")
            td = row.find("td")
            if not anchor or not td:
                continue

            title = anchor.get_text(strip=True)
            url = anchor.get("href", "")
            source_td = row.find_all("td")
            source = source_td[1].get_text(strip=True) if len(source_td) > 1 else ""

            raw = td.get_text(strip=True).split()
            if len(raw) == 1:
                time_str = raw[0]
            else:
                current_date = raw[0]
                time_str = raw[1]

            records.append({
                "ticker": ticker.upper(),
                "date": current_date,
                "time": time_str,
                "title": title,
                "source": source,
                "url": url,
                "scraped_at": datetime.now().isoformat(timespec="seconds"),
            })
        except Exception as e:
            logger.debug(f"Row parse error ({ticker}): {e}")
            continue

    return records


def scrape_tickers(tickers: list[str], delay: float = 1.5) -> pd.DataFrame:
    """
    Scrape all tickers and return a single consolidated DataFrame.

    Args:
        tickers: list of ticker symbols (e.g. ['AAPL', 'TSLA'])
        delay:   seconds to wait between requests (be polite to FinViz)
    """
    all_records = []
    for ticker in tickers:
        logger.info(f"Scraping {ticker}...")
        table = fetch_news_table(ticker)
        if table:
            rows = parse_news_table(table, ticker)
            all_records.extend(rows)
            logger.info(f"  → {len(rows)} headlines collected")
        time.sleep(delay)

    df = pd.DataFrame(all_records)
    if df.empty:
        logger.warning("No data collected.")
        return df

    # Parse dates properly
    df["date"] = pd.to_datetime(df["date"], format="%b-%d-%y", errors="coerce")
    df.dropna(subset=["date", "title"], inplace=True)
    df.sort_values(["ticker", "date", "time"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape FinViz news headlines")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "TSLA", "JPM", "PFE", "AMZN"],
                        help="Stock ticker symbols to scrape")
    parser.add_argument("--output", default="data/raw_headlines.csv",
                        help="Output CSV path")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Delay between requests in seconds")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = scrape_tickers(args.tickers, args.delay)

    if not df.empty:
        df.to_csv(args.output, index=False)
        logger.info(f"\nSaved {len(df)} records to '{args.output}'")
        print(df[["ticker", "date", "title"]].head(10).to_string(index=False))
