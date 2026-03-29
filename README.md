# Stock Market News Sentiment Analyser
### NLP Subject Project — Comparative Analysis: VADER Heuristics vs Custom ML Model

> **Base Repository:** `mrinankmj/Stock_News_Sentiment_Analysis`  
> **Extended by:** Pushkar Sawarkar

---
Show Image
Show Image
Show Image


## Project Overview

This project builds an end-to-end NLP pipeline for **real-time financial news sentiment analysis**. Starting from the open-source FinViz scraper as a base, it extends the original into a full research pipeline with:

1. **Automated headline scraping** from FinViz (no API key needed)
2. **Comprehensive EDA** — distribution analysis, n-gram frequency, baseline VADER scoring
3. **Custom ML sentiment classifier** trained on the Financial PhraseBank dataset
4. **Head-to-head live comparison** — VADER vs your custom model on the same real-time headlines

The central research question is: *Does a domain-adapted ML model outperform a general-purpose rule-based system (VADER) on financial news?*

---

## Directory Structure

```
stock_sentiment_nlp/
│
├── main.py                      # Orchestrates all 4 phases
│
├── utils/
│   ├── scraper.py               # Phase 1: FinViz headline scraper
│   ├── eda.py                   # Phase 2: Exploratory Data Analysis
│   ├── train_model.py           # Phase 3: TF-IDF + ML model training
│   └── compare.py               # Phase 4: Live VADER vs Custom comparison
│
├── data/
│   ├── raw_headlines.csv        # Output of Phase 1 (created on run)
│   └── financial_phrasebank.csv # Training data — see Setup below
│
├── models/
│   ├── best_model.pkl           # Saved sklearn Pipeline (created on run)
│   └── model_metadata.json      # Accuracy / F1 scores
│
├── outputs/
│   ├── eda/                     # EDA plots (Phase 2)
│   ├── models/                  # Model evaluation plots (Phase 3)
│   └── comparison/              # Live comparison plots + CSVs (Phase 4)
│
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the base repository (for reference)
```bash
git clone https://github.com/mrinankmj/Stock_News_Sentiment_Analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK data (one-time)
```python
import nltk
nltk.download(['punkt', 'stopwords', 'vader_lexicon', 'wordnet'])
```

### 4. Get the Financial PhraseBank (for Phase 3)
The Financial PhraseBank (Malo et al., 2014) is the gold-standard dataset for
financial sentiment classification. Use the 75% agreement split for clean labels.

**Option A — HuggingFace (recommended):**
```python
from datasets import load_dataset
ds = load_dataset("financial_phrasebank", "sentences_75agree")
import pandas as pd
df = pd.DataFrame(ds["train"])
df.to_csv("data/financial_phrasebank.csv", index=False)
```

**Option B — Kaggle:**  
Search "Financial PhraseBank" on Kaggle and download the CSV.

> If the PhraseBank is unavailable, the training phase uses synthetic demo data.
> For your actual college submission, use the real dataset.

---

## Running the Pipeline

### Full pipeline (all 4 phases):
```bash
python main.py --tickers AAPL TSLA JPM PFE AMZN
```

### With the Financial PhraseBank:
```bash
python main.py --tickers AAPL TSLA JPM PFE AMZN \
               --use_phrasebank data/financial_phrasebank.csv
```

### Run individual phases:
```bash
# Phase 1 only
python utils/scraper.py --tickers AAPL TSLA --output data/raw_headlines.csv

# Phase 2 only (needs Phase 1 output)
python utils/eda.py --input data/raw_headlines.csv --out_dir outputs/eda

# Phase 3 only (train the model)
python utils/train_model.py --phrasebank data/financial_phrasebank.csv

# Phase 4 only (live comparison)
python utils/compare.py --tickers AAPL TSLA --model_path models/best_model.pkl
```

---

## Outputs

### Phase 2 — EDA Plots
| File | Description |
|------|-------------|
| `01_headline_lengths.png` | Violin + box plot of headline length by ticker |
| `02_top_ngrams.png` | Top 20 unigrams and bigrams |
| `03_vader_distribution.png` | VADER score histogram + pie + per-ticker stacked bar |
| `04_temporal_sentiment.png` | 7-day rolling VADER compound score per ticker |
| `05_correlation_heatmap.png` | Feature correlation heatmap |

### Phase 3 — Model Evaluation Plots
| File | Description |
|------|-------------|
| `06_confusion_matrices.png` | Side-by-side confusion matrices for all 3 classifiers |
| `07_model_comparison.png` | Accuracy / F1 / CV-F1 bar chart comparison |
| `08_top_features.png` | Most predictive TF-IDF features per sentiment class |

### Phase 4 — Live Comparison Plots
| File | Description |
|------|-------------|
| `09_agreement_overview.png` | Agree/disagree pie + per-class bar chart |
| `10_prediction_matrix.png` | Cross-tabulation heatmap: VADER vs Custom |
| `11_compound_by_custom.png` | VADER scores grouped by custom model label |
| `12_daily_dual_trend.png` | Dual-axis: VADER compound vs custom % positive |
| `disagreement_cases.csv` | Top 15 headlines where models strongly disagreed |
| `agreement_summary.json` | Overall agreement rate + Cohen's Kappa |

---

## Technical Approach

### Text Preprocessing
- Lower-casing, punctuation removal, whitespace normalisation
- NLTK tokenisation + WordNet lemmatisation
- Standard English stopwords + domain-specific financial stopwords removed

### VADER (Baseline)
VADER (Valence Aware Dictionary and sEntiment Reasoner) is a **rule-based** model
tuned on social media text. It uses a lexicon of sentiment-bearing words with
manually calibrated intensifiers. While fast and interpretable, it was not designed
for financial domain jargon.

**Threshold:** compound ≥ 0.05 → positive | ≤ −0.05 → negative | else → neutral

### Custom Model (TF-IDF + ML)
Three classifiers are trained and the best is saved:

| Model | Key strength |
|-------|-------------|
| **Logistic Regression** | Linear, interpretable feature weights |
| **Naive Bayes** | Fast, strong prior on term frequencies |
| **Linear SVM** | Robust on high-dimensional sparse text |

**Feature engineering:** TF-IDF with unigrams and bigrams, sublinear TF scaling,
min_df=2 to ignore hapax legomena, up to 10,000 features.

### Evaluation Metrics
- **Accuracy** — overall correctness
- **Weighted F1** — handles class imbalance (neutral headlines dominate)
- **5-fold cross-validated F1** — guards against lucky train/test splits
- **Cohen's Kappa** — inter-model agreement (Phase 4)

---

## Key Findings (Template — fill in with your results)

| Metric | VADER | Custom Model |
|--------|-------|-------------|
| Accuracy on phrasebank | — | **XX.X%** |
| Weighted F1 | — | **X.XXX** |
| % positive (live FinViz) | XX.X% | XX.X% |
| Agreement with other model | — | **XX.X%** |
| Cohen's Kappa | — | **0.XXX** |

**Key observation:** VADER tends to classify financial jargon such as _"beats estimates"_
as neutral (generic verb), while the custom model — trained on domain-specific data —
correctly identifies the phrase as positive. See `disagreement_cases.csv`.

---

## References

1. Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). *Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts.* Journal of the American Society for Information Science and Technology, 65(4), 782–796.
2. Hutto, C.J. & Gilbert, E.E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.* ICWSM-14.
3. Base repository: [mrinankmj/Stock_News_Sentiment_Analysis](https://github.com/mrinankmj/Stock_News_Sentiment_Analysis)
4. Scikit-learn: Machine Learning in Python. *JMLR* 12, pp. 2825-2830, 2011.

---

## Academic Integrity

This project extends an open-source repository (credited above) with significant
original work: comprehensive EDA, custom model development, and comparative analysis.
All external data sources are cited. The Financial PhraseBank is used for academic,
non-commercial purposes under its licence terms.
