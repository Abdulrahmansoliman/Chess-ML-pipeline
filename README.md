# ♟️ Chess ML Pipeline

A machine learning pipeline built on my personal chess game data to predict game outcomes from early opening behavior and context features.

## Why Chess?

I've been playing chess on and off since childhood, and my progress has stalled. Instead of just grinding games, I decided to use machine learning to analyze my own play — understand which opening patterns lead to wins, what behaviors hurt my chances, and where to focus my improvement.

This project uses **~1,450 real games** from three accounts spanning different eras of my chess journey:

| Era | Platform | Account | Games | Rating Range |
|-----|----------|---------|-------|--------------|
| Childhood | Chess.com | Abdulrahmansoli | 1,013 | ~700–815 |
| High School | Lichess | Abdulrahmansoli | 259 | ~1,300–1,578 |
| Present | Chess.com | AbdulrahmanSoliman2 | 169 | ~1,000–1,200 |

## Pipeline Overview

```
Raw PGN Data → Data Cleaning → Feature Extraction → Train/Test Split → Model Training → Evaluation
 (1,451 games)   (1,250 games)    (11 features)        (80/20)         (LogReg + RF)    (Accuracy, PR, ROC)
```

### Data Collection
Chess.com games are downloaded automatically using [`download_chess_pgn.py`](download_chess_pgn.py), which calls the Chess.com public API with retry logic and rate limiting. Lichess games were downloaded directly.

### Data Cleaning
- Removed bullet games (< 3 min) — too fast for meaningful analysis
- Removed early disconnects (< 10 plies)
- Removed empty/incomplete games
- Result: **1,250 games** ready for analysis

### Feature Engineering
Instead of using raw move sequences (99.9% unique — too sparse), I compress the first **N = 14 plies** (~7 moves) into behavioral features:

**Context features** (known before game):
- `rating_diff` — my rating minus opponent's
- `base_sec`, `inc_sec` — time control
- `is_white` — color played

**Behavioral features** (from first N plies):
- `my_castled_by_N`, `opp_castled_by_N` — king safety
- `my_queen_moved_by_N` — early queen development
- `my_pawn_moves_N`, `my_minor_moves_N` — opening style
- `my_captures_N`, `my_checks_N` — aggression

### Models

| Model | CV Accuracy | Notes |
|-------|-------------|-------|
| Logistic Regression (sklearn) | ~0.719 | Interpretable baseline |
| Logistic Regression (from scratch) | ~0.713 | Manual gradient descent implementation |
| **Random Forest** | **~0.795** | Best performer — captures non-linear patterns |

### Key Findings
- **`rating_diff` dominates** predictions (expected — opponent strength matters most)
- **Early queen moves hurt** — associated with lower win probability
- **Random Forest outperforms LogReg by ~13 points**, indicating non-linear feature interactions matter
- **Behavioral features add signal** beyond just context (rating, time, color)

## Notebook Structure

The main analysis is in [`chess_pipeline_v2.ipynb`](chess_pipeline_v2.ipynb):

1. **Section 1**: Motivation & project overview
2. **Section 2**: Data ingestion (OOP: `PGNParser`, `PGNLoader`)
3. **Section 3**: Data cleaning (`DataCleaner` class)
4. **Section 4**: Feature extraction design & `FeatureExtractor` class
5. **Section 6**: Model implementation — math (sigmoid, cross-entropy, gradients), sklearn baseline, `ModelEvaluator` class
6. **Section 8**: Logistic Regression from scratch — manual gradient descent with loss tracking
7. **Section 7**: Experiments — ablation study, N-plies comparison, Random Forest
8. **Section 9**: Bayesian perspective — MAP/regularization connection
9. **Data Sources**: API references & profile details

### Visualizations
- Pipeline flowchart
- Logistic regression architecture diagram (neural network view)
- Gradient descent training loop diagram
- Confusion matrices, PR curves, ROC curves, calibration curves, feature importance plots
- Loss convergence curve (from-scratch model)

## Tech Stack

- **Python 3.12**
- **Core**: NumPy, Pandas, Matplotlib, Seaborn
- **ML**: scikit-learn (LogisticRegression, RandomForestClassifier, StandardScaler, cross_val_score)
- **Chess**: python-chess (PGN parsing, board simulation, move validation)
- **Data**: Chess.com Public API, Lichess API

## How to Run

```bash
# Clone the repo
git clone https://github.com/Abdulrahmansoliman/Chess-ML-pipeline.git
cd Chess-ML-pipeline

# Create virtual environment
python -m venv chess_venv
chess_venv\Scripts\activate  # Windows
# source chess_venv/bin/activate  # macOS/Linux

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn python-chess requests

# (Optional) Download fresh PGN data
python download_chess_pgn.py

# Open the notebook
jupyter notebook chess_pipeline_v2.ipynb
```

## Repo Structure

```
Chess-ML-pipeline/
├── chess_pipeline_v2.ipynb    # Main analysis notebook
├── download_chess_pgn.py      # Chess.com PGN downloader script
├── PGN-data/                  # Raw PGN game files
├── data_processed/            # Processed data outputs
├── .gitignore
└── README.md
```

## Course

**CS156** — Machine Learning Pipeline (Assignment 1)

---

*Built with real chess data, real frustration, and a genuine desire to stop losing.*
