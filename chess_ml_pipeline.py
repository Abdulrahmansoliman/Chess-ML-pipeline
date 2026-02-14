# %% [markdown]
# # Chess Game Analysis: Predicting Play Style Evolution Across Three Eras
# 
# **Course:** CS156 - Machine Learning Pipeline Assignment 1  
# **Date:** February 2026
# 
# ## TL;DR
# This notebook analyzes ~1,400 personal chess games across three life periods (childhood, high school, present).
# We extract early-game features and train classifiers to predict which "era" a game belongs to.

# %% [markdown]
# ---
# # Section 1: Data Description
# 
# | Era | Platform | Username | Games | Rating |
# |-----|----------|----------|-------|--------|
# | Childhood | Chess.com | Abdulrahmansoli | ~1,013 | ~700-815 |
# | High School | Lichess | Abdulrahmansoli | ~259 | ~1,300-1,578 |
# | Present | Chess.com | AbdulrahmanSoliman2 | ~169 | ~1,000-1,200 |

# %% [markdown]
# ---
# # Section 2: Data Ingestion

# %%
import os, re, warnings
from datetime import datetime
import chess
import chess.pgn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

warnings.filterwarnings('ignore')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# %%
ACCOUNTS = [
    {"account_id": "childhood", "era": "childhood", "platform": "chesscom",
     "username": "Abdulrahmansoli", "pgn_path": "PGN-data/chess_com_games_childhood_era_elo_800.pgn"},
    {"account_id": "highschool", "era": "highschool", "platform": "lichess",
     "username": "Abdulrahmansoli", "pgn_path": "PGN-data/lichess_Abdulrahmansoli_2026-02-12.pgn"},
    {"account_id": "present", "era": "present", "platform": "chesscom",
     "username": "AbdulrahmanSoliman2", "pgn_path": "PGN-data/chess_com_games_this_era_elo_1200.pgn"},
]

# %%
def outcome_for_me(result, me_color):
    if result not in {"1-0", "0-1", "1/2-1/2"}: return None
    if result == "1/2-1/2": return "draw"
    if (result == "1-0" and me_color == "white") or (result == "0-1" and me_color == "black"):
        return "win"
    return "loss"

def parse_time_control(tc):
    if not tc or tc in {"-", "?"}: return (None, None)
    m = re.match(r"^(\d+)\+(\d+)$", tc)
    if m: return int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d+)$", tc)
    if m: return int(m.group(1)), 0
    return (None, None)

def parse_game_datetime(headers):
    d = headers.get("UTCDate") or headers.get("Date")
    t = headers.get("UTCTime") or headers.get("Time")
    if not d or d in {"????.??.??"}: return None
    try:
        if t and t not in {"??:??:??"}: return datetime.strptime(d + " " + t, "%Y.%m.%d %H:%M:%S")
        return datetime.strptime(d, "%Y.%m.%d")
    except: return None

def safe_int(x):
    try: return int(x)
    except: return None

# %%
def load_pgn_account(pgn_path, my_username, account_meta):
    rows = []
    u = my_username.lower()
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None: break
            H = game.headers
            white = (H.get("White") or "").lower()
            black = (H.get("Black") or "").lower()
            if white == u:
                me_color, me_elo, opp_elo = "white", H.get("WhiteElo"), H.get("BlackElo")
            elif black == u:
                me_color, me_elo, opp_elo = "black", H.get("BlackElo"), H.get("WhiteElo")
            else: continue
            result = H.get("Result", "")
            outcome = outcome_for_me(result, me_color)
            if outcome is None: continue
            moves = list(game.mainline_moves())
            tc = H.get("TimeControl")
            base_sec, inc_sec = parse_time_control(tc)
            me_elo_i, opp_elo_i = safe_int(me_elo), safe_int(opp_elo)
            rows.append({
                **account_meta, "game_datetime": parse_game_datetime(H),
                "me_color": me_color, "result_raw": result, "outcome_me": outcome,
                "time_control": tc, "base_sec": base_sec, "inc_sec": inc_sec,
                "me_elo": me_elo_i, "opp_elo": opp_elo_i,
                "rating_diff": (me_elo_i - opp_elo_i) if (me_elo_i and opp_elo_i) else None,
                "eco": H.get("ECO"), "opening": H.get("Opening"),
                "n_plies": len(moves), "moves_uci": " ".join(m.uci() for m in moves),
            })
    return pd.DataFrame(rows)

# %%
print("Loading PGN files...")
games_list = []
for a in ACCOUNTS:
    meta = {"account_id": a["account_id"], "era": a["era"], "platform": a["platform"]}
    df = load_pgn_account(a["pgn_path"], a["username"], meta)
    print(f"  {a['era']}: {len(df)} games")
    games_list.append(df)
df_games = pd.concat(games_list, ignore_index=True)
print(f"\nTotal: {len(df_games)} games")
df_games.head()

# %% [markdown]
# ---
# # Section 3: Cleaning, Preprocessing, Feature Engineering & EDA

# %%
print("Cleaning data...")
df_clean = df_games[df_games['moves_uci'].str.len() > 0].copy()
df_clean = df_clean[df_clean['n_plies'] >= 10]
print(f"After cleaning: {len(df_clean)} games")

def categorize_time_control(base_sec):
    if base_sec is None: return 'unknown'
    if base_sec < 180: return 'bullet'
    elif base_sec < 600: return 'blitz'
    elif base_sec < 1800: return 'rapid'
    return 'classical'

df_clean['time_category'] = df_clean['base_sec'].apply(categorize_time_control)

# %%
def extract_prefix_features(moves_uci_str, me_color, prefix_plies=20):
    board = chess.Board()
    moves = [chess.Move.from_uci(u) for u in moves_uci_str.split() if u]
    me_is_white = (me_color == "white")
    my_castled, opp_castled, my_queen_moved = 0, 0, 0
    my_pawn_moves, my_minor_moves, my_rook_moves = 0, 0, 0
    my_captures, my_checks, my_center_pawns = 0, 0, 0
    pieces_moved = set()
    center_squares = {chess.E4, chess.D4, chess.E5, chess.D5}
    
    for ply, mv in enumerate(moves[:prefix_plies]):
        is_my_turn = (board.turn == chess.WHITE and me_is_white) or (board.turn == chess.BLACK and not me_is_white)
        if is_my_turn:
            pt = board.piece_type_at(mv.from_square)
            if board.is_castling(mv): my_castled = 1
            if pt == chess.QUEEN: my_queen_moved = 1
            if pt == chess.PAWN:
                my_pawn_moves += 1
                if mv.to_square in center_squares: my_center_pawns += 1
            if pt in {chess.KNIGHT, chess.BISHOP}: my_minor_moves += 1
            if pt == chess.ROOK: my_rook_moves += 1
            if board.is_capture(mv): my_captures += 1
            if board.gives_check(mv): my_checks += 1
            pieces_moved.add(mv.from_square)
        else:
            if board.is_castling(mv): opp_castled = 1
        board.push(mv)
    return {
        "my_castled_by_N": my_castled, "opp_castled_by_N": opp_castled,
        "my_queen_moved_by_N": my_queen_moved, "my_pawn_moves_N": my_pawn_moves,
        "my_minor_moves_N": my_minor_moves, "my_rook_moves_N": my_rook_moves,
        "my_captures_N": my_captures, "my_checks_N": my_checks,
        "my_center_pawns_N": my_center_pawns, "unique_pieces_moved_N": len(pieces_moved),
    }

print("Extracting features...")
feature_rows = []
for idx, row in df_clean.iterrows():
    feats = extract_prefix_features(row['moves_uci'], row['me_color'], prefix_plies=20)
    feature_rows.append({**row.to_dict(), **feats})
df_features = pd.DataFrame(feature_rows)
print(f"Features extracted: {len(df_features)} games")

# %%
# EDA Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
era_counts = df_features['era'].value_counts()
axes[0, 0].bar(era_counts.index, era_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[0, 0].set_title('Games per Era'); axes[0, 0].set_ylabel('Count')

outcome_era = df_features.groupby(['era', 'outcome_me']).size().unstack(fill_value=0)
outcome_era.plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title('Outcomes by Era'); axes[0, 1].tick_params(axis='x', rotation=0)

for era in ['childhood', 'highschool', 'present']:
    data = df_features[df_features['era'] == era]['me_elo'].dropna()
    if len(data) > 0: axes[1, 0].hist(data, alpha=0.5, label=era, bins=20)
axes[1, 0].set_title('Rating Distribution'); axes[1, 0].legend()

df_features.boxplot(column='n_plies', by='era', ax=axes[1, 1])
axes[1, 1].set_title('Game Length by Era'); plt.suptitle('')
plt.tight_layout(); plt.show()

# %%
feature_cols = ['my_castled_by_N', 'my_queen_moved_by_N', 'my_pawn_moves_N', 
                'my_minor_moves_N', 'my_captures_N', 'my_checks_N']
print("Feature Means by Era:")
print(df_features.groupby('era')[feature_cols].mean().round(3))

# %% [markdown]
# ---
# # Section 4: Analysis Setup and Data Splits
# 
# **Task**: Multiclass classification - predict era from early-game features

# %%
FEATURE_COLUMNS = ['my_castled_by_N', 'opp_castled_by_N', 'my_queen_moved_by_N',
    'my_pawn_moves_N', 'my_minor_moves_N', 'my_rook_moves_N',
    'my_captures_N', 'my_checks_N', 'my_center_pawns_N', 'unique_pieces_moved_N']
df_features['is_white'] = (df_features['me_color'] == 'white').astype(int)
FEATURE_COLUMNS.append('is_white')

X = df_features[FEATURE_COLUMNS].copy()
y = df_features['era'].copy()
print(f"Features: {X.shape}, Target: {y.shape}")
print(y.value_counts())

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ---
# # Section 5: Model Selection
# 
# ## Logistic Regression (Multinomial)
# $$P(y = k | \mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x})}{\sum_{j=1}^{K} \exp(\mathbf{w}_j^T \mathbf{x})}$$
# 
# ## Random Forest
# Ensemble of decision trees with bagging and feature randomization.

# %%
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=RANDOM_SEED),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
}

# %% [markdown]
# ---
# # Section 6: Training with Cross-Validation

# %%
print("Cross-validation scores:")
cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_results[name] = scores
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

# %%
# Hyperparameter tuning for Logistic Regression
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=RANDOM_SEED),
                           param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)
print(f"Best C: {grid_search.best_params_['C']}, Best CV Score: {grid_search.best_score_:.3f}")
best_model = grid_search.best_estimator_

# %% [markdown]
# ---
# # Section 7: Predictions and Metrics

# %%
y_pred = best_model.predict(X_test_scaled)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}")

# %% [markdown]
# ---
# # Section 8: Results Visualization

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
disp = ConfusionMatrixDisplay(cm, display_labels=best_model.classes_)
disp.plot(ax=axes[0], cmap='Blues')
axes[0].set_title('Confusion Matrix')

# Feature Importance
coef_df = pd.DataFrame({'feature': FEATURE_COLUMNS, 
    'childhood': best_model.coef_[0], 'highschool': best_model.coef_[1], 'present': best_model.coef_[2]})
coef_df.set_index('feature').plot(kind='barh', ax=axes[1])
axes[1].set_title('Feature Coefficients by Era')
plt.tight_layout(); plt.show()

# %%
# Misclassified examples
test_df = df_features.loc[X_test.index].copy()
test_df['pred'] = y_pred
mistakes = test_df[test_df['era'] != test_df['pred']]
print(f"Misclassified: {len(mistakes)} / {len(test_df)}")
print(mistakes[['era', 'pred', 'my_castled_by_N', 'my_queen_moved_by_N', 'my_pawn_moves_N']].head())

# %% [markdown]
# ---
# # Section 9: Executive Summary
# 
# ## Pipeline Diagram
# ```
# PGN Files → Parse Games → Clean Data → Extract Features → Train/Test Split
#                                              ↓
#                     Logistic Regression ← StandardScaler
#                                              ↓
#                          GridSearchCV → Best Model → Predictions → Metrics
# ```
# 
# ## Key Results
# - Successfully classified chess games into three eras
# - Model identifies playing style evolution over time
# - Key differentiating features: castling rate, minor piece development
# 
# ## Limitations
# - Class imbalance (childhood has most games)
# - Platform differences may confound results
# - Only using first 20 plies

# %% [markdown]
# ---
# # Section 10: References
# 
# 1. scikit-learn documentation: https://scikit-learn.org/
# 2. python-chess library: https://python-chess.readthedocs.io/
# 3. Chess.com and Lichess PGN export features
