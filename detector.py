# =============================================================================
# FAKE NEWS DETECTOR — detector.py
# Uses PassiveAggressiveClassifier + TF-IDF Vectorization
# LEVEL 1 UPGRADES:
#   ✔ Confidence score shown with every prediction
#   ✔ Every prediction saved to predictions_log.csv with timestamp
# =============================================================================

import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ── ANSI colour helpers (makes terminal output easier to read) ──────────────
GREEN  = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"
YELLOW = "\033[93m"; BOLD = "\033[1m"; DIM  = "\033[2m"; RESET = "\033[0m"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATASET
# We read news.csv which must have a 'text' column and a 'label' column.
# Labels are expected to be the strings "REAL" or "FAKE".
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path="news.csv"):
    try:
        df = pd.read_csv(path)
        # Drop rows where either column is missing — dirty data kills accuracy
        df.dropna(subset=["text", "label"], inplace=True)
        return df["text"], df["label"]
    except FileNotFoundError:
        print(f"{RED}✗ '{path}' not found. Place news.csv in the same folder.{RESET}")
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — TRAIN / TEST SPLIT
# 80 % of rows train the model; 20 % are held back to measure real-world perf.
# random_state=42 ensures the same split every run (reproducibility).
# ─────────────────────────────────────────────────────────────────────────────
def split_data(X, y):
    return train_test_split(X, y, test_size=0.20, random_state=42)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TF-IDF VECTORISATION
# Converts raw text → a matrix of numbers.
# TF  (Term Frequency)        = how often a word appears in ONE article.
# IDF (Inverse Doc Frequency) = penalises words common across ALL articles.
# stop_words="english" drops filler words ("the", "is", …) automatically.
# max_df=0.7 ignores words that appear in >70 % of docs (too common to matter).
# ─────────────────────────────────────────────────────────────────────────────
def build_vectorizer():
    return TfidfVectorizer(stop_words="english", max_df=0.7)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — TRAIN THE MODEL
# PassiveAggressiveClassifier is an "online" learning algorithm.
# • Passive  → if the prediction is correct, the model doesn't change.
# • Aggressive → if wrong, it updates weights proportionally to the error.
# This makes it exceptionally fast and accurate on high-dimensional text data.
# max_iter=50 is enough for convergence on most news datasets.
# ─────────────────────────────────────────────────────────────────────────────
def train_model(X_train_tfidf, y_train):
    pac = PassiveAggressiveClassifier(max_iter=50, random_state=42)
    pac.fit(X_train_tfidf, y_train)
    return pac

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — EVALUATE PERFORMANCE
# Accuracy  = (correct predictions) / (total predictions)
# Confusion Matrix shows:
#       Predicted FAKE  |  Predicted REAL
#  FAKE   True Neg (TN) |  False Pos (FP)
#  REAL   False Neg (FN)|  True Pos (TP)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, X_test_tfidf, y_test):
    y_pred = model.predict(X_test_tfidf)
    acc    = accuracy_score(y_test, y_pred) * 100
    cm     = confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"])

    print(f"\n{BOLD}{'─'*45}{RESET}")
    print(f"  {BOLD}Model Accuracy:{RESET}  {GREEN}{acc:.2f}%{RESET}")
    print(f"{BOLD}{'─'*45}{RESET}")
    print(f"\n  {BOLD}Confusion Matrix{RESET}  (rows=Actual, cols=Predicted)")
    print(f"\n  {'':10}  {'FAKE':>8}  {'REAL':>8}")
    print(f"  {'Actual FAKE':10}  {cm[0][0]:>8}  {cm[0][1]:>8}")
    print(f"  {'Actual REAL':10}  {cm[1][0]:>8}  {cm[1][1]:>8}\n")
    return acc

# ─────────────────────────────────────────────────────────────────────────────
# UPGRADE 1 — CONFIDENCE SCORE
# decision_function() returns a raw score for each prediction.
# A high absolute value = the model is very sure.
# A low absolute value = the model is uncertain (close to the decision boundary).
# We use np.tanh() to squash any number smoothly into (0, 100%) range
# so it is easy for a human to understand.
# ─────────────────────────────────────────────────────────────────────────────
def get_confidence(model, vec):
    # decision_function gives a signed distance from the decision boundary
    raw_score  = model.decision_function(vec)[0]
    # np.tanh squashes any number into (-1, 1); abs() + *100 = percentage
    confidence = abs(float(np.tanh(raw_score))) * 100
    return round(confidence, 1)

# ─────────────────────────────────────────────────────────────────────────────
# UPGRADE 2 — PREDICTION LOG
# Every prediction is appended to predictions_log.csv automatically.
# Columns: timestamp | headline | prediction | confidence
# This creates a permanent record — great to show during Viva as proof
# the tool was actively tested with many real examples.
# ─────────────────────────────────────────────────────────────────────────────
LOG_FILE = "predictions_log.csv"

def log_prediction(headline, label, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row   = pd.DataFrame([{
        "timestamp":  timestamp,
        "headline":   headline,
        "prediction": label,
        "confidence": f"{confidence}%"
    }])
    try:
        # If log already exists → append without re-writing the header
        existing = pd.read_csv(LOG_FILE)
        updated  = pd.concat([existing, new_row], ignore_index=True)
    except FileNotFoundError:
        # First ever run → create a fresh log with header
        updated = new_row

    updated.to_csv(LOG_FILE, index=False)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — draws a small ASCII progress bar for confidence
# e.g.  [████████░░] 80.0%
# Makes output much more readable at a glance during a live demo
# ─────────────────────────────────────────────────────────────────────────────
def confidence_bar(confidence):
    filled = int(confidence / 10)
    empty  = 10 - filled
    bar    = "█" * filled + "░" * empty
    return f"[{bar}] {confidence}%"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — INTERACTIVE PREDICTION LOOP
# Vectorizer must be the *same* one fitted on training data (never re-fit).
# transform() (not fit_transform()) maps new text into the learned TF-IDF space.
# ─────────────────────────────────────────────────────────────────────────────
def prediction_loop(model, vectorizer):
    print(f"\n{CYAN}{BOLD}  ╔══════════════════════════════════════════╗")
    print(f"  ║        FAKE NEWS DETECTOR  v2.0          ║")
    print(f"  ║   Now with Confidence Score + Log File   ║")
    print(f"  ╚══════════════════════════════════════════╝{RESET}")
    print(f"  {DIM}Type a headline and press Enter.  ('quit' to exit){RESET}")
    print(f"  {DIM}All predictions are saved to → {LOG_FILE}{RESET}\n")

    while True:
        try:
            headline = input(f"  {BOLD}Enter News Headline:{RESET} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {DIM}Session ended.{RESET}\n")
            break

        if headline.lower() in ("quit", "exit", "q"):
            print(f"\n  {DIM}Goodbye! Check {LOG_FILE} to review all predictions.{RESET}\n")
            break
        if not headline:
            continue

        vec        = vectorizer.transform([headline])   # same TF-IDF space
        label      = model.predict(vec)[0]
        confidence = get_confidence(model, vec)

        colour = GREEN if label == "REAL" else RED
        icon   = "✔"  if label == "REAL" else "✘"

        print(f"\n  {colour}{BOLD}{icon}  Prediction : {label}{RESET}")
        print(f"     {BOLD}Confidence: {colour}{confidence_bar(confidence)}{RESET}")

        # Warn if the model is close to the boundary and not very sure
        if confidence < 55:
            print(f"  {YELLOW}⚠  Low confidence — result may be unreliable{RESET}")

        print()
        log_prediction(headline, label, confidence)   # save to CSV

# ─────────────────────────────────────────────────────────────────────────────
# MAIN — wires everything together
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{BOLD}  Training model… please wait.{RESET}")

    X, y                             = load_data("news.csv")
    X_train, X_test, y_train, y_test = split_data(X, y)

    vectorizer    = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    model = train_model(X_train_tfidf, y_train)
    evaluate(model, X_test_tfidf, y_test)
    prediction_loop(model, vectorizer)

if __name__ == "__main__":
    main()