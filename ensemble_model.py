import pandas as pd
import psycopg2
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
import joblib
import time
import os
from datetime import datetime
import numpy as np
from collections import Counter
from sklearn.ensemble import VotingClassifier

# DB connection config
db_config = {
    "host": "localhost",
    "port": "5432",
    "database": "financial_streaming",
    "user": "postgres",
    "password": "123456@Toka"
}

MIN_ROWS_TO_TRAIN = 100

def load_data():
    query = """
        SELECT 
            sentiment_score,
            is_strong_sentiment,
            sentiment_time_category,
            movement,
            full_date
        FROM training_data_by_sentiment_clean
    """
    conn = psycopg2.connect(**db_config)
    df = pd.read_sql(query, conn)
    conn.close()
    df = df[df["movement"].isin(["up", "down"])]
    return df

def train_ensemble(df):
    df = pd.get_dummies(df, columns=["sentiment_time_category"], drop_first=True)
    X = df.drop(columns=["movement", "full_date"])
    y = (df["movement"] == "up").astype(int)

    if len(np.unique(y)) < 2:
        print("❌ Only one class in data. Skipping training.")
        return

    # Undersampling
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    print("📊 After undersampling:", dict(Counter(y_resampled)))

    # Split into train, val, test
    X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Define models
    xgb_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss", verbosity=0)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    log_model = LogisticRegression(max_iter=1000)

    # Voting Ensemble
    ensemble = VotingClassifier(estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('lr', log_model)
    ], voting='hard')

    start = datetime.now()
    ensemble.fit(X_train, y_train)
    train_time = (datetime.now() - start).total_seconds()
    print(f"⏱️ Training time: {train_time:.2f} seconds")

    # Evaluate
    for split_name, X_split, y_split in [
        ("Train", X_train, y_train),
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test),
    ]:
        preds = ensemble.predict(X_split)
        acc = accuracy_score(y_split, preds)
        print(f"✅ {split_name} Accuracy: {acc:.4f}")
        if split_name == "Test":
            print("📊 Final Test Report:\n", classification_report(y_split, preds, target_names=["down", "up"]))

    joblib.dump(ensemble, "ensemble_stock_model.pkl")
    print(f"✅ Model saved to ensemble_stock_model.pkl")

if __name__ == "__main__":
    print("🔁 Starting ensemble model training loop...")
    while True:
        df = load_data()
        if df.empty:
            print(f"[{datetime.now()}] 📭 No data available.")
        elif len(df) < MIN_ROWS_TO_TRAIN:
            print(f"[{datetime.now()}] ⚠️ Only {len(df)} rows — waiting for more data.")
        else:
            print(f"[{datetime.now()}] 🧠 Training on {len(df)} rows...")
            train_ensemble(df)
        time.sleep(300)  # 5 minutes
