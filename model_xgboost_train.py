import pandas as pd
import psycopg2
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample, compute_class_weight
import numpy as np
import joblib
import time
import os
from datetime import datetime

# === PostgreSQL Configuration ===
db_config = {
    "host": "localhost",
    "port": "5432",
    "database": "financial_streaming",
    "user": "postgres",
    "password": "123456@Toka"
}

MIN_ROWS_TO_TRAIN = 100
LAST_DATE_FILE = "last_training_date_xgboost.txt"

def get_last_training_time():
    if os.path.exists(LAST_DATE_FILE):
        with open(LAST_DATE_FILE, "r") as f:
            return f.read().strip()
    return "2000-01-01 00:00:00"

def save_last_training_time(new_time):
    with open(LAST_DATE_FILE, "w") as f:
        f.write(str(new_time))

def load_data():
    query = """
        SELECT sentiment_score, is_strong_sentiment, sentiment_time_category, movement, full_date
        FROM training_data_by_sentiment_clean
    """
    conn = psycopg2.connect(**db_config)
    df = pd.read_sql(query, conn)
    conn.close()
    df = df[df["movement"].isin(["up", "down"])]
    return df

def train_model(df):
    df = pd.get_dummies(df, columns=["sentiment_time_category"], drop_first=True)

    # Encode label
    df["label"] = (df["movement"] == "up").astype(int)
    X = df.drop(["movement", "full_date", "label"], axis=1)
    y = df["label"]

    # === Undersampling ===
    df["label"] = y
    down_df = df[df["label"] == 0]
    up_df = df[df["label"] == 1]
    min_size = min(len(down_df), len(up_df))

    df_balanced = pd.concat([
        down_df.sample(n=min_size, random_state=42),
        up_df.sample(n=min_size, random_state=42)
    ])
    X = df_balanced.drop(["movement", "full_date", "label"], axis=1)
    y = df_balanced["label"]

    print(f"📊 After undersampling: {y.value_counts().to_dict()}")

    # === Split into train, validation, test ===
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # === Train XGBoost ===
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=0
    )

    start = datetime.now()
    model.fit(X_train, y_train)
    end = datetime.now()
    print(f"⏱️ Training time: {(end - start).total_seconds():.2f} seconds")

    # === Evaluate ===
    preds_train = model.predict(X_train)
    preds_val = model.predict(X_val)
    preds_test = model.predict(X_test)

    print(f"✅ Train Accuracy: {accuracy_score(y_train, preds_train):.4f}")
    print(f"✅ Validation Accuracy: {accuracy_score(y_val, preds_val):.4f}")
    print(f"✅ Test Accuracy: {accuracy_score(y_test, preds_test):.4f}")
    print("📊 Final Test Report:\n", classification_report(y_test, preds_test, target_names=["down", "up"]))

    joblib.dump(model, "stock_movement_model_xgboost.pkl")
    print(f"✅ Model saved to stock_movement_model_xgboost.pkl")

    with open("training_log_xgboost.txt", "a") as f:
        f.write(
            f"{datetime.now()} | Train: {accuracy_score(y_train, preds_train):.4f} | "
            f"Val: {accuracy_score(y_val, preds_val):.4f} | Test: {accuracy_score(y_test, preds_test):.4f} | "
            f"Rows: {len(df_balanced)}\n"
        )

if __name__ == "__main__":
    print("🔁 Starting streaming training loop...")

    while True:
        df = load_data()
        if df.empty:
            print("📭 No new data.")
        elif len(df) < MIN_ROWS_TO_TRAIN:
            print(f"⚠️ Only {len(df)} rows. Skipping.")
        else:
            print(f"[{datetime.now()}] 🧠 Training on {len(df)} new rows...")
            train_model(df)
        print("⏳ Waiting 5 minutes before next training round...\n")
        time.sleep(300)
