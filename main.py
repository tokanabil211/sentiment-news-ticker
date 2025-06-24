# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import xgboost as xgb
import os

app = FastAPI()

# ============ Load model ============
model_path = "stock_movement_model_xgboost.pkl"
if not os.path.exists(model_path):
    # Dummy training for safety if model is missing
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, (iris.target == 0).astype(int), test_size=0.2)

    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
else:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

# ============ Input schema ============
class TickerInput(BaseModel):
    ticker: str

# ============ Dummy feature extractor ============
def extract_features_from_ticker(ticker: str):
    # Replace with real feature generation
    return [0.1, 0.5, 0.3, 0.2]  # Must match training feature size

# ============ API endpoint ============
@app.post("/predict")
def predict(ticker_input: TickerInput):
    features = extract_features_from_ticker(ticker_input.ticker)
    prediction = model.predict([features])[0]
    return {
        "ticker": ticker_input.ticker,
        "movement": "📈 Up" if prediction == 1 else "📉 Down"
    }
