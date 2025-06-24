# stock_ui.py
import streamlit as st
import requests
import os

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") or "zN6tsCIZILQxFgsQxYlGqIc0HTY44Rg2"

def get_prediction_and_explanation(ticker):
    try:
        res = requests.post("http://localhost:8000/predict", json={"ticker": ticker})
        if res.status_code != 200:
            return None, f"⚠️ Could not get prediction for {ticker}"
        movement = res.json().get("movement", "Unknown")
    except Exception as e:
        return None, f"❌ Error connecting to model API: {e}"

    mistral_url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "system", "content": "You're a friendly assistant that explains stock predictions."},
        {"role": "user", "content": f"The prediction for stock {ticker} is: {movement}. Please explain it simply and helpfully."}
    ]

    data = {
        "model": "mistral-small",
        "messages": messages,
        "temperature": 0.5
    }

    try:
        r = requests.post(mistral_url, headers=headers, json=data)
        r.raise_for_status()
        response = r.json()["choices"][0]["message"]["content"]
        return movement, response
    except Exception as e:
        return movement, f"❌ Failed to get a response from Mistral: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="📊 Stock Movement Prediction", layout="centered")
st.title("📈 Stock Movement Prediction Assistant")

ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, AMZN):").upper()

if st.button("Predict"):
    if not ticker:
        st.warning("Please enter a ticker.")
    else:
        with st.spinner("Analyzing..."):
            movement, explanation = get_prediction_and_explanation(ticker)

        if movement:
            st.markdown(f"### 📊 **Model Prediction for {ticker}**: {movement}")
        st.write("🤖 **Mistral says:**")
        st.markdown(explanation)
