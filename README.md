
## 📈 Real-Time Sentiment-Aware News Ticker for Financial Markets

This project implements a full-stack real-time data pipeline that monitors financial news and social media discussions, performs sentiment analysis, and predicts stock movement direction using machine learning. Designed for financial analysts, data scientists, and investors, the system provides live insights into market sentiment trends and potential stock movements.

### 🔧 Key Features

* **Live Data Collection**
  Ingests real-time data from **Reddit** (e.g., r/stocks, r/investing) and **NewsAPI**, combined with live stock prices.

* **NLP & Sentiment Analysis**
  Applies advanced natural language processing to clean and tokenize incoming text. Uses **VADER** and **BERT-based models** to compute sentiment scores.

* **Data Warehouse**
  Stores structured and enriched data in **PostgreSQL** using a **star schema**, optimizing it for analytics and modeling.

* **Sentiment Dashboard**
  Visualizes public sentiment and emotional trends over time via a **Power BI** dashboard connected directly to the database.

* **Predictive Modeling**
  An **XGBoost** model trained on engineered features from both sentiment data and stock price history predicts the direction of stock movements (up/down).

* **Streaming & Automation**
  Fully automated pipeline built in Python, with real-time updates and periodic model retraining for adaptive performance.

### 🧠 Technologies Used

* Python (Pandas, Scikit-learn, XGBoost, NLTK, Transformers)
* PostgreSQL with star schema design
* Power BI for visualization
* Reddit API & NewsAPI for data ingestion
* Streamlit (optional) or Flask for frontend deployment

### 📊 Potential Use Cases

* Real-time market sentiment monitoring
* Retail investor decision support
* Academic research in behavioral finance
* Algorithmic trading signal enrichment
![Screenshot 2025-06-01 122025](https://github.com/user-attachments/assets/c31f1e85-1903-4d30-8463-f32bbcb013ec)
![Screenshot 2025-05-26 231105](https://github.com/user-attachments/assets/71a5929f-8249-459d-9cf8-0a9b75462ca3)


