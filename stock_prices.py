import yfinance as yf
import psycopg2
from datetime import datetime, timedelta
import time

# PostgreSQL connection
conn = psycopg2.connect(
    dbname='financial_streaming',
    user='postgres',
    password='123456@Toka',
    host='localhost',
    port='5432'
)
conn.autocommit = True
cur = conn.cursor()

try:
    while True:
        # Get tickers from dim_ticker
        cur.execute("SELECT ticker FROM dim_ticker")
        tickers = [row[0] for row in cur.fetchall()]

        now = datetime.now()
        today = now.date()

        for ticker in tickers:
            try:
                # Download data ±3 days around today
                df = yf.download(ticker, start=today - timedelta(days=3), end=today + timedelta(days=3), progress=False)

                if df.empty:
                    print(f"⚠️ No data at all for {ticker} near {today}")
                    continue

                # Get closest available trading date
                df.index = df.index.date
                closest_date = min(df.index, key=lambda d: abs(d - today))
                row = df.loc[closest_date]

                open_price = row['Open'].item()
                close_price = row['Close'].item()
                high_price = row['High'].item()
                low_price = row['Low'].item()
                volume = int(row['Volume'].item())

                # Insert if not already present (based on full row uniqueness)
                cur.execute("""
                    INSERT INTO stock_prices (
                        ticker, price_date, open, close, high, low, volume, fetched_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT DO NOTHING
                """, (
                    ticker, closest_date, open_price, close_price, high_price, low_price, volume
                ))

                print(f"✅ Inserted price for {ticker} on {closest_date}")

            except Exception as e:
                print(f"❌ Error for {ticker}: {e}")

        print("🔁 Waiting 24 hours for next fetch...")
        time.sleep(86400)

except KeyboardInterrupt:
    print("🛑 Stopped fetching stock prices.")
finally:
    cur.close()
    conn.close()
    print("📉 PostgreSQL connection closed.")
