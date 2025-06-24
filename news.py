import requests
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import time
import os
import re
import psycopg2

# NewsAPI Setup
NEWS_API_KEY = '521ba49c6d8c4c519306ec43a519fb34'
NEWS_API_URL = 'https://newsapi.org/v2/everything'

analyzer = SentimentIntensityAnalyzer()

conn = psycopg2.connect(
    dbname='financial_streaming',
    user='postgres',
    password='123456@Toka',
    host='localhost',
    port='5432'
)
conn.autocommit = True
cur = conn.cursor()

# Mapping (same as previous script, trimmed here for brevity)
company_ticker_map = {
    # شركات تكنولوجيا كبرى
    'tesla': 'TSLA', 'tsla': 'TSLA',
    'apple': 'AAPL', 'aapl': 'AAPL',
    'amazon': 'AMZN', 'amzn': 'AMZN',
    'microsoft': 'MSFT', 'msft': 'MSFT',
    'google': 'GOOG', 'goog': 'GOOG', 'googl': 'GOOG', 'alphabet': 'GOOG',
    'meta': 'META', 'facebook': 'META', 'meta platforms': 'META',
    'netflix': 'NFLX', 'nflx': 'NFLX',
    'nvidia': 'NVDA', 'nvda': 'NVDA',
    'amd': 'AMD',
    'intel': 'INTC', 'intc': 'INTC',
    'paypal': 'PYPL', 'pypl': 'PYPL',
    'zoom': 'ZM', 'zm': 'ZM',
    'spotify': 'SPOT',
    'shopify': 'SHOP',
    'snowflake': 'SNOW', 'snow': 'SNOW',
    'crowdstrike': 'CRWD', 'crwd': 'CRWD',
    'docusign': 'DOCU', 'docu': 'DOCU',
    'cloudflare': 'NET', 'net': 'NET',
    'palantir': 'PLTR', 'pltr': 'PLTR',
    'unity': 'U', 'u': 'U',
    'roku': 'ROKU',
    'pinterest': 'PINS', 'pins': 'PINS',
    'etsy': 'ETSY',
    'airbnb': 'ABNB',
    'twilio': 'TWLO',

    # شركات مالية وبنوك
    'bank of america': 'BAC', 'bac': 'BAC',
    'jpmorgan': 'JPM', 'jpm': 'JPM',
    'goldman sachs': 'GS', 'gs': 'GS',
    'morgan stanley': 'MS', 'ms': 'MS',
    'blackrock': 'BLK', 'blk': 'BLK',
    'sofi': 'SOFI',
    'square': 'SQ', 'block': 'SQ',
    'coinbase': 'COIN', 'coin': 'COIN',
    'robinhood': 'HOOD', 'hood': 'HOOD',

    # شركات استهلاكية
    'ford': 'F',
    'gm': 'GM', 'general motors': 'GM',
    'walmart': 'WMT', 'wmt': 'WMT',
    'target': 'TGT', 'tgt': 'TGT',
    'costco': 'COST',
    'starbucks': 'SBUX',
    'mcdonalds': 'MCD', 'mcdonald': 'MCD',
    'wayfair': 'W', 'wayfair inc': 'W',
    'etsy': 'ETSY',

    # قطاع الطاقة
    'exxon': 'XOM', 'xom': 'XOM',
    'chevron': 'CVX',
    'bp': 'BP',

    # أدوية ورعاية صحية
    'pfizer': 'PFE', 'pfe': 'PFE',
    'moderna': 'MRNA',
    'johnson and johnson': 'JNJ',
    'unitedhealth': 'UNH', 'unh': 'UNH',

    # ETFs وصناديق
    'sp500': 'SPY', 'spy': 'SPY',
    'qqq': 'QQQ', 'nasdaq': 'QQQ', 'ndx': 'QQQ',
    'voo': 'VOO',
    'ugl': 'UGL',
    'gld': 'GLD',

    # أسهم Reddit الشهيرة
    'gamestop': 'GME',
    'amc': 'AMC',
    'lucid': 'LCID',
    'rivian': 'RIVN',
    'riot': 'RIOT',
    'snap': 'SNAP', 'snapchat': 'SNAP',
    'doge': 'DOGE',  # عملة رقمية

    # إضافات مشهورة
    'trump': 'TRUMP',
    'fix': 'FIX',
    'ntb': 'NTB',
    'aal': 'AAL',
    'now': 'NOW',
    'mbly': 'MBLY',
    'ibm': 'IBM',
    'clf': 'CLF',
    'bkng': 'BKNG',
    'adbe': 'ADBE',
    'crm': 'CRM',
    'afrm': 'AFRM',
    'dash': 'DASH',
    'team': 'TEAM',
    'z': 'Z',
    'fubo': 'FUBO',
    'blk': 'BLK',
    'etr': 'ETR',
    'qcom': 'QCOM',
    'pton': 'PTON',
    'bbby': 'BBBY',
    'm': 'M',
    'wen': 'WEN',
    'znga': 'ZNGA',
        'robinhood': 'HOOD',
    'fidelity': 'FNF',
    'bofa': 'BAC',
    'bank of america': 'BAC'
}


# Helper functions

def extract_tickers(text, mapping):
    text = text.lower()
    tickers_found = set()
    for name, ticker in mapping.items():
        pattern = r'\b' + re.escape(name.lower()) + r'\b'
        if re.search(pattern, text):
            tickers_found.add(ticker)
    return list(tickers_found)

def get_or_create_id(cursor, table, column, value):
    id_column = {
        'dim_author': 'author_id',
        'dim_ticker': 'ticker_id'
    }[table]

    cursor.execute(f"SELECT {id_column} FROM {table} WHERE {column} = %s", (value,))
    result = cursor.fetchone()
    if result:
        return result[0]
    cursor.execute(f"INSERT INTO {table} ({column}) VALUES (%s) RETURNING {id_column}", (value,))
    return cursor.fetchone()[0]


def get_or_create_date_id(cursor, timestamp_str):
    dt = datetime.fromisoformat(timestamp_str)
    dt = dt.replace(second=0, microsecond=0)  # normalize to the minute

    # Check if the timestamp exists
    cursor.execute("SELECT date_id FROM dim_time WHERE full_date = %s", (dt,))
    result = cursor.fetchone()
    if result:
        return result[0]

    # Insert if not found
    cursor.execute("""
        INSERT INTO dim_time (full_date, year, month, day, hour, minute)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING date_id
    """, (dt, dt.year, dt.month, dt.day, dt.hour, dt.minute))
    return cursor.fetchone()[0]


def save_news_to_star_schema(posts, cursor):
    cursor.execute("SELECT external_id FROM fact_sentiment WHERE source = 'NewsAPI'")
    existing_ids = {row[0] for row in cursor.fetchall()}
    new_posts = [post for post in posts if post['id'] not in existing_ids]

    for post in new_posts:
        author_id = get_or_create_id(cursor, "dim_author", "author_name", post['author'])
        date_id = get_or_create_date_id(cursor, post['created_utc'])

        cursor.execute("""
            INSERT INTO fact_sentiment (
                external_id, title, content, sentiment_score, prediction,
                source, fetched_at, author_id, date_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING sentiment_id
        """, (
            post['id'], post['title'], post['clean_title'], post['sentiment_score'],
            post['prediction'], 'NewsAPI', datetime.now(), author_id, date_id
        ))
        sentiment_id = cursor.fetchone()[0]

        for ticker in post['tickers']:
            ticker_id = get_or_create_id(cursor, "dim_ticker", "ticker", ticker)
            cursor.execute("""
                INSERT INTO fact_sentiment_ticker (sentiment_id, ticker_id)
                VALUES (%s, %s) ON CONFLICT DO NOTHING
            """, (sentiment_id, ticker_id))

    cursor.connection.commit()
    return len(new_posts)

def fetch_financial_news(query='stock market', page_size=100):
    params = {
        'q': query,
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': page_size,
        'apiKey': NEWS_API_KEY
    }
    response = requests.get(NEWS_API_URL, params=params)
    return response.json().get('articles', [])

def analyze_and_save_news():
    start_total = datetime.now()
    print(f"🔁 Checking for new news at {start_total}...")

    # ⏱️ 1. Fetching from NewsAPI
    start_fetch = datetime.now()
    articles = fetch_financial_news('Tesla OR Apple OR stock market')
    end_fetch = datetime.now()

    # ⏱️ 2. Preprocessing + sentiment
    start_preprocess = datetime.now()
    posts = []
    for article in articles:
        title = article['title']
        description = article.get('description') or ''
        content = f"{title}. {description}"
        article_url = article['url']
        published_at = article['publishedAt']
        source_name = article['source']['name']

        tickers = extract_tickers(content, company_ticker_map)
        if not tickers:
            continue

        sentiment_score = analyzer.polarity_scores(content)['compound']
        prediction = 'Neutral'
        if sentiment_score > 0.2:
            prediction = 'Likely Up 📈'
        elif sentiment_score < -0.2:
            prediction = 'Likely Down 📉'

        post = {
            'id': article_url,
            'title': title,
            'author': source_name,
            'created_utc': datetime.fromisoformat(published_at.replace('Z', '+00:00')).isoformat(),
            'clean_title': content,
            'sentiment_score': sentiment_score,
            'tickers': tickers,
            'prediction': prediction
        }
        posts.append(post)
    end_preprocess = datetime.now()

    # ⏱️ 3. DB Insert
    start_insert = datetime.now()
    inserted_count = save_news_to_star_schema(posts, cur)
    end_insert = datetime.now()

    end_total = datetime.now()

    print(f"✅ Inserted {inserted_count} new articles.")
    print(f"🕒 Fetch: {(end_fetch - start_fetch).total_seconds()}s | "
          f"Preprocessing: {(end_preprocess - start_preprocess).total_seconds()}s | "
          f"Insert: {(end_insert - start_insert).total_seconds()}s | "
          f"TOTAL: {(end_total - start_total).total_seconds()}s")


try:
    while True:
        analyze_and_save_news()
        time.sleep(900)
except KeyboardInterrupt:
    print("🛑 Stopped streaming.")
finally:
    cur.close()
    conn.close()
    print("📉 PostgreSQL connection closed.")
