import praw
import json
from datetime import datetime
import time
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import psycopg2

# 📝 Reddit API Setup
reddit = praw.Reddit(client_id='jocweWxmCGIUeIV7fTw8-Q',
                     client_secret='TlN-wZ1A2HnAYhxZV448WrFwvUoDvQ',
                     user_agent='FinancialSentimentApp by /u/toka_nabil211')

# 📝 Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# PostgreSQL Connection Setup
conn = psycopg2.connect(
    dbname='financial_streaming',
    user='postgres',
    password='123456@Toka',
    host='localhost',
    port='5432'
)
conn.autocommit = True
cur = conn.cursor()

# Ticker Map
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


# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s$]", "", text)
    return text

def extract_tickers(text, mapping, original_text=None):
    tickers_found = set()
    text = text.lower()
    words = set(re.findall(r'\b\w+\b', text))
    if original_text is None:
        original_text = text
    for company, ticker in mapping.items():
        company_clean = re.sub(r'[^a-z0-9\s]', '', company.lower())
        if len(company_clean) <= 2:
            pattern = r'\b' + re.escape(ticker) + r'\b'
            if re.search(pattern, original_text):
                tickers_found.add(ticker)
        elif ' ' in company_clean:
            if company_clean in text:
                tickers_found.add(ticker)
        else:
            if company_clean in words:
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

def save_to_star_schema(posts, cursor):
    cursor.execute("SELECT external_id FROM fact_sentiment WHERE source = 'Reddit'")
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
            post['prediction'], 'Reddit', datetime.now(), author_id, date_id
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

print("\U0001F680 Starting Reddit Stream with Sentiment... Press CTRL+C to stop.")

try:
    while True:
        start_total = datetime.now()

        # ⏱️ Step 1: Reddit API Fetch
        start_fetch = datetime.now()
        posts = reddit.subreddit('wallstreetbets+stocks+investing+StockMarket').new(limit=150)
        end_fetch = datetime.now()

        new_posts_data = []

        # ⏱️ Step 2: Preprocessing + Sentiment
        start_preprocess = datetime.now()
        raw_samples = []
        processed_samples = []

        for post in posts:
            combined_text = f"{post.title} {post.selftext}"
            clean_combined = clean_text(combined_text)
            sentiment = analyzer.polarity_scores(clean_combined)['compound']
            tickers = extract_tickers(clean_combined, company_ticker_map, combined_text)
            
            if not tickers or len(clean_combined.split()) < 5 or post.score < 1:
                continue

            prediction = "Neutral"
            if sentiment > 0.2:
                prediction = "Likely Up 📈"
            elif sentiment < -0.2:
                prediction = "Likely Down 📉"

            post_info = {
                'id': post.id,
                'title': post.title,
                'author': str(post.author),
                'created_utc': datetime.fromtimestamp(post.created_utc).isoformat(),
                'score': post.score,
                'num_comments': post.num_comments,
                'url': post.url,
                'selftext': post.selftext,
                'clean_title': clean_combined,
                'sentiment_score': sentiment,
                'tickers': tickers,
                'prediction': prediction
            }
            new_posts_data.append(post_info)
        end_preprocess = datetime.now()

        # ⏱️ Step 3: Insert to PostgreSQL
        start_insert = datetime.now()
        if new_posts_data:
            inserted = save_to_star_schema(new_posts_data, cur)
            end_insert = datetime.now()

            print(f"✅ Inserted {inserted} new posts at {datetime.now()}")
            print(f"⏱️ Reddit Fetch: {(end_fetch - start_fetch).total_seconds()}s | "
                  f"Preprocessing: {(end_preprocess - start_preprocess).total_seconds()}s | "
                  f"DB Insert: {(end_insert - start_insert).total_seconds()}s | "
                  f"TOTAL: {(end_insert - start_total).total_seconds()}s")
        else:
            print(f"🔁 No matching posts at {datetime.now()}")
        print("\n📄 Sample Preprocessing Output (for paper):")

        


        time.sleep(60)
except KeyboardInterrupt:
    print("🛑 Streaming stopped.")
finally:
    cur.close()
    conn.close()
    print("📉 PostgreSQL connection closed.")
