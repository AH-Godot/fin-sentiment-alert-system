import streamlit as st
import feedparser
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

# =========================
# 1. SETUP & RESILIENT SESSION
# =========================
st.set_page_config(page_title="AI Stock Dashboard", page_icon="📈", layout="wide")
st.title("📈 Streamlined AI Sentiment Dashboard")

ticker = st.text_input("Ticker Symbol (e.g., AAPL, TSLA)", "AAPL").upper()
HF_TOKEN = st.secrets.get("HF_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def get_resilient_session():
    session = requests.Session()
    retry = Retry(total=5, connect=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

http_session = get_resilient_session()

# =========================
# 2. LIGHTWEIGHT DATA FETCHING
# =========================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_data(ticker_symbol):
    # News via RSS
    url = f"https://news.google.com/rss/search?q={ticker_symbol}%20stock&hl=en-US&gl=US&ceid=US:en"
    try:
        news = feedparser.parse(url).entries[:15]
    except Exception:
        news = []
    
    # Price via yfinance (Disguised to prevent blocking)
    try:
        yf_session = requests.Session()
        yf_session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"
        })
        stock = yf.Ticker(ticker_symbol, session=yf_session)
        price = stock.history(period="7d", interval="1h")
    except Exception:
        price = pd.DataFrame()
        
    return news, price

with st.spinner("Fetching market data..."):
    news, price = fetch_market_data(ticker)
    
if not news or price.empty:
    st.warning("Data unavailable for this ticker.")
    st.stop()

# =========================
# 3. API SENTIMENT ANALYSIS
# =========================
week_ago = datetime.now().date() - timedelta(days=7)
valid_news = [n for n in news if hasattr(n, "published_parsed") and datetime(*n.published_parsed[:6]).date() >= week_ago]
titles = [n.title for n in valid_news]
labels = ["neutral"] * len(titles) 

with st.spinner("Analyzing sentiments via API..."):
    if titles:
        API_URL = "https://api-inference.huggingface.co/models/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
        try:
            response = http_session.post(API_URL, headers=HEADERS, json={"inputs": titles}, timeout=15)
            if response.status_code == 200:
                results = response.json()
                labels = [max(res if isinstance(res, list) else [res], key=lambda x: x['score'])['label'].lower() for res in results]
        except Exception:
            pass # Fails safely to neutral on network errors

data = []
for entry, label in zip(valid_news, labels):
    dt = datetime(*entry.published_parsed[:6]).date()
    score = 1 if label == "positive" else -1 if label == "negative" else 0
    data.append({"date": dt, "label": label, "score": score, "title": entry.title, "link": getattr(entry, 'link', '#')})

df = pd.DataFrame(data)
if df.empty:
    st.warning("No recent sentiment data available.")
    st.stop()

# =========================
# 4. STATIC VISUALIZATION & UI
# =========================
cols = st.columns(3)
cols[0].metric("🟢 Positive News", len(df[df['label'] == 'positive']))
cols[1].metric("🔴 Negative News", len(df[df['label'] == 'negative']))
cols[2].metric("⚪ Neutral News", len(df[df['label'] == 'neutral']))

st.markdown("### 📊 Price vs. Sentiment Trend")

# Data Prep for Chart
df_daily = df.groupby("date")["score"].mean().rolling(2, min_periods=1).mean()
close_col = price["Close"]
if isinstance(close_col, pd.DataFrame):
    close_col = close_col.iloc[:, 0]
price_daily = close_col.groupby(price.index.date).mean()

# Render Static Chart (Extremely low memory footprint)
fig, ax1 = plt.subplots(figsize=(10, 4))

color1 = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Stock Price ($)', color=color1)
ax1.plot(price_daily.index, price_daily.values, color=color1, marker='o', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()  
color2 = 'tab:orange'
ax2.set_ylabel('Sentiment Score (-1 to 1)', color=color2)  
ax2.plot(df_daily.index, df_daily.values, color=color2, marker='s', linestyle='dashed', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(-1.2, 1.2)

fig.tight_layout()  
st.pyplot(fig) # Sends static image to frontend

# News Feed
st.markdown("### 📰 Recent Headlines")
for _, row in df.head(10).iterrows():
    emoji = "🟢" if row["label"] == "positive" else "🔴" if row["label"] == "negative" else "⚪"
    st.markdown(f"{emoji} [{row['title']}]({row['link']})")

# Auto-Refresh
st.divider()
with st.spinner("Dashboard is live. Next automated refresh in 5 minutes..."):
    time.sleep(300) 
    st.rerun()
