import streamlit as st
import feedparser
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import io
import time
from gtts import gTTS
import torch
from transformers import pipeline

# =========================
# 1. SETUP 
# =========================
st.set_page_config(page_title="AI Stock Dashboard (DistilBERT)", page_icon="⚡", layout="wide")
st.title("⚡ Ultra-Fast Sentiment Dashboard")
st.caption("Powered by distilbert-base-uncased-finetuned-sst-2-english (No yfinance required)")

ticker = st.text_input("Ticker Symbol (e.g., AAPL, TSLA)", "AAPL").upper()

# =========================
# 2. RAW JSON DATA FETCHING (No yfinance)
# =========================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_data(ticker_symbol):
    # 1. Fetch News
    url_news = f"https://news.google.com/rss/search?q={ticker_symbol}%20stock&hl=en-US&gl=US&ceid=US:en"
    try:
        news = feedparser.parse(url_news).entries[:30]
    except Exception:
        news = []
    
    # 2. Fetch Raw Price Data via hidden Yahoo JSON endpoint
    price_series = pd.Series(dtype=float)
    try:
        url_price = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker_symbol}?range=10d&interval=1d"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"}
        
        response = requests.get(url_price, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()['chart']['result'][0]
            dates = [datetime.fromtimestamp(ts).date() for ts in data['timestamp']]
            closes = data['indicators']['quote'][0]['close']
            price_series = pd.Series(closes, index=dates).dropna()
    except Exception as e:
        pass 
        
    return news, price_series

with st.spinner("Fetching market data..."):
    news, price_daily = fetch_market_data(ticker)
    
if not news:
    st.error(f"No news data available for {ticker}. Check ticker symbol.")
    st.stop()
    
if price_daily.empty:
    st.warning("⚠️ Could not fetch price data. Displaying AI Sentiment without stock price overlay.")

# =========================
# 3. GLOBAL MODEL LOADING (CPU Warning Removed)
# =========================
@st.cache_resource(show_spinner=False)
def load_model():
    if torch.cuda.is_available():
        device_id = 0
    elif torch.backends.mps.is_available():
        device_id = "mps" 
    else:
        device_id = -1

    return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=device_id), device_id

sentiment_model, active_device = load_model()

# Only show success message if a GPU is actually found, otherwise remain completely silent
if active_device != -1:
    st.success("⚡ Model successfully loaded onto the GPU!")

# =========================
# 4. SENTIMENT ANALYSIS
# =========================
data = []
week_ago = datetime.now().date() - timedelta(days=7)

with st.spinner("Analyzing sentiments at lightning speed..."):
    for entry in news:
        if not hasattr(entry, "published_parsed"): continue
        dt = datetime(*entry.published_parsed[:6]).date()
        if dt < week_ago: continue
            
        result = sentiment_model(entry.title)[0]
        label = result['label'].lower()
        score = 1 if label == "positive" else -1
        
        data.append({
            "date": dt, 
            "label": label, 
            "score": score, 
            "title": entry.title,
            "link": getattr(entry, 'link', '#') 
        })

df = pd.DataFrame(data)
if df.empty:
    st.warning("No recent sentiment data available.")
    st.stop()

df_daily = df.groupby("date")["score"].mean()
smoothed_daily = df_daily.rolling(2, min_periods=1).mean()

# =========================
# 5. TEXT-TO-SPEECH (TTS) ALERT w/ AUTOPLAY
# =========================
st.markdown("### 🎙️ Audio Market Signal")

with st.spinner("Generating Audio Report..."):
    if len(df_daily) >= 2:
        today_score = df_daily.iloc[-1]
        yest_score = df_daily.iloc[-2]
        
        if today_score > 0 and yest_score <= 0:
            alert_text = f"Alert! A bullish reversal has been detected for {ticker}."
            st.success("🟢 Bullish Reversal Detected")
        elif today_score < 0 and yest_score >= 0:
            alert_text = f"Warning! A bearish reversal has been detected for {ticker}."
            st.error("🔴 Bearish Reversal Detected")
        else:
            trend = "positive" if today_score > 0 else "negative" 
            alert_text = f"The current market sentiment for {ticker} continues to be {trend}."
            st.info(f"⚪ Trend Continuation: {trend.title()}")
    else:
        alert_text = f"Analyzing {ticker}."
        st.info("⚪ Gathering history...")

    try:
        tts = gTTS(text=alert_text, lang='en', slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # 🔴 FIX: Added autoplay=True so it reads the market signal out loud instantly
        st.audio(audio_buffer, format='audio/mp3', autoplay=True)
    except Exception:
        st.warning("Audio generation temporarily unavailable.")

# =========================
# 6. DYNAMIC VISUALIZATION & UI
# =========================
cols = st.columns(2) 
cols[0].metric("🟢 Positive News", len(df[df['label'] == 'positive']))
cols[1].metric("🔴 Negative News", len(df[df['label'] == 'negative']))

st.markdown("### 📊 Price vs. Sentiment Trend")

fig, ax1 = plt.subplots(figsize=(10, 4))

if not price_daily.empty:
    color1 = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Price ($)', color=color1)
    ax1.plot(price_daily.index, price_daily.values, color=color1, marker='o', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx() 
else:
    ax1.set_xlabel('Date')
    ax1.grid(True, alpha=0.3)
    ax2 = ax1 

color2 = 'tab:orange'
ax2.set_ylabel('Sentiment Score (-1 to 1)', color=color2)  
ax2.plot(smoothed_daily.index, smoothed_daily.values, color=color2, marker='s', linestyle='dashed', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(-1.2, 1.2)

fig.tight_layout()  
st.pyplot(fig) 

# News Feed
st.markdown("### 📰 Recent Headlines")
for _, row in df.head(10).iterrows():
    emoji = "🟢" if row["label"] == "positive" else "🔴"
    st.markdown(f"{emoji} [{row['title']}]({row['link']})")

# =========================
# 7. AUTO-REFRESH LOGIC
# =========================
st.divider()
with st.spinner("Dashboard is live. Next automated refresh in 5 minutes..."):
    time.sleep(300) 
    st.rerun()
