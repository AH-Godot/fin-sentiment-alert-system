# =========================
# 1. CRITICAL THREAD & RESOURCE LOCKS (Must be first)
# =========================
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Lock PyTorch memory before it can spawn background threads
import torch
torch.set_num_threads(1)
torch.set_grad_enabled(False) # Prevents storing unnecessary training tensors

import gc
import time
import io
import streamlit as st
import feedparser
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
from gtts import gTTS
from transformers import pipeline

# =========================
# 2. SETUP 
# =========================
st.set_page_config(page_title="AI Stock Dashboard", page_icon="📈", layout="wide")
st.title("📈 AI Stock Sentiment & Audio Alert Dashboard")

ticker = st.text_input("Ticker Symbol (e.g., AAPL, TSLA)", "AAPL").upper()

# =========================
# 3. LIGHTWEIGHT DATA FETCHING
# =========================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_data(ticker_symbol):
    # Fetch News (Limited to 10 to save RAM during local inference)
    url = f"https://news.google.com/rss/search?q={ticker_symbol}%20stock&hl=en-US&gl=US&ceid=US:en"
    try:
        news = feedparser.parse(url).entries[:10]
    except Exception:
        news = []
    
    # Fetch Price (Disguised as a real browser)
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
# 4. LOCAL PIPELINE SENTIMENT ANALYSIS
# =========================
data = []
week_ago = datetime.now().date() - timedelta(days=7)

with st.spinner("Loading Local AI Model & Analyzing Sentiments (This takes a moment)..."):
    # Load the smaller distilled model locally
    sentiment_model = pipeline(
        "sentiment-analysis", 
        model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    )
    
    for entry in news:
        if not hasattr(entry, "published_parsed"): continue
        dt = datetime(*entry.published_parsed[:6]).date()
        if dt < week_ago: continue
            
        # Get real prediction
        result = sentiment_model(entry.title)[0]
        label = result['label'].lower()
        score = 1 if label == "positive" else -1 if label == "negative" else 0
        
        data.append({
            "date": dt, 
            "label": label, 
            "score": score, 
            "title": entry.title,
            "link": getattr(entry, 'link', '#') 
        })
        
    # 🔴 CRITICAL MEMORY SAVER: Delete model instantly to prevent server crash
    del sentiment_model
    gc.collect()

df = pd.DataFrame(data)
if df.empty:
    st.warning("No recent sentiment data available.")
    st.stop()

df_daily = df.groupby("date")["score"].mean()
smoothed_daily = df_daily.rolling(2, min_periods=1).mean()

# =========================
# 5. TEXT-TO-SPEECH (TTS) REVERSAL ALERT
# =========================
st.markdown("### 🎙️ Audio Market Signal")

with st.spinner("Generating Audio Report..."):
    if len(df_daily) >= 2:
        today_score = df_daily.iloc[-1]
        yest_score = df_daily.iloc[-2]
        
        if today_score > 0 and yest_score <= 0:
            alert_text = f"Alert! A bullish reversal has been detected for {ticker}. The sentiment has shifted from negative to positive."
            st.success("🟢 Bullish Reversal Detected")
        elif today_score < 0 and yest_score >= 0:
            alert_text = f"Warning! A bearish reversal has been detected for {ticker}. The sentiment has shifted from positive to negative."
            st.error("🔴 Bearish Reversal Detected")
        else:
            trend = "positive" if today_score > 0 else "negative" if today_score < 0 else "neutral"
            alert_text = f"The current market sentiment for {ticker} continues to be {trend}. No major reversals detected today."
            st.info(f"⚪ Trend Continuation: {trend.title()}")
    else:
        alert_text = f"Analyzing {ticker}. Not enough daily history to detect a reversal yet."
        st.info("⚪ Gathering history...")

    try:
        tts = gTTS(text=alert_text, lang='en', slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        st.audio(audio_buffer, format='audio/mp3')
    except Exception:
        st.warning("Audio generation temporarily unavailable.")

# =========================
# 6. STATIC VISUALIZATION & UI
# =========================
cols = st.columns(3)
cols[0].metric("🟢 Positive News", len(df[df['label'] == 'positive']))
cols[1].metric("🔴 Negative News", len(df[df['label'] == 'negative']))
cols[2].metric("⚪ Neutral News", len(df[df['label'] == 'neutral']))

st.markdown("### 📊 Price vs. Sentiment Trend")

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
ax2.plot(smoothed_daily.index, smoothed_daily.values, color=color2, marker='s', linestyle='dashed', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(-1.2, 1.2)

fig.tight_layout()  
st.pyplot(fig) 

# News Feed
st.markdown("### 📰 Recent Headlines")
for _, row in df.head(10).iterrows():
    emoji = "🟢" if row["label"] == "positive" else "🔴" if row["label"] == "negative" else "⚪"
    st.markdown(f"{emoji} [{row['title']}]({row['link']})")

# =========================
# 7. AUTO-REFRESH LOGIC
# =========================
st.divider()
with st.spinner("Dashboard is live. Next automated refresh in 5 minutes..."):
    time.sleep(300) 
    st.rerun()
