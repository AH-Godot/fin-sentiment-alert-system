import streamlit as st
import feedparser
import pandas as pd
import yfinance as yf
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
st.caption("Powered by distilbert/distilbert-base-uncased-finetuned-sst-2-english")

ticker = st.text_input("Ticker Symbol (e.g., AAPL, TSLA)", "AAPL").upper()

# =========================
# 2. UNRESTRICTED DATA FETCHING
# =========================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_data(ticker_symbol):
    # Fetching up to 30 headlines 
    url = f"https://news.google.com/rss/search?q={ticker_symbol}%20stock&hl=en-US&gl=US&ceid=US:en"
    try:
        news = feedparser.parse(url).entries[:30]
    except Exception:
        news = []
    
    try:
        price = yf.download(ticker_symbol, period="10d", interval="1d", threads=False)
    except Exception:
        price = pd.DataFrame()
        
    return news, price

with st.spinner("Fetching market data..."):
    news, price = fetch_market_data(ticker)
    
if not news or price.empty:
    st.warning(f"Data unavailable for {ticker}. Check ticker symbol or Yahoo Finance connection.")
    st.stop()

# =========================
# 3. GLOBAL GPU MODEL LOADING
# =========================
@st.cache_resource(show_spinner=False)
def load_gpu_model():
    # Check for GPU availability
    if torch.cuda.is_available():
        device_id = 0
        print("GPU Detected: Loading model to VRAM...")
    elif torch.backends.mps.is_available():
        # Support for Apple Silicon GPUs (M1/M2/M3 MacBooks)
        device_id = "mps" 
        print("Apple Metal GPU Detected...")
    else:
        device_id = -1
        print("No GPU detected. Falling back to CPU...")

    # Load your requested DistilBERT model
    return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=device_id), device_id

sentiment_model, active_device = load_gpu_model()

if active_device == -1:
    st.error("⚠️ GPU not detected by PyTorch. The model is running on the CPU.")
else:
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
        # SST-2 outputs "POSITIVE" or "NEGATIVE"
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
# 5. TEXT-TO-SPEECH (TTS) ALERT
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
            trend = "positive" if today_score > 0 else "negative" 
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
cols = st.columns(2) # Only 2 columns since SST-2 doesn't output Neutral
cols[0].metric("🟢 Positive News", len(df[df['label'] == 'positive']))
cols[1].metric("🔴 Negative News", len(df[df['label'] == 'negative']))

st.markdown("### 📊 Price vs. Sentiment Trend")

# Safely extract Close price
close_col = price["Close"]
if isinstance(close_col, pd.DataFrame):
    close_col = close_col.iloc[:, 0]
price_daily = close_col.groupby(close_col.index.date).mean()

# Render Static Chart 
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
    emoji = "🟢" if row["label"] == "positive" else "🔴"
    st.markdown(f"{emoji} [{row['title']}]({row['link']})")

# =========================
# 7. AUTO-REFRESH LOGIC
# =========================
st.divider()
with st.spinner("Dashboard is live. Next automated refresh in 5 minutes..."):
    time.sleep(300) 
    st.rerun()
