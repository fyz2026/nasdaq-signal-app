import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# إعداد الصفحة
st.set_page_config(page_title="البحث الذكي عن فرص الدخول في أسهم ناسداك", layout="wide")

st.title("📈 ابحث الذكي عن فرص الدخول في أسهم ناسداك")
st.markdown("تحليل فني تلقائي على الفاصل اليومي لاكتشاف الأسهم التي تمر بنموذج اختراق بعد تجميع.")

# دالة لحساب RSI يدويًا
def rsi(series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

# قائمة أسهم افتراضية
nasdaq_tickers = ["AAPL", "NVDA", "AMD", "MSFT", "TSLA"]

# اختيار الأسهم
selected_tickers = st.multiselect("اختر الأسهم المراد تحليلها:", nasdaq_tickers, default=nasdaq_tickers)

# دالة التحليل
def analyze_stock(ticker):
    end = datetime.date.today()
    start = end - datetime.timedelta(days=180)
    df = yf.download(ticker, start=start, end=end)

    if df.empty:
        return None

    df["RSI"] = rsi(df["Close"], length=14)

    latest_rsi = df["RSI"].iloc[-1]
    latest_close = df["Close"].iloc[-1]
    signal = None

    if latest_rsi < 30:
        signal = "📉 تشبع بيعي (قد تكون فرصة شراء)"
    elif latest_rsi > 70:
        signal = "📈 تشبع شرائي (احذر من التصحيح)"
    else:
        signal = "⚪ حيادي"

    return {
        "Ticker": ticker,
        "Close": round(latest_close, 2),
        "RSI": round(latest_rsi, 2),
        "Signal": signal
    }

# زر البحث
if st.button("🔍 ابحث عن فرص"):
    results = []
    for ticker in selected_tickers:
        data = analyze_stock(ticker)
        if data:
            results.append(data)

    if results:
        df_results = pd.DataFrame(results)
        st.dataframe(df_results)
    else:
        st.warning("لم يتم العثور على بيانات.")