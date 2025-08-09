
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="البحث عن فرص في ناسداك", layout="centered")

st.title("📈 الباحث الذكي عن فرص الدخول في أسهم ناسداك")
st.markdown("تحليل فني تلقائي على الفاصل اليومي لاكتشاف الأسهم التي تمر بنموذج اختراق بعد تجميع.")

# قائمة أسهم ناسداك (أمثلة، ويمكن التعديل لاحقًا)
nasdaq_tickers = ["AAPL", "NVDA", "AMD", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "NFLX", "INTC"]

selected_tickers = st.multiselect("اختر الأسهم المراد تحليلها:", nasdaq_tickers, default=nasdaq_tickers[:5])


import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="البحث عن فرص في ناسداك", layout="centered")

st.title("📈 الباحث الذكي عن فرص الدخول في أسهم ناسداك")
st.markdown("تحليل فني تلقائي على الفاصل اليومي لاكتشاف الأسهم التي تمر بنموذج اختراق بعد تجميع.")

# قائمة أسهم ناسداك (أمثلة، ويمكن التعديل لاحقًا)
nasdaq_tickers = ["AAPL", "NVDA", "AMD", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "NFLX", "INTC"]

selected_tickers = st.multiselect("اختر الأسهم المراد تحليلها:", nasdaq_tickers, default=nasdaq_tickers[:5])

def analyze_stock(ticker):
    data = yf.download(ticker, period="6mo", interval="1d")
    if data.empty or len(data) < 30:
        return None

    data['20MA'] = data['Close'].rolling(window=20).mean()
    data['VolumeAvg'] = data['Volume'].rolling(window=20).mean()
    last = data.iloc[-1]

    # تحليل بسيط لاكتشاف الاختراق بفوليوم عالي
    if last['Close'] > last['20MA'] and last['Volume'] > 1.5 * last['VolumeAvg']:
        signal = "✅ مناسب للدخول (اختراق بفوليوم)"
    elif last['Close'] > last['20MA']:
        signal = "🟡 اختراق لكن بدون فوليوم قوي"
    else:
        signal = "❌ لا يوجد إشارة دخول واضحة"

    return {
        "signal": signal,
        "data": data
    }

if st.button("🔍 ابحث عن فرص"):
    for tkr in selected_tickers:
        with st.spinner(f"جاري تحليل {tkr}..."):
            result = analyze_stock(tkr)
            if result:
                st.subheader(f"📌 {tkr}")
                st.write(result['signal'])

                fig, ax = plt.subplots(figsize=(10, 3))
                result['data']['Close'].plot(ax=ax, label="السعر")
                result['data']['20MA'].plot(ax=ax, label="متوسط 20 يوم")
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning(f"لا توجد بيانات كافية للسهم {tkr}.")
    data = yf.download(ticker, period="6mo", interval="1d")
    if data.empty or len(data) < 30:
        return None

    data['20MA'] = data['Close'].rolling(window=20).mean()
    data['VolumeAvg'] = data['Volume'].rolling(window=20).mean()
    last = data.iloc[-1]

    # تحليل بسيط لاكتشاف الاختراق بفوليوم عالي
    if last['Close'] > last['20MA'] and last['Volume'] > 1.5 * last['VolumeAvg']:
        signal = "✅ مناسب للدخول (اختراق بفوليوم)"
    elif last['Close'] > last['20MA']:
        signal = "🟡 اختراق لكن بدون فوليوم قوي"
    else:
        signal = "❌ لا يوجد إشارة دخول واضحة"

    return {
        "signal": signal,
        "data": data
    }

if st.button("🔍 ابحث عن فرص"):
    for tkr in selected_tickers:
        with st.spinner(f"جاري تحليل {tkr}..."):
            result = analyze_stock(tkr)
            if result:
                st.subheader(f"📌 {tkr}")
                st.write(result['signal'])

                fig, ax = plt.subplots(figsize=(10, 3))
                result['data']['Close'].plot(ax=ax, label="السعر")
                result['data']['20MA'].plot(ax=ax, label="متوسط 20 يوم")
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning(f"لا توجد بيانات كافية للسهم {tkr}.")
