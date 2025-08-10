import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_ta as ta

st.set_page_config(page_title="الباحث الذكي لفرص ناسداك", layout="centered")
st.title("📈 الباحث الذكي عن فرص الدخول في أسهم ناسداك")
st.markdown("تحليل فني تلقائي على الفاصل اليومي لاكتشاف **اختراق بعد تصحيح** مع **فوليوم مرتفع**.")

sector_etfs = ["XLK","XLE","XLF","XLV","XLC","XLY","XLP","XLI","XLU","XLB","XLRE","SOXX","SMH","XSD"]
default_list = ["AAPL","NVDA","AMD","MSFT","TSLA","AMZN","META","GOOGL","NFLX","INTC"]

st.sidebar.header("⚙️ خيارات المسح")
raw_list = st.sidebar.text_area("ألصق رموز مفصولة بفواصل", ",".join(default_list), height=100)
extra = st.sidebar.multiselect("أضف صناديق قطاعات (ETFs)", sector_etfs)
tickers = [t.strip().upper() for t in (raw_list.split(",") + extra) if t.strip()]
period = st.sidebar.selectbox("الفترة الزمنية", ["6mo","1y","2y"], index=0)
max_symbols = st.sidebar.slider("عدد الرموز المُحلَّلة", 5, 100, 30)
vol_mult = st.sidebar.slider("شرط الفوليوم: أكبر من متوسط 20 يوم ×", 1.0, 3.0, 1.5, 0.1)

@st.cache_data(show_spinner=False, ttl=60*30)
def load_data(tkr: str, period: str):
    df = yf.download(tkr, period=period, interval="1d", auto_adjust=True, progress=False)
    return df

def analyze_stock(ticker: str, period: str = "6mo", vol_factor: float = 1.5):
    df = load_data(ticker, period)
    if df.empty or len(df) < 60:
        return None

    # مؤشرات
    df["EMA20"]  = df["Close"].ewm(span=20).mean()
    df["EMA50"]  = df["Close"].ewm(span=50).mean()
    df["EMA200"] = df["Close"].ewm(span=200).mean()
    df["RSI"]    = ta.rsi(df["Close"], length=14)

    # VWAP تراكمي
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (tp * df["Volume"]).cumsum() / df["Volume"].replace(0, np.nan).cumsum()

    # متوسط حجم 20 يوم
    df["Vol20"] = df["Volume"].rolling(20).mean()

    # تقريب ترند هابط لآخر 20 قمة
    N = 20
    highs = df["High"].tail(N).reset_index(drop=True)
    x = np.arange(len(highs))
    try:
        m, b = np.polyfit(x, highs.values, 1)
    except Exception:
        m = 0.0

    last = df.iloc[-1]

    vol_spike   = (df["Vol20"].iloc[-1] > 0) and (last["Volume"] > vol_factor * df["Vol20"].iloc[-1])
    above_ema20 = last["Close"] > last["EMA20"]
    above_vwap  = last["Close"] > last["VWAP"]
    above_ema50 = last["Close"] > last["EMA50"]
    downtrend   = m < 0
    breakout    = above_ema20 and above_vwap

    # تصنيف الإشارة
    score = 0.0
    signal = "❌ لا توجد إشارة واضحة"
    if downtrend and breakout and vol_spike and last["RSI"] < 75:
        signal = "✅ دخول محتمل (اختراق بفوليوم)"
        score = 1.0
    elif breakout and last["RSI"] < 70:
        signal = "🟡 اختراق بدون فوليوم قوي (انتظار)"
        score = 0.6

    # إدارة مخاطر بسيطة
    recent_lows = df["Low"].tail(3)
    stop = float(recent_lows.min()) if recent_lows.notna().all() else float(last["Low"])
    risk = max(1e-6, last["Close"] - stop)
    target1 = float(last["Close"] + 2 * risk)

    return {
        "data": df,
        "signal": signal,
        "score": score,
        "stop": stop,
        "target1": target1,
        "last_close": float(last["Close"]),
        "rsi": float(last["RSI"]) if not np.isnan(last["RSI"]) else None,
        "above_ema50": bool(above_ema50)
    }

if st.button("🔍 ابحث عن فرص"):
    if not tickers:
        st.warning("أضف رموزًا أولاً.")
    else:
        results = []
        for t in tickers[:max_symbols]:
            with st.spinner(f"تحليل {t}..."):
                try:
                    r = analyze_stock(t, period=period, vol_factor=vol_mult)
                except Exception as e:
                    st.info(f"{t}: تعذر التحليل ({e})")
                    continue
                if r:
                    results.append((t, r["score"], r))

        # ترتيب بحسب أعلى Score
        results.sort(key=lambda x: x[1], reverse=True)

        if not results:
            st.info("لا توجد نتائج وفق الشروط الحالية.")
        else:
            for t, sc, r in results:
                df = r["data"]
                st.subheader(f"📌 {t} — {r['signal']}")
                rsi_txt = f"{r['rsi']:.1f}" if r["rsi"] is not None else "—"
                st.caption(
                    f"سعر الإغلاق: {r['last_close']:.2f} | RSI: {rsi_txt} | "
                    f"وقف مقترح: {r['stop']:.2f} | الهدف₁: {r['target1']:.2f} | Score: {sc:.2f}"
                )

                # الرسم
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(df.index, df["Close"], label="السعر")
                ax.plot(df.index, df["EMA20"], label="EMA20")
                ax.plot(df.index, df["EMA50"], label="EMA50")
                ax.plot(df.index, df["EMA200"], label="EMA200")
                ax.plot(df.index, df["VWAP"], label="VWAP")
                ax.legend()
                st.pyplot(fig)

                st.bar_chart(df[["Volume","Vol20"]].tail(60))
else:
    st.info("اختر الرموز من الشريط الجانبي ثم اضغط زر **🔍 ابحث عن فرص**.")