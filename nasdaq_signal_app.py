import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# -----------------------------
# إعداد الصفحة والعنوان
# -----------------------------
st.set_page_config(page_title="البحث الذكي لفرص ناسداك", layout="centered")
st.title("📈 الباحث الذكي عن فرص الدخول في أسهم ناسداك")
st.markdown("تحليل فني تلقائي على الفاصل اليومي لاكتشاف **اختراق بعد تصحيح** مع **فوليوم مرتفع** + ماسح **زخم اليوم**.")

# -----------------------------
# دوال مساعدة (RSI يدوي)
# -----------------------------
def rsi(series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

# -----------------------------
# قوائم افتراضية
# -----------------------------
sector_etfs = ["XLK","XLE","XLF","XLV","XLC","XLY","XLP","XLI","XLU","XLB","XLRE","SOXX","SMH","XSD"]
default_list = ["AAPL","NVDA","AMD","MSFT","TSLA","AMZN","META","GOOGL","NFLX","INTC"]

# قائمة أساس للزخم (قابلة للتوسيع)
BASE_LIST = [
    "AAPL","NVDA","AMD","MSFT","TSLA","AMZN","META","GOOGL","NFLX","INTC",
    "AVGO","ADBE","COST","PEP","ASML","LIN","CSCO","TMUS","TXN","QCOM",
    "AMAT","HON","ADI","INTU","VRTX","MU","MRVL","PANW","PDD","ABNB"
]

# -----------------------------
# خيارات جانبية
# -----------------------------
st.sidebar.header("⚙️ خيارات المسح")
raw_list = st.sidebar.text_area("ألصق رموز مفصولة بفواصل", ",".join(default_list), height=100)
extra = st.sidebar.multiselect("أضف صناديق قطاعات (ETFs)", sector_etfs)
tickers = [t.strip().upper() for t in (raw_list.split(",") + extra) if t.strip()]
period = st.sidebar.selectbox("الفترة الزمنية", ["6mo","1y","2y"], index=0)
max_symbols = st.sidebar.slider("عدد الرموز المُحلَّلة", 5, 100, 30)
vol_mult = st.sidebar.slider("شرط الفوليوم: أكبر من متوسط 20 يوم ×", 1.0, 3.0, 1.5, 0.1)
run_momentum = st.sidebar.checkbox("تشغيل ماسح الزخم (Top 10 اليوم)", value=True)

# -----------------------------
# جلب البيانات مع كاش
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60*30)
def load_data(tkr: str, period: str):
    return yf.download(tkr, period=period, interval="1d", auto_adjust=True, progress=False)

# -----------------------------
# ماسح الزخم (Top 10) — مقارنات scalar
# -----------------------------
def scan_momentum(tickers, vol_mult=1.5):
    end = datetime.date.today()
    start = end - datetime.timedelta(days=60)
    rows = []
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
            if df.empty or len(df) < 25:
                continue

            df["Vol20"] = df["Volume"].rolling(20).mean()
            today = df.iloc[-1]
            prev  = df.iloc[-2]

            pct = float((float(today["Close"]) - float(prev["Close"])) / float(prev["Close"]) * 100.0)

            vol20 = float(df["Vol20"].iloc[-1]) if not np.isnan(df["Vol20"].iloc[-1]) else 0.0
            vol_today = float(today["Volume"])
            vol_ok = (vol20 > 0.0) and (vol_today > vol_mult * vol20)

            rows.append({
                "Ticker": t,
                "PctToday": round(pct, 2),
                "Volume": int(vol_today),
                "Vol20": int(vol20),
                "VolSpike": vol_ok
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    dfm = pd.DataFrame(rows)
    dfm = dfm[(dfm["VolSpike"]) & (dfm["PctToday"] > 0)]
    return dfm.sort_values("PctToday", ascending=False).head(10)

# -----------------------------
# دالة التحليل الأساسية — كل المقارنات scalar
# -----------------------------
def analyze_stock(ticker: str, period: str = "6mo", vol_factor: float = 1.5):
    df = load_data(ticker, period)
    if df.empty or len(df) < 60:
        return None

    # مؤشرات
    df["EMA20"]  = df["Close"].ewm(span=20).mean()
    df["EMA50"]  = df["Close"].ewm(span=50).mean()
    df["EMA200"] = df["Close"].ewm(span=200).mean()
    df["RSI"]    = rsi(df["Close"], length=14)

    # VWAP تراكمي
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (tp * df["Volume"]).cumsum() / df["Volume"].replace(0, np.nan).cumsum()

    # متوسط حجم 20 يوم
    df["Vol20"] = df["Volume"].rolling(20).mean()

    # تقريب ترند هابط لآخر 20 قمة
    highs = df["High"].tail(20).reset_index(drop=True)
    x = np.arange(len(highs))
    try:
        m, _ = np.polyfit(x, highs.values, 1)
    except Exception:
        m = 0.0

    last = df.iloc[-1]

    close_v   = float(last["Close"])
    ema20_v   = float(last["EMA20"])
    ema50_v   = float(last["EMA50"])
    vwap_v    = float(last["VWAP"])
    rsi_v     = float(last["RSI"]) if not np.isnan(last["RSI"]) else 50.0
    vol20_v   = float(df["Vol20"].iloc[-1]) if not np.isnan(df["Vol20"].iloc[-1]) else 0.0
    vol_today = float(last["Volume"])

    vol_spike   = (vol20_v > 0.0) and (vol_today > vol_factor * vol20_v)
    above_ema20 = close_v > ema20_v
    above_vwap  = close_v > vwap_v
    above_ema50 = close_v > ema50_v
    downtrend   = bool(m < 0.0)    # scalar
    breakout    = bool(above_ema20 and above_vwap)

    # تصنيف الإشارة
    score = 0.0
    signal = "❌ لا توجد إشارة واضحة"
    if downtrend and breakout and vol_spike and rsi_v < 75.0:
        signal = "✅ دخول محتمل (اختراق بفوليوم)"
        score = 1.0
    elif breakout and rsi_v < 70.0:
        signal = "🟡 اختراق بدون فوليوم قوي (انتظار)"
        score = 0.6

    # إدارة مخاطر بسيطة
    recent_lows = df["Low"].tail(3)
    stop = float(recent_lows.min()) if recent_lows.notna().all() else float(last["Low"])
    risk = max(1e-6, close_v - stop)
    target1 = close_v + 2 * risk

    return {
        "data": df,
        "signal": signal,
        "score": score,
        "stop": stop,
        "target1": target1,
        "last_close": close_v,
        "rsi": rsi_v,
        "above_ema50": bool(above_ema50)
    }

# -----------------------------
# قسم الزخم قبل التحليل
# -----------------------------
st.subheader("🚀 أفضل 10 أسهم زخم اليوم")
top_df = pd.DataFrame()
if run_momentum:
    with st.spinner("جاري مسح الزخم..."):
        base_pool = list(set(BASE_LIST + tickers)) if tickers else BASE_LIST
        top_df = scan_momentum(base_pool, vol_mult=vol_mult)
    if top_df.empty:
        st.info("لا توجد أسهم زخم قوية اليوم وفق الفلاتر الحالية.")
    else:
        st.dataframe(top_df, use_container_width=True)
        if st.button("حلّل نتائج الزخم"):
            tickers = top_df["Ticker"].tolist()

# -----------------------------
# زر التشغيل وعرض النتائج
# -----------------------------
if st.button("🔍 ابحث عن فرص"):
    if not tickers:
        st.warning("أضف رموزًا أولاً من الشريط الجانبي أو اضغط 'حلّل نتائج الزخم'.")
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

                # الرسم: سعر + EMA/VWAP
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(df.index, df["Close"], label="السعر")
                ax.plot(df.index, df["EMA20"], label="EMA20")
                ax.plot(df.index, df["EMA50"], label="EMA50")
                ax.plot(df.index, df["EMA200"], label="EMA200")
                ax.plot(df.index, df["VWAP"], label="VWAP")
                ax.legend()
                st.pyplot(fig)

                # أحجام التداول
                st.bar_chart(df[["Volume","Vol20"]].tail(60))
else:
    st.info("اختر الرموز من الشريط الجانبي ثم اضغط **🔍 ابحث عن فرص** أو استخدم **حلّل نتائج الزخم**.")