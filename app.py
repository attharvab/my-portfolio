import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="Global Alpha Strategy", layout="wide", page_icon="📈")

SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUHmE__8dpl_nKGv5F5mTXO7e3EyVRqz-PJF_4yyIrfJAa7z8XgzkIw6IdLnaotkACka2Q-PvP8P-z/pub?output=csv"

# -----------------------------
# HELPERS
# -----------------------------
def parse_weight_to_decimal(series: pd.Series) -> pd.Series:
    """
    Accepts:
      "15.29%" -> 0.1529
      15.29    -> 0.1529 (assume percent)
      0.1529   -> 0.1529 (already decimal)
    """
    s = series.astype(str).str.strip().str.replace("%", "", regex=False).str.replace(",", "", regex=False)
    w = pd.to_numeric(s, errors="coerce")
    return w.where(w <= 1, w / 100.0)

@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv(SHEET_URL)
    df.columns = df.columns.str.strip()

    # remove summary rows like "India Portfolio", "US Portfolio", "Total"
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].astype(str)
        df = df[~df["Ticker"].str.contains("PORTFOLIO|TOTAL", case=False, na=False)]
        df = df[df["Ticker"].str.lower().ne("nan")]
        df = df.dropna(subset=["Ticker"])

    return df

@st.cache_data(ttl=300)
def fetch_market_data(tickers):
    """
    Live:
      - intraday 5m last close (more live)
      - fallback to latest daily close
    Prev:
      - previous daily close (yesterday)
    """
    rows = []
    for t in tickers:
        live, prev = None, None

        # 1) intraday "live-ish"
        try:
            intraday = yf.download(t, period="1d", interval="5m", progress=False)
            if not intraday.empty:
                c = intraday["Close"].dropna()
                if len(c) > 0:
                    live = float(c.iloc[-1])
        except Exception:
            pass

        # 2) daily fallback + prev close
        try:
            daily = yf.download(t, period="10d", interval="1d", progress=False)
            dc = daily["Close"].dropna()

            if live is None and len(dc) >= 1:
                live = float(dc.iloc[-1])

            if len(dc) >= 2:
                prev = float(dc.iloc[-2])
            elif len(dc) == 1:
                prev = float(dc.iloc[-1])
        except Exception:
            pass

        rows.append({"Ticker": t, "Live": live, "Prev": prev})

    return pd.DataFrame(rows)

# -----------------------------
# DATA ENGINE
# -----------------------------
try:
    df = load_data()

    required = ["Ticker", "PORTFOLIO WEIGHT", "AvgCost", "Type", "Thesis", "Region"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Header Mismatch: Missing {missing}")
        st.stop()

    df["AvgCost"] = pd.to_numeric(df["AvgCost"], errors="coerce")
    df["Weight"] = parse_weight_to_decimal(df["PORTFOLIO WEIGHT"])
    df["Region"] = df["Region"].astype(str).fillna("").str.strip()
    df["Thesis"] = df["Thesis"].astype(str).fillna("").str.strip()

    df = df.dropna(subset=["Ticker", "AvgCost", "Weight"])
    df = df[df["Weight"] > 0]

    if df.empty:
        st.error("No valid rows after cleaning. Check that PORTFOLIO WEIGHT and AvgCost are numeric.")
        st.stop()

    price_info = fetch_market_data(df["Ticker"].tolist())
    df = df.merge(price_info, on="Ticker", how="left")

    # CALCULATIONS
    df["Stock Return %"] = ((df["Live"] - df["AvgCost"]) / df["AvgCost"]) * 100
    df["Day Return %"] = ((df["Live"] - df["Prev"]) / df["Prev"]) * 100

    df["W_Return"] = df["Weight"] * df["Stock Return %"]
    df["W_Day"] = df["Weight"] * df["Day Return %"]

except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()

# -----------------------------
# UI - HEADER
# -----------------------------
st.title("🏛️ Global Alpha Strategy Terminal")
c1, c2, c3, c4 = st.columns(4)

total_ret = df["W_Return"].sum(skipna=True)
day_ret = df["W_Day"].sum(skipna=True)

c1.metric("Overall Strategy Return", f"{total_ret:.2f}%")
c2.metric("Day Change (Weighted)", f"{day_ret:.2f}%")
c3.metric("Assets Tracked", int(df["Ticker"].nunique()))
c4.metric("Last Sync", datetime.now().strftime("%H:%M:%S"))

st.divider()

with st.expander("🔧 Debug: price fetch status"):
    st.write(f"Tickers missing Live: {int(df['Live'].isna().sum())}")
    st.dataframe(df[["Ticker", "PORTFOLIO WEIGHT", "Weight", "AvgCost", "Live", "Prev"]], use_container_width=True)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🌎 Global View", "📌 Holdings & Thesis", "📈 Benchmarks"])

with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        reg_df = df.groupby("Region")["Weight"].sum().reset_index()
        fig1 = px.pie(reg_df, values="Weight", names="Region", title="Regional Exposure", hole=0.4)
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        reg_perf = df.groupby("Region").apply(lambda x: (x["W_Return"].sum() / x["Weight"].sum())).reset_index()
        reg_perf.columns = ["Region", "Perf %"]
        fig2 = px.bar(reg_perf, x="Region", y="Perf %", title="Regional Performance (Normalized)")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Interactive Strategy Map")

    st.dataframe(
        df[["Ticker", "Region", "Weight", "AvgCost", "Live", "Prev", "Stock Return %", "Day Return %"]],
        column_config={
            "Weight": st.column_config.NumberColumn(format="%.2%"),
            "Stock Return %": st.column_config.NumberColumn(format="%.2f%%"),
            "Day Return %": st.column_config.NumberColumn(format="%.2f%%"),
        },
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    tickers = sorted(df["Ticker"].unique().tolist())
    selected = st.selectbox("Select Asset to view Thesis", tickers)
    selection_df = df[df["Ticker"] == selected]

    if not selection_df.empty:
        row = selection_df.iloc[0]
        st.info(f"**{selected} Thesis:** {row['Thesis'] if pd.notna(row['Thesis']) else 'No notes added.'}")
    else:
        st.warning("Select a valid ticker to view details.")

with tab3:
    st.subheader("Performance vs. Market Benchmarks")
    bench_map = {"S&P 500": "^GSPC", "Nifty 50": "^NSEI", "Gold": "GC=F"}

    try:
        b_data = yf.download(list(bench_map.values()), period="3mo", interval="1d", progress=False)["Close"]
        if isinstance(b_data, pd.Series):
            b_data = b_data.to_frame()

        b_data = b_data.dropna(how="all").ffill()
        b_norm = (b_data / b_data.iloc[0] * 100)

        fig_b = px.line(b_norm, title="3-Month Indexed Performance (Base 100)")
        st.plotly_chart(fig_b, use_container_width=True)
    except Exception:
        st.info("Benchmark chart temporarily unavailable.")
