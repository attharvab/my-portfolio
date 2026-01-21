import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
import plotly.express as px

# -----------------------------
# 1) SETUP
# -----------------------------
st.set_page_config(page_title="Global Alpha Strategy", layout="wide", page_icon="📈")

SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUHmE__8dpl_nKGv5F5mTXO7e3EyVRqz-PJF_4yyIrfJAa7z8XgzkIw6IdLnaotkACka2Q-PvP8P-z/pub?output=csv"

# -----------------------------
# 2) HELPERS
# -----------------------------
@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(SHEET_URL)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data(ttl=300)
def fetch_prices_individual(tickers_list):
    """
    Streamlit Cloud friendly: fetch per ticker to reduce yf.download failures.
    Returns: DataFrame with Ticker, Live Price, Prev Close
    """
    rows = []
    clean = []
    for t in tickers_list:
        if isinstance(t, str) and t.strip() and t.lower() != "nan":
            clean.append(t.strip())
    tickers_list = sorted(list(set(clean)))

    for t in tickers_list:
        try:
            hist = yf.Ticker(t).history(period="7d", interval="1d")
            hist = hist.dropna(subset=["Close"])
            if hist.empty:
                rows.append({"Ticker": t, "Live Price": None, "Prev Close": None})
                continue

            live = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else live
            rows.append({"Ticker": t, "Live Price": live, "Prev Close": prev})
        except Exception:
            rows.append({"Ticker": t, "Live Price": None, "Prev Close": None})

    return pd.DataFrame(rows)

@st.cache_data(ttl=600)
def fetch_benchmarks(bench_map, period="3mo"):
    # Benchmarks are fine with yf.download usually, but keep it simple.
    tickers = list(bench_map.values())
    bench = yf.download(tickers, period=period, interval="1d", progress=False)["Close"]
    if isinstance(bench, pd.Series):
        bench = bench.to_frame()
    rename_map = {v: k for k, v in bench_map.items()}
    bench = bench.rename(columns=rename_map)
    bench = bench.dropna(how="all").ffill()
    bench = bench.loc[bench.notna().any(axis=1)]
    return bench

# -----------------------------
# 3) LOAD + VALIDATE
# -----------------------------
try:
    df = load_data()
except Exception as e:
    st.error(f"Spreadsheet Connection Error: {e}")
    st.stop()

if df.empty:
    st.warning("Google Sheet is empty.")
    st.stop()

required = ["Ticker", "PORTFOLIO WEIGHT", "AvgCost", "Type", "Thesis", "Region", "Benchmark"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Header Mismatch: Missing {missing}")
    st.info("Your sheet must include exactly: Ticker, PORTFOLIO WEIGHT, AvgCost, Type, Thesis, Region, Benchmark")
    st.stop()

df["Ticker"] = df["Ticker"].astype(str).str.strip()
df["Type"] = df["Type"].astype(str).fillna("").str.strip()
df["Thesis"] = df["Thesis"].astype(str).fillna("").str.strip()
df["Region"] = df["Region"].astype(str).fillna("").str.strip()
df["Benchmark"] = df["Benchmark"].astype(str).fillna("").str.strip()
df["AvgCost"] = pd.to_numeric(df["AvgCost"], errors="coerce")

# weights accept "15.29%" or 15.29
w = df["PORTFOLIO WEIGHT"].astype(str).str.replace("%", "", regex=False).str.strip()
df["Weight"] = pd.to_numeric(w, errors="coerce") / 100.0

df = df.dropna(subset=["Ticker", "AvgCost", "Weight", "Region"])

tickers = df["Ticker"].dropna().unique().tolist()

with st.spinner("Syncing market data..."):
    prices_df = fetch_prices_individual(tickers)

df = df.merge(prices_df, on="Ticker", how="left")

# -----------------------------
# 4) RETURNS (WEIGHTS MODEL)
# -----------------------------
df["Stock Return %"] = None
mask = (df["AvgCost"] > 0) & (df["Live Price"].notna())
df.loc[mask, "Stock Return %"] = ((df.loc[mask, "Live Price"] / df.loc[mask, "AvgCost"]) - 1) * 100

df["Contribution %"] = df["Weight"] * df["Stock Return %"]
portfolio_return = df["Contribution %"].sum(skipna=True)

# Day return
df["Day Return %"] = None
mask_day = (df["Prev Close"].notna()) & (df["Prev Close"] > 0) & (df["Live Price"].notna())
df.loc[mask_day, "Day Return %"] = ((df.loc[mask_day, "Live Price"] / df.loc[mask_day, "Prev Close"]) - 1) * 100
df["Day Contribution %"] = df["Weight"] * df["Day Return %"]
portfolio_day_return = df["Day Contribution %"].sum(skipna=True)

def region_return(dfr):
    wsum = dfr["Weight"].sum()
    if wsum <= 0:
        return 0.0
    return float(dfr["Contribution %"].sum(skipna=True) / wsum)

regions = sorted(df["Region"].unique().tolist())
region_perf = {r: region_return(df[df["Region"] == r]) for r in regions}

# -----------------------------
# 5) UI HEADER
# -----------------------------
st.title("🌍 Global Alpha Strategy Terminal")
st.caption("Weights-based dashboard (no real investment values).")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Portfolio Return (Since AvgCost)", f"{portfolio_return:.2f}%")
c2.metric("Day Return (Approx)", f"{portfolio_day_return:.2f}%")
c3.metric("Holdings Count", f"{len(df)}")
c4.metric("Last Sync (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))

# -----------------------------
# 6) DEBUG (THIS WILL SHOW YOU WHAT IS FAILING)
# -----------------------------
with st.expander("🔧 Debug: price fetch status", expanded=False):
    missing_px = df[df["Live Price"].isna()][["Ticker", "Region", "AvgCost", "Weight"]]
    st.write(f"Tickers missing Live Price: {len(missing_px)}")
    if len(missing_px) > 0:
        st.dataframe(missing_px, use_container_width=True)
    st.dataframe(df[["Ticker", "AvgCost", "Weight", "Live Price", "Prev Close"]], use_container_width=True)

st.divider()

# -----------------------------
# 7) TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🌎 Global Summary", "📌 Holdings + Thesis", "📈 Benchmarks"])

with tab1:
    st.subheader("Global Allocation")
    alloc = df.groupby("Region")["Weight"].sum().reset_index()
    alloc["Weight %"] = alloc["Weight"] * 100

    fig_alloc = px.bar(alloc, x="Region", y="Weight %", text="Weight %", title="Portfolio Weight by Region")
    st.plotly_chart(fig_alloc, use_container_width=True)

    st.subheader("Regional Performance (Normalized)")
    perf_df = pd.DataFrame({"Region": list(region_perf.keys()), "Return %": list(region_perf.values())})
    fig_perf = px.bar(perf_df, x="Region", y="Return %", text="Return %", title="Region Return % (normalized)")
    st.plotly_chart(fig_perf, use_container_width=True)

    st.subheader("Top Contributors")
    topc = df.dropna(subset=["Contribution %"]).sort_values("Contribution %", ascending=False).head(10)
    st.dataframe(
        topc[["Ticker", "Region", "Type", "Weight", "Stock Return %", "Contribution %"]],
        column_config={
            "Weight": st.column_config.NumberColumn(format="%.4f"),
            "Stock Return %": st.column_config.NumberColumn(format="%.2f %%"),
            "Contribution %": st.column_config.NumberColumn(format="%.2f %%"),
        },
        use_container_width=True,
        hide_index=True
    )

with tab2:
    st.subheader("📊 Holdings (Weights-based)")
    show_df = df.copy()
    show_df["Weight %"] = show_df["Weight"] * 100

    st.dataframe(
        show_df[["Ticker", "Region", "Type", "Weight %", "AvgCost", "Live Price", "Stock Return %", "Contribution %"]],
        column_config={
            "Weight %": st.column_config.NumberColumn(format="%.2f %%"),
            "Stock Return %": st.column_config.NumberColumn(format="%.2f %%"),
            "Contribution %": st.column_config.NumberColumn(format="%.2f %%"),
            "Live Price": st.column_config.NumberColumn(format="%.2f"),
            "AvgCost": st.column_config.NumberColumn(format="%.2f"),
        },
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.subheader("🧠 Thesis Viewer")
    tickers_sorted = sorted(df["Ticker"].dropna().unique().tolist())
    selected = st.selectbox("Select a ticker", tickers_sorted)
    row = df[df["Ticker"] == selected].iloc[0]

    st.markdown(f"**Region:** {row.get('Region','')}")
    st.markdown(f"**Type:** {row.get('Type','')}")
    st.markdown(f"**Weight:** {(row.get('Weight',0)*100):.2f}%")
    st.text_area("Investment thesis (edit in Google Sheet)", value=str(row.get("Thesis","")), height=220)

with tab3:
    st.subheader("Benchmarks (Indexed to 100)")
    bench_map = {"S&P 500": "^GSPC", "NIFTY 50": "^NSEI", "Gold": "GC=F"}

    try:
        bench = fetch_benchmarks(bench_map, period="3mo")
        if bench.empty:
            raise ValueError("Benchmark data empty.")

        bench_norm = (bench / bench.iloc[0] * 100).reset_index().melt(
            id_vars="Date",
            var_name="Benchmark",
            value_name="Index (Base=100)"
        )
        fig = px.line(bench_norm, x="Date", y="Index (Base=100)", color="Benchmark")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Benchmark chart temporarily unavailable.")

st.divider()
st.caption("Data source: Yahoo Finance (may be delayed). Educational project, not investment advice.")
