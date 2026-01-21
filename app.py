import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
import plotly.express as px

# 1) SETUP
st.set_page_config(page_title="Global Alpha Strategy", layout="wide", page_icon="📈")

SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUHmE__8dpl_nKGv5F5mTXO7e3EyVRqz-PJF_4yyIrfJAa7z8XgzkIw6IdLnaotkACka2Q-PvP8P-z/pub?output=csv"

@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv(SHEET_URL)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data(ttl=1800)
def get_usdinr_rate():
    try:
        fx = yf.download("USDINR=X", period="5d", interval="1d", progress=False)
        rate = float(fx["Close"].dropna().iloc[-1])
        return rate if rate > 0 else 83.0
    except Exception:
        return 83.0

@st.cache_data(ttl=300)
def fetch_prices(tickers_list):
    tickers_list = [t for t in tickers_list if isinstance(t, str) and t.strip() and t.lower() != "nan"]
    tickers_list = sorted(list(set([t.strip() for t in tickers_list])))

    if not tickers_list:
        return pd.DataFrame(columns=["Ticker", "Live Price", "Prev Close"])

    prices = yf.download(
        tickers=tickers_list,
        period="5d",
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=True
    )

    rows = []
    for t in tickers_list:
        try:
            if isinstance(prices.columns, pd.MultiIndex):
                close = prices[t]["Close"].dropna()
            else:
                # single ticker fallback
                close = prices["Close"].dropna()

            live = float(close.iloc[-1]) if len(close) >= 1 else None
            prev = float(close.iloc[-2]) if len(close) >= 2 else live
            rows.append({"Ticker": t, "Live Price": live, "Prev Close": prev})
        except Exception:
            rows.append({"Ticker": t, "Live Price": None, "Prev Close": None})

    return pd.DataFrame(rows)

# 2) LOAD + VALIDATE
try:
    df = load_data()
except Exception as e:
    st.error(f"Spreadsheet Connection Error: {e}")
    st.stop()

if df.empty:
    st.warning("Google Sheet is empty. Add tickers like AAPL or RELIANCE.NS.")
    st.stop()

required = ["Ticker", "Quantity", "AvgCost"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Header Mismatch: Missing {missing}")
    st.info("Your sheet must include exactly: Ticker, Quantity, AvgCost")
    st.stop()

# Clean types (important)
df["Ticker"] = df["Ticker"].astype(str).str.strip()
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["AvgCost"] = pd.to_numeric(df["AvgCost"], errors="coerce")
df = df.dropna(subset=["Ticker", "Quantity", "AvgCost"])

tickers = df["Ticker"].dropna().unique().tolist()

with st.spinner("Syncing Global Market Data..."):
    prices_df = fetch_prices(tickers)
    usd_inr_rate = get_usdinr_rate()

df = df.merge(prices_df, on="Ticker", how="left")

# 3) FLAGS + LOCAL CALCS
df["Is_India"] = df["Ticker"].str.contains(r"\.(NS|BO)$", case=False, na=False, regex=True)

df["Market Value Local"] = df["Quantity"] * df["Live Price"]
df["Cost Basis Local"] = df["Quantity"] * df["AvgCost"]

df["Market Value (USD)"] = df.apply(
    lambda x: x["Market Value Local"] / usd_inr_rate if x["Is_India"] else x["Market Value Local"], axis=1
)
df["Cost Basis (USD)"] = df.apply(
    lambda x: x["Cost Basis Local"] / usd_inr_rate if x["Is_India"] else x["Cost Basis Local"], axis=1
)

# 4) P&L CALCS
df["Gain/Loss %"] = None
mask = (df["AvgCost"].notna()) & (df["AvgCost"] > 0) & (df["Live Price"].notna())
df.loc[mask, "Gain/Loss %"] = ((df.loc[mask, "Live Price"] - df.loc[mask, "AvgCost"]) / df.loc[mask, "AvgCost"]) * 100

df["Total Gain (USD)"] = df["Market Value (USD)"] - df["Cost Basis (USD)"]

total_value_usd = df["Market Value (USD)"].sum(skipna=True)
total_cost_usd = df["Cost Basis (USD)"].sum(skipna=True)
total_gain_usd = total_value_usd - total_cost_usd
total_gain_pct = (total_gain_usd / total_cost_usd * 100) if total_cost_usd else 0

day_gain_usd = None
try:
    df["Day Gain Local"] = df["Quantity"] * (df["Live Price"] - df["Prev Close"])
    df["Day Gain (USD)"] = df.apply(
        lambda x: x["Day Gain Local"] / usd_inr_rate if x["Is_India"] else x["Day Gain Local"], axis=1
    )
    day_gain_usd = df["Day Gain (USD)"].sum(skipna=True)
except Exception:
    day_gain_usd = None

# 5) UI
st.title("🌍 Global Alpha Strategy Terminal")
st.markdown(f"**Portfolio Currency: USD** | FX: 1 USD = **{usd_inr_rate:.2f} INR**")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Net Liquidity", f"${total_value_usd:,.2f}")
c2.metric("Total P&L (USD)", f"${total_gain_usd:,.2f}", f"{total_gain_pct:,.2f}%")
c3.metric("Day P&L (USD)", f"${day_gain_usd:,.2f}" if day_gain_usd is not None else "NA")
c4.metric("Last Sync (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))

st.divider()

st.subheader("📊 Active Holdings")
st.dataframe(
    df[["Ticker", "Quantity", "AvgCost", "Live Price", "Gain/Loss %", "Market Value (USD)", "Total Gain (USD)"]],
    column_config={
        "Gain/Loss %": st.column_config.NumberColumn(format="%.2f %%"),
        "Market Value (USD)": st.column_config.NumberColumn(format="$ %.2f"),
        "Total Gain (USD)": st.column_config.NumberColumn(format="$ %.2f"),
        "Live Price": st.column_config.NumberColumn(format="%.2f"),
        "AvgCost": st.column_config.NumberColumn(format="%.2f"),
    },
    use_container_width=True,
    hide_index=True
)

# 6) BENCHMARKS
st.divider()
st.subheader("📈 Benchmarks (3M indexed to 100)")
bench_tickers = {"S&P 500": "^GSPC", "NIFTY 50": "^NSEI", "Gold": "GC=F"}

try:
    bench = yf.download(list(bench_tickers.values()), period="3mo", interval="1d", progress=False)["Close"]
    if isinstance(bench, pd.Series):
        bench = bench.to_frame()

    rename_map = {v: k for k, v in bench_tickers.items()}
    bench = bench.rename(columns=rename_map)

    bench = bench.dropna(how="all").ffill()
    # Ensure first row is not NaN for normalization
    bench = bench.loc[bench.notna().any(axis=1)]
    if bench.empty:
        raise ValueError("Benchmark data empty after cleaning.")

    bench_norm = (bench / bench.iloc[0] * 100).reset_index().melt(
        id_vars="Date",
        var_name="Benchmark",
        value_name="Index (Base=100)"
    )

    fig = px.line(bench_norm, x="Date", y="Index (Base=100)", color="Benchmark",
                  labels={"Index (Base=100)": "Indexed Performance", "Date": ""})
    fig.update_layout(legend_title_text="Benchmark")
    st.plotly_chart(fig, use_container_width=True)
except Exception:
    st.info("Benchmark chart temporarily unavailable.")

st.divider()
st.caption("Data source: Yahoo Finance (may be delayed). Educational project, not investment advice.")
