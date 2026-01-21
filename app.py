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

# If your sheet is weights-based now, set this to True.
# Expected headers when True: Ticker, PORTFOLIO WEIGHT, AvgCost, Type, Thesis, Region, Benchmark(optional)
USE_WEIGHTS = True

# -----------------------------
# 2) DATA LOADERS
# -----------------------------
@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv(SHEET_URL)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data(ttl=1800)
def get_usdinr_rate():
    """Fetch USD/INR FX rate. Fallback if Yahoo fails."""
    try:
        fx = yf.download("USDINR=X", period="5d", interval="1d", progress=False)
        rate = float(fx["Close"].dropna().iloc[-1])
        return rate if rate > 0 else 83.0
    except Exception:
        return 83.0  # fallback

@st.cache_data(ttl=300)
def fetch_prices(tickers_list):
    """
    Live Price logic:
      - Try intraday 5m close (true live-ish) if available
      - Fallback to last daily close
    Prev Close logic:
      - Always prior daily close (yesterday close)
    """
    tickers_list = [t for t in tickers_list if isinstance(t, str) and t.strip() and t.lower() != "nan"]
    tickers_list = sorted(list(set([t.strip() for t in tickers_list])))

    if not tickers_list:
        return pd.DataFrame(columns=["Ticker", "Live Price", "Prev Close"])

    rows = []
    for t in tickers_list:
        live_price = None
        prev_close = None

        # 1) Try intraday first (true live-ish)
        try:
            intraday = yf.download(t, period="1d", interval="5m", progress=False)
            if not intraday.empty:
                s = intraday["Close"].dropna()
                if len(s) > 0:
                    live_price = float(s.iloc[-1])
        except Exception:
            pass

        # 2) Daily data for fallback + prev close
        try:
            daily = yf.download(t, period="10d", interval="1d", progress=False)
            daily_close = daily["Close"].dropna()

            if live_price is None and len(daily_close) >= 1:
                live_price = float(daily_close.iloc[-1])

            if len(daily_close) >= 2:
                prev_close = float(daily_close.iloc[-2])
            elif len(daily_close) == 1:
                prev_close = float(daily_close.iloc[-1])
        except Exception:
            pass

        rows.append({"Ticker": t, "Live Price": live_price, "Prev Close": prev_close})

    return pd.DataFrame(rows)

# -----------------------------
# 3) LOAD + VALIDATE
# -----------------------------
try:
    df = load_data()
except Exception as e:
    st.error(f"Spreadsheet Connection Error: {e}")
    st.stop()

if df.empty:
    st.warning("Google Sheet is empty. Add holdings rows first.")
    st.stop()

df["Ticker"] = df["Ticker"].astype(str).str.strip()

if USE_WEIGHTS:
    # Your new sheet columns (as in screenshot)
    # Ticker | PORTFOLIO WEIGHT | AvgCost | Type | Thesis | Region | Benchmark
    required = ["Ticker", "PORTFOLIO WEIGHT", "AvgCost", "Type", "Thesis", "Region"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Header Mismatch: Missing {missing}")
        st.info("Expected headers: Ticker, PORTFOLIO WEIGHT, AvgCost, Type, Thesis, Region (Benchmark optional)")
        st.stop()

    df["PORTFOLIO WEIGHT"] = pd.to_numeric(df["PORTFOLIO WEIGHT"], errors="coerce")
    df["AvgCost"] = pd.to_numeric(df["AvgCost"], errors="coerce")
    df["Type"] = df["Type"].astype(str).fillna("").str.strip()
    df["Thesis"] = df["Thesis"].astype(str).fillna("").str.strip()
    df["Region"] = df["Region"].astype(str).fillna("").str.strip()

    # Keep only valid holdings rows (exclude any summary rows if they sneak in)
    df = df.dropna(subset=["Ticker", "PORTFOLIO WEIGHT", "AvgCost"])
    df = df[df["Ticker"].str.lower().ne("nan")]
    df = df[df["PORTFOLIO WEIGHT"] > 0]

    # Normalize weights to sum to 1 (per region + global)
    df["Weight"] = df["PORTFOLIO WEIGHT"] / 100.0

else:
    # Old quantity-based sheet
    required = ["Ticker", "Quantity", "AvgCost", "Type", "Thesis"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Header Mismatch: Missing {missing}")
        st.info("Expected headers: Ticker, Quantity, AvgCost, Type, Thesis")
        st.stop()

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["AvgCost"] = pd.to_numeric(df["AvgCost"], errors="coerce")
    df["Type"] = df["Type"].astype(str).fillna("").str.strip()
    df["Thesis"] = df["Thesis"].astype(str).fillna("").str.strip()
    df = df.dropna(subset=["Ticker", "Quantity", "AvgCost"])

# -----------------------------
# 4) FETCH PRICES + FX
# -----------------------------
tickers = df["Ticker"].dropna().unique().tolist()

with st.spinner("Syncing market data..."):
    prices_df = fetch_prices(tickers)
    usd_inr_rate = get_usdinr_rate()

df = df.merge(prices_df, on="Ticker", how="left")

# -----------------------------
# 5) CALCS
# -----------------------------
df["Is_India"] = df["Ticker"].str.contains(r"\.(NS|BO)$", case=False, na=False, regex=True)

# Stock return since AvgCost
df["Stock Return %"] = None
mask = (df["AvgCost"].notna()) & (df["AvgCost"] > 0) & (df["Live Price"].notna())
df.loc[mask, "Stock Return %"] = ((df.loc[mask, "Live Price"] - df.loc[mask, "AvgCost"]) / df.loc[mask, "AvgCost"]) * 100

# Day return (approx): (Live - PrevClose) / PrevClose
df["Day Return %"] = None
mask_day = (df["Prev Close"].notna()) & (df["Prev Close"] > 0) & (df["Live Price"].notna())
df.loc[mask_day, "Day Return %"] = ((df.loc[mask_day, "Live Price"] - df.loc[mask_day, "Prev Close"]) / df.loc[mask_day, "Prev Close"]) * 100

if USE_WEIGHTS:
    # Contribution = weight * return
    df["Contribution %"] = (df["Weight"] * df["Stock Return %"]).astype(float)
    df["Day Contribution %"] = (df["Weight"] * df["Day Return %"]).astype(float)

    portfolio_return = df["Contribution %"].sum(skipna=True)
    day_return = df["Day Contribution %"].sum(skipna=True)

    holdings_count = int(df["Ticker"].nunique())

else:
    # Quantity-based (USD view)
    df["Market Value Local"] = df["Quantity"] * df["Live Price"]
    df["Cost Basis Local"] = df["Quantity"] * df["AvgCost"]

    df["Market Value (USD)"] = df.apply(
        lambda x: x["Market Value Local"] / usd_inr_rate if x["Is_India"] else x["Market Value Local"], axis=1
    )
    df["Cost Basis (USD)"] = df.apply(
        lambda x: x["Cost Basis Local"] / usd_inr_rate if x["Is_India"] else x["Cost Basis Local"], axis=1
    )

    df["Total Gain (USD)"] = df["Market Value (USD)"] - df["Cost Basis (USD)"]

    total_value_usd = df["Market Value (USD)"].sum(skipna=True)
    total_cost_usd = df["Cost Basis (USD)"].sum(skipna=True)
    total_gain_usd = total_value_usd - total_cost_usd
    portfolio_return = (total_gain_usd / total_cost_usd * 100) if total_cost_usd else 0

    # day P&L based on PrevClose
    df["Day Gain Local"] = df["Quantity"] * (df["Live Price"] - df["Prev Close"])
    df["Day Gain (USD)"] = df.apply(
        lambda x: x["Day Gain Local"] / usd_inr_rate if x["Is_India"] else x["Day Gain Local"], axis=1
    )
    day_gain_usd = df["Day Gain (USD)"].sum(skipna=True)
    day_return = None  # not meaningful without a portfolio base here
    holdings_count = int(df["Ticker"].nunique())

# -----------------------------
# 6) UI
# -----------------------------
st.title("🌍 Global Alpha Strategy Terminal")

if USE_WEIGHTS:
    st.caption("Weights-based dashboard (no real investment values).")
else:
    st.markdown(f"**Portfolio Currency: USD** | FX: 1 USD = **{usd_inr_rate:.2f} INR**")

c1, c2, c3, c4 = st.columns(4)

if USE_WEIGHTS:
    c1.metric("Portfolio Return (Since AvgCost)", f"{portfolio_return:,.2f}%")
    c2.metric("Day Return (Approx)", f"{day_return:,.2f}%")
    c3.metric("Holdings Count", f"{holdings_count}")
    c4.metric("Last Sync (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
else:
    c1.metric("Total Net Liquidity", f"${total_value_usd:,.2f}")
    c2.metric("Total P&L (USD)", f"${total_gain_usd:,.2f}", f"{portfolio_return:,.2f}%")
    c3.metric("Day P&L (USD)", f"${day_gain_usd:,.2f}")
    c4.metric("Last Sync (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))

st.divider()

# Optional debug
with st.expander("🔧 Debug: price fetch status"):
    miss = df["Live Price"].isna().sum()
    st.write(f"Tickers missing Live Price: {miss}")
    show_cols = ["Ticker", "AvgCost", "Live Price", "Prev Close"]
    if USE_WEIGHTS:
        show_cols.insert(2, "Weight")
    st.dataframe(df[show_cols].head(30), use_container_width=True)

# -----------------------------
# 7) TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🌍 Global Summary", "📌 Holdings + Thesis", "📈 Benchmarks"])

with tab2:
    st.subheader("📊 Holdings (Weights-based)" if USE_WEIGHTS else "📊 Active Holdings")

    if USE_WEIGHTS:
        view = df[["Ticker", "Region", "Type", "PORTFOLIO WEIGHT", "AvgCost", "Live Price", "Stock Return %", "Contribution %", "Thesis"]].copy()
        view = view.rename(columns={"PORTFOLIO WEIGHT": "Weight %"})
        st.dataframe(
            view,
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
    else:
        st.dataframe(
            df[["Ticker", "Type", "Quantity", "AvgCost", "Live Price", "Stock Return %", "Market Value (USD)", "Total Gain (USD)"]],
            column_config={
                "Stock Return %": st.column_config.NumberColumn(format="%.2f %%"),
                "Market Value (USD)": st.column_config.NumberColumn(format="$ %.2f"),
                "Total Gain (USD)": st.column_config.NumberColumn(format="$ %.2f"),
                "Live Price": st.column_config.NumberColumn(format="%.2f"),
                "AvgCost": st.column_config.NumberColumn(format="%.2f"),
            },
            use_container_width=True,
            hide_index=True
        )

    st.divider()
    tickers_sorted = sorted(df["Ticker"].dropna().unique().tolist())
    selected = st.selectbox("Select a ticker", tickers_sorted)
    row = df[df["Ticker"] == selected].iloc[0]
    st.markdown("### 🧠 Thesis")
    st.markdown(f"**Type:** {row.get('Type','')}")
    st.text_area("Investment thesis (edit in Google Sheet)", value=str(row.get("Thesis","")), height=220)

with tab1:
    st.subheader("Global Allocation")

    if USE_WEIGHTS:
        # Portfolio weight by Region
        regional = df.groupby("Region")["Weight"].sum().reset_index()
        regional["Weight %"] = regional["Weight"] * 100

        fig_region = px.bar(regional, x="Region", y="Weight %", title="Portfolio Weight by Region")
        st.plotly_chart(fig_region, use_container_width=True)

        # Pie by Type
        by_type = df.groupby("Type")["Weight"].sum().reset_index()
        by_type["Weight %"] = by_type["Weight"] * 100
        fig_type = px.pie(by_type, values="Weight %", names="Type", hole=0.5, title="Allocation by Asset Type")
        st.plotly_chart(fig_type, use_container_width=True)

        # Contribution breakdown
        contrib = df[["Ticker", "Contribution %"]].dropna().sort_values("Contribution %", ascending=False)
        fig_contrib = px.bar(contrib, x="Ticker", y="Contribution %", title="Return Contribution by Holding")
        st.plotly_chart(fig_contrib, use_container_width=True)
    else:
        st.info("Global summary charts are optimized for weights-based mode. Switch USE_WEIGHTS=True to use the new layout.")

with tab3:
    st.subheader("📈 Benchmarks (3M indexed to 100)")

    # If you added a Benchmark column, use it; otherwise default set
    default_bench = {"S&P 500": "^GSPC", "NIFTY 50": "^NSEI", "Gold": "GC=F"}

    try:
        bench_list = list(default_bench.values())
        bench = yf.download(bench_list, period="3mo", interval="1d", progress=False)["Close"]
        if isinstance(bench, pd.Series):
            bench = bench.to_frame()

        rename_map = {v: k for k, v in default_bench.items()}
        bench = bench.rename(columns=rename_map)

        bench = bench.dropna(how="all").ffill()
        bench = bench.loc[bench.notna().any(axis=1)]
        if bench.empty:
            raise ValueError("Benchmark data empty after cleaning.")

        bench_norm = (bench / bench.iloc[0] * 100).reset_index().melt(
            id_vars="Date",
            var_name="Benchmark",
            value_name="Index (Base=100)"
        )

        fig = px.line(
            bench_norm,
            x="Date",
            y="Index (Base=100)",
            color="Benchmark",
            labels={"Index (Base=100)": "Indexed Performance", "Date": ""}
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Benchmark chart temporarily unavailable.")

# -----------------------------
# 8) FOOTER
# -----------------------------
st.divider()
st.caption("Data source: Yahoo Finance (may be delayed). Educational project, not investment advice.")
