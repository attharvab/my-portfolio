import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

# 1) SETUP & CONFIG
st.set_page_config(page_title="Global Alpha Strategy", layout="wide", page_icon="📈")

# Your Published CSV Link
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUHmE__8dpl_nKGv5F5mTXO7e3EyVRqz-PJF_4yyIrfJAa7z8XgzkIw6IdLnaotkACka2Q-PvP8P-z/pub?output=csv"

@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv(SHEET_URL)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data(ttl=3600)
def get_conversion_rate():
    """Fetches Live USD/INR rate"""
    try:
        data = yf.download("USDINR=X", period="1d", interval="1m")
        return data['Close'].iloc[-1]
    except:
        return 83.0 # Fallback rate if API fails

# 2) LOAD + VALIDATE DATA
try:
    df = load_data()
except Exception as e:
    st.error(f"Spreadsheet Connection Error: {e}")
    st.stop()

if df.empty:
    st.warning("Google Sheet is empty. Add tickers like 'AAPL' or 'RELIANCE.NS'.")
    st.stop()

required = ["Ticker", "Quantity", "AvgCost"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Header Mismatch: Missing {missing}")
    st.stop()

# Data Cleaning
df["Ticker"] = df["Ticker"].astype(str).str.strip()
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["AvgCost"] = pd.to_numeric(df["AvgCost"], errors="coerce")

# 3) DYNAMIC PRICE ENGINE
tickers = df["Ticker"].dropna().unique().tolist()

@st.cache_data(ttl=300)
def fetch_prices(tickers_list):
    if not tickers_list: return pd.DataFrame()
    prices = yf.download(tickers_list, period="5d", interval="1d", progress=False)
    
    rows = []
    for t in tickers_list:
        try:
            if len(tickers_list) > 1:
                close = prices["Close"][t].dropna()
            else:
                close = prices["Close"].dropna()
            live = float(close.iloc[-1]) if not close.empty else None
            rows.append({"Ticker": t, "Live Price": live})
        except:
            rows.append({"Ticker": t, "Live Price": None})
    return pd.DataFrame(rows)

with st.spinner("Syncing Global Market Data..."):
    prices_df = fetch_prices(tickers)
    usd_inr_rate = get_conversion_rate()

df = df.merge(prices_df, on="Ticker", how="left")

# 4) CURRENCY & P&L CALCULATIONS
# Identify Indian Stocks
df['Is_India'] = df['Ticker'].str.contains('.NS|.BO', case=False, na=False)

# Calculate Individual Value in Original Currency
df['Market Value Local'] = df['Quantity'] * df['Live Price']

# Convert Everything to USD for Portfolio Totals
df['Market Value (USD)'] = df.apply(
    lambda x: x['Market Value Local'] / usd_inr_rate if x['Is_India'] else x['Market Value Local'], axis=1
)

# Gain/Loss Calculations
df["Gain/Loss %"] = None
mask = (df["AvgCost"].notna()) & (df["AvgCost"] > 0) & (df["Live Price"].notna())
df.loc[mask, "Gain/Loss %"] = ((df.loc[mask, "Live Price"] - df.loc[mask, "AvgCost"]) / df.loc[mask, "AvgCost"]) * 100

total_value_usd = df["Market Value (USD)"].sum(skipna=True)

# 5) THE DASHBOARD UI
st.title("🌍 Global Alpha Strategy Terminal")
st.markdown(f"**Portfolio Currency: USD** | Live Exchange Rate: 1 USD = {usd_inr_rate:.2f} INR")

col1, col2, col3 = st.columns(3)
col1.metric("Total Net Liquidity", f"${total_value_usd:,.2f}")
col2.metric("Market Status", "🟢 OPEN" if datetime.utcnow().hour < 20 else "🔴 CLOSED")
col3.metric("Last Sync", datetime.now().strftime("%H:%M:%S"))

st.divider()

# Main Holdings Table
st.subheader("📊 Active Holdings")
st.dataframe(
    df[["Ticker", "Quantity", "AvgCost", "Live Price", "Gain/Loss %", "Market Value (USD)"]],
    column_config={
        "Gain/Loss %": st.column_config.NumberColumn(format="%.2f %%"),
        "Market Value (USD)": st.column_config.NumberColumn(format="$ %.2f"),
        "Live Price": st.column_config.NumberColumn(format="%.2f"),
        "AvgCost": st.column_config.NumberColumn(format="%.2f"),
    },
    use_container_width=True,
    hide_index=True
)

# 6) BENCHMARK COMPARISON
st.divider()
st.subheader("📈 Performance vs Global Benchmarks")
bench_tickers = ["^GSPC", "^NSEI", "GC=F"] # S&P 500, Nifty 50, Gold
bench_data = yf.download(bench_tickers, period="3mo")['Close']
# Normalize to 100
bench_norm = (bench_data / bench_data.iloc[0] * 100)

st.line_chart(bench_norm)
st.caption("Indexed Performance (Base 100): Blue = S&P 500 | Red = Nifty 50 | Green = Gold")

st.divider()
st.caption("DISCLAIMER: This is a personal project for educational purposes. Data is delayed by 15 mins.")
