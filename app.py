import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

# 1) SETUP
st.set_page_config(page_title="Global Alpha Terminal", layout="wide", page_icon="📈")

SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUHmE__8dpl_nKGv5F5mTXO7e3EyVRqz-PJF_4yyIrfJAa7z8XgzkIw6IdLnaotkACka2Q-PvP8P-z/pub?output=csv"

@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv(SHEET_URL)
    df.columns = df.columns.str.strip()
    return df

st.title("🌍 Global Alpha Strategy Terminal")

# 2) LOAD + VALIDATE
try:
    df = load_data()
except Exception as e:
    st.error(f"Spreadsheet Error: {e}")
    st.stop()

if df.empty:
    st.warning("No data found. Ensure your Google Sheet has rows filled in.")
    st.stop()

required = ["Ticker", "Quantity", "AvgCost"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns in Google Sheet: {missing}")
    st.info("Fix the header names in Google Sheets to match exactly: Ticker, Quantity, AvgCost")
    st.stop()

# Clean types
df["Ticker"] = df["Ticker"].astype(str).str.strip()
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["AvgCost"] = pd.to_numeric(df["AvgCost"], errors="coerce")

st.markdown(f"**Tracking {df['Ticker'].nunique()} Active Positions**")

tickers = df["Ticker"].dropna().unique().tolist()
tickers = [t for t in tickers if t and t.lower() != "nan"]

# 3) PRICE FETCH (robust)
@st.cache_data(ttl=300)
def fetch_prices(tickers_list: list[str]) -> pd.DataFrame:
    if not tickers_list:
        return pd.DataFrame(columns=["Ticker", "Live Price"])
    prices = yf.download(
        tickers=tickers_list,
        period="5d",
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=True,
    )

    rows = []
    for t in tickers_list:
        try:
            if isinstance(prices.columns, pd.MultiIndex):
                close = prices[t]["Close"].dropna()
            else:
                close = prices["Close"].dropna()

            live = float(close.iloc[-1]) if not close.empty else None
            rows.append({"Ticker": t, "Live Price": live})
        except Exception:
            rows.append({"Ticker": t, "Live Price": None})

    return pd.DataFrame(rows)

with st.spinner("Syncing with Global Markets..."):
    prices_df = fetch_prices(tickers)

df = df.merge(prices_df, on="Ticker", how="left")

# 4) CALCULATIONS (safe)
df["Value ($)"] = df["Quantity"] * df["Live Price"]

# Avoid division by zero / NaN
df["Gain/Loss %"] = None
mask = (df["AvgCost"].notna()) & (df["AvgCost"] != 0) & (df["Live Price"].notna())
df.loc[mask, "Gain/Loss %"] = ((df.loc[mask, "Live Price"] - df.loc[mask, "AvgCost"]) / df.loc[mask, "AvgCost"]) * 100

# Portfolio totals
total_val = df["Value ($)"].sum(skipna=True)

# 5) DISPLAY
col1, col2, col3 = st.columns(3)
col1.metric("Portfolio Value", f"${total_val:,.2f}")
col2.metric("Market Status", "🟢 Live Data")
col3.metric("Last Updated", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))

st.subheader("📊 Current Holdings")
st.dataframe(
    df[["Ticker", "Quantity", "AvgCost", "Live Price", "Gain/Loss %", "Value ($)"]],
    column_config={
        "Gain/Loss %": st.column_config.NumberColumn(format="%.2f %%"),
        "Value ($)": st.column_config.NumberColumn(format="$ %.2f"),
        "Live Price": st.column_config.NumberColumn(format="%.2f"),
        "AvgCost": st.column_config.NumberColumn(format="%.2f"),
    },
    use_container_width=True,
    hide_index=True
)

st.divider()
st.caption("Data source: Yahoo Finance. Updates every 5 minutes. Educational purposes only, not investment advice.")


is this a better code?
