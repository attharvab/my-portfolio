import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px

# --- CONFIG ---
st.set_page_config(page_title="Atharva's Global Alpha", layout="wide")
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUHmE__8dpl_nKGv5F5mTXO7e3EyVRqz-PJF_4yyIrfJAa7z8XgzkIw6IdLnaotkACka2Q-PvP8P-z/pub?output=csv"

# --- LOAD DATA ---
@st.cache_data(ttl=600)
def load_data():
    df = pd.read_csv(SHEET_URL)
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# --- LIVE PRICE ENGINE ---
st.title("🌍 Global Alpha Strategy Terminal")
if not df.empty:
    tickers = df['Ticker'].unique().tolist()
    data = yf.download(tickers, period="1d")['Close'].iloc[-1]
    
    # Process Portfolio
    df['Current Price'] = df['Ticker'].map(data)
    df['Total Gain %'] = ((df['Current Price'] - df['AvgCost']) / df['AvgCost']) * 100
    
    # Dashboard Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Positions", len(df))
    m2.metric("Market Status", "Live", delta="Updating")
    
    # The "Kitty" Table
    st.subheader("Current Holdings & Live Valuation")
    st.dataframe(df[['Ticker', 'Quantity', 'AvgCost', 'Current Price', 'Total Gain %']], use_container_width=True)

    # Benchmark Comparison (Simplified)
    st.subheader("Performance vs Benchmarks")
    bench = yf.download(['^GSPC', '^NSEI'], period="1mo")['Close']
    bench_norm = (bench / bench.iloc[0] - 1) * 100
    st.line_chart(bench_norm)
else:
    st.error("Check your Google Sheet! No tickers found.")
