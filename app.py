import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
import plotly.express as px

# 1) SETUP
st.set_page_config(page_title="Global Alpha Strategy", layout="wide", page_icon="📈")

SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUHmE__8dpl_nKGv5F5mTXO7e3EyVRqz-PJF_4yyIrfJAa7z8XgzkIw6IdLnaotkACka2Q-PvP8P-z/pub?output=csv"

# 2) HELPERS
@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv(SHEET_URL)
    df.columns = df.columns.str.strip()
    # CLEANING: Remove summary rows like 'INDIAN PORTFOLIO' or 'US PORTFOLIO'
    if 'Ticker' in df.columns:
        df = df[~df['Ticker'].str.contains("PORTFOLIO|Total|INDIAN|US", case=False, na=False)]
        df = df.dropna(subset=['Ticker'])
    return df

@st.cache_data(ttl=300)
def fetch_market_data(tickers):
    if not tickers: return pd.DataFrame()
    # Fetch 5 days to ensure we get a valid 'Previous Close' even on Mondays
    data = yf.download(tickers, period="5d", interval="1d", progress=False)
    
    rows = []
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                hist = data['Close'][t].dropna()
            else:
                hist = data['Close'].dropna()
            
            live = float(hist.iloc[-1])
            prev = float(hist.iloc[-2]) if len(hist) > 1 else live
            rows.append({"Ticker": t, "Live": live, "Prev": prev})
        except:
            rows.append({"Ticker": t, "Live": None, "Prev": None})
    return pd.DataFrame(rows)

# 3) DATA ENGINE
try:
    df = load_data()
    # Parse Weights and Costs
    df['Weight'] = pd.to_numeric(df['PORTFOLIO WEIGHT'].astype(str).str.replace('%',''), errors='coerce') / 100
    df['AvgCost'] = pd.to_numeric(df['AvgCost'], errors='coerce')
    
    # Sync Prices
    price_info = fetch_market_data(df['Ticker'].tolist())
    df = df.merge(price_info, on="Ticker", how="left")
    
    # CALCULATIONS
    # Return since purchase
    df['Stock Return %'] = ((df['Live'] - df['AvgCost']) / df['AvgCost']) * 100
    # Today's move
    df['Day Return %'] = ((df['Live'] - df['Prev']) / df['Prev']) * 100
    # Weighted contribution to the TOTAL portfolio
    df['W_Return'] = df['Weight'] * df['Stock Return %']
    df['W_Day'] = df['Weight'] * df['Day Return %']

except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()

# 4) UI - HEADER
st.title("🏛️ Global Alpha Strategy Terminal")
c1, c2, c3, c4 = st.columns(4)

total_ret = df['W_Return'].sum()
day_ret = df['W_Day'].sum()

c1.metric("Overall Strategy Return", f"{total_ret:.2f}%")
c2.metric("Day Change (Weighted)", f"{day_ret:.2f}%")
c3.metric("Assets Tracked", len(df))
c4.metric("Last Sync", datetime.now().strftime("%H:%M:%S"))

st.divider()

# 5) TABS
tab1, tab2, tab3 = st.tabs(["🌎 Global View", "📌 Holdings & Thesis", "📈 Benchmarks"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        # Allocation by Region
        reg_df = df.groupby('Region')['Weight'].sum().reset_index()
        fig1 = px.pie(reg_df, values='Weight', names='Region', title="Regional Exposure", hole=0.4)
        st.plotly_chart(fig1, use_container_width=True)
    with col_b:
        # Performance by Region
        # We divide by regional sum to show how the region itself performed
        reg_perf = df.groupby('Region').apply(lambda x: (x['W_Return'].sum() / x['Weight'].sum())).reset_index()
        reg_perf.columns = ['Region', 'Perf %']
        fig2 = px.bar(reg_perf, x='Region', y='Perf %', title="Regional Performance (Normalized)")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    # THE FIX FOR YOUR INDEX ERROR: 
    # Use selection_df and an 'if not empty' check
    st.subheader("Interactive Strategy Map")
    
    # Table View
    st.dataframe(df[['Ticker', 'Region', 'Weight', 'AvgCost', 'Live', 'Stock Return %']], 
                 column_config={"Weight": st.column_config.NumberColumn(format="%.2%"),
                                "Stock Return %": st.column_config.NumberColumn(format="%.2f%%")},
                 hide_index=True, use_container_width=True)
    
    st.divider()
    
    # Thesis Viewer
    selected = st.selectbox("Select Asset to view Thesis", df['Ticker'].unique())
    selection_df = df[df['Ticker'] == selected]
    
    if not selection_df.empty:
        row = selection_df.iloc[0]
        st.info(f"**{selected} Thesis:** {row['Thesis'] if pd.notna(row['Thesis']) else 'No notes added.'}")
    else:
        st.warning("Select a valid ticker to view details.")

with tab3:
    st.subheader("Performance vs. Market Benchmarks")
    bench_map = {"S&P 500": "^GSPC", "Nifty 50": "^NSEI", "Gold": "GC=F"}
    b_data = yf.download(list(bench_map.values()), period="3mo")['Close']
    b_norm = (b_data / b_data.iloc[0] * 100) # Index to 100
    fig_b = px.line(b_norm, title="3-Month Indexed Performance (Base 100)")
    st.plotly_chart(fig_b, use_container_width=True)
