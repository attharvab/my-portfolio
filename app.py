import streamlit as st
import pandas as pd
import yfinance as yf

# 1. SETUP
st.set_page_config(page_title="Global Alpha Terminal", layout="wide", page_icon="📈")

# YOUR PUBLISHED CSV LINK FROM SCREENSHOT
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUHmE__8dpl_nKGv5F5mTXO7e3EyVRqz-PJF_4yyIrfJAa7z8XgzkIw6IdLnaotkACka2Q-PvP8P-z/pub?output=csv"

@st.cache_data(ttl=300)
def load_data():
    try:
        df = pd.read_csv(SHEET_URL)
        df.columns = df.columns.str.strip() # Removes accidental spaces
        return df
    except Exception as e:
        st.error(f"Spreadsheet Error: {e}")
        return pd.DataFrame()

# 2. DATA PROCESSING
df = load_data()

st.title("🌍 Global Alpha Strategy Terminal")
st.markdown(f"**Tracking {len(df)} Active Positions**")

if not df.empty:
    # Get unique tickers (handles AAPL, RELIANCE.NS, etc.)
    tickers = df['Ticker'].dropna().unique().tolist()
    
    # Robust price fetching
    with st.spinner('Syncing with Global Markets...'):
        prices_df = yf.download(tickers, period="5d", interval="1d", progress=False)

    def get_price(ticker):
        try:
            # Multi-ticker vs Single-ticker check
            if len(tickers) > 1:
                return prices_df['Close'][ticker].dropna().iloc[-1]
            else:
                return prices_df['Close'].dropna().iloc[-1]
        except:
            return 0.0

    # 3. CALCULATIONS
    df['Live Price'] = df['Ticker'].apply(get_price)
    df['Gain/Loss %'] = ((df['Live Price'] - df['AvgCost']) / df['AvgCost']) * 100
    df['Value ($)'] = df['Quantity'] * df['Live Price']

    # 4. DISPLAY
    col1, col2 = st.columns(2)
    total_val = df['Value ($)'].sum()
    col1.metric("Portfolio Value", f"${total_val:,.2f}")
    col2.metric("Market Status", "🟢 Live Data")

    st.subheader("📊 Current Holdings")
    st.dataframe(
        df[['Ticker', 'Quantity', 'AvgCost', 'Live Price', 'Gain/Loss %', 'Value ($)']],
        column_config={
            "Gain/Loss %": st.column_config.NumberColumn(format="%.2f %%"),
            "Value ($)": st.column_config.NumberColumn(format="$ %.2f"),
        },
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("No data found. Ensure your Google Sheet Ticker column is filled!")

st.divider()
st.caption("Data source: Yahoo Finance. Updates every 5 minutes.")
