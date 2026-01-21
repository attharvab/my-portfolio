import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
import plotly.express as px

# --- SETUP ---
st.set_page_config(page_title="Global Alpha Terminal", layout="wide", page_icon="🏛️")

SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUHmE__8dpl_nKGv5F5mTXO7e3EyVRqz-PJF_4yyIrfJAa7z8XgzkIw6IdLnaotkACka2Q-PvP8P-z/pub?output=csv"

# --- 1. DEFENSIVE DATA LOADING (ChatGPT logic + Gemini Lean style) ---
@st.cache_data(ttl=300)
def load_and_clean_data(url):
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        
        # Filter out empty or summary rows
        df = df[df['Ticker'].notna()]
        df = df[~df['Ticker'].str.contains("TOTAL|PORTFOLIO|SUMMARY|CASH", case=False, na=False)]
        
        # Region Normalizer
        def normalize_region(r):
            r = str(r).upper().strip()
            if r in ['US', 'USA', 'UNITED STATES']: return 'US'
            if r in ['INDIA', 'IN', 'IND']: return 'India'
            return r
        df['Region'] = df['Region'].apply(normalize_region)
        
        # Numeric Force
        df['QTY'] = pd.to_numeric(df['QTY'], errors='coerce')
        df['AvgCost'] = pd.to_numeric(df['AvgCost'], errors='coerce')
        
        # Aggregate Duplicates (Handle multiple buys)
        df['Total_Cost'] = df['QTY'] * df['AvgCost']
        agg = df.groupby(['Ticker', 'Region', 'Benchmark']).agg({
            'QTY': 'sum',
            'Total_Cost': 'sum',
            'Type': 'first',
            'Thesis': 'first'
        }).reset_index()
        agg['AvgCost'] = agg['Total_Cost'] / agg['QTY']
        
        return agg.dropna(subset=['Ticker', 'QTY', 'AvgCost'])
    except Exception as e:
        st.error(f"Spreadsheet Error: {e}")
        return pd.DataFrame()

# --- 2. FAST-INFO PRICE ENGINE ---
def get_live_and_prev(tickers):
    # Combined approach: yfinance download for history + fast_info for intraday
    data = yf.download(tickers, period="5d", interval="1d", progress=False)['Close']
    price_map = {}
    
    for t in tickers:
        try:
            # Fallback chain for live price
            ticker_obj = yf.Ticker(t)
            live = ticker_obj.fast_info.get("last_price")
            
            series = data[t].dropna()
            if live is None or pd.isna(live):
                live = float(series.iloc[-1])
            
            # Prev close is always the one prior to today's data
            prev = float(series.iloc[-2]) if len(series) >= 2 else live
            
            price_map[t] = {'live': live, 'prev': prev}
        except:
            price_map[t] = {'live': None, 'prev': None}
    return price_map

# --- DATA EXECUTION ---
df_sheet = load_and_clean_data(SHEET_URL)
if df_sheet.empty: st.stop()

all_tickers = list(set(df_sheet['Ticker'].tolist() + df_sheet['Benchmark'].tolist() + ["USDINR=X", "^GSPC", "^NSEI", "GC=F"]))

with st.spinner("Executing Alpha Engine..."):
    prices = get_live_and_prev(all_tickers)
    live_fx = prices.get("USDINR=X", {}).get("live", 83.5)
    trend_data = yf.download(["^GSPC", "^NSEI", "GC=F"], period="5y", interval="1mo", progress=False)['Close']

# --- 3. ALPHA CALCULATIONS ---
rows = []
failures = []

for _, row in df_sheet.iterrows():
    t, b = row['Ticker'], row['Benchmark']
    p_tk = prices.get(t, {})
    p_bench = prices.get(b, {})

    if p_tk.get('live') and p_bench.get('live'):
        # Native Performance (FX Ignored)
        s_total_ret = (p_tk['live'] - row['AvgCost']) / row['AvgCost']
        s_day_ret = (p_tk['live'] - p_tk['prev']) / p_tk['prev']
        
        # Benchmark Performance
        b_day_ret = (p_bench['live'] - p_bench['prev']) / p_bench['prev']
        
        # Individual Alpha (Philosophy: Did the pick beat its benchmark today?)
        alpha_day = s_day_ret - b_day_ret
        
        # Global Weighting (Value in INR)
        val_inr = row['QTY'] * p_tk['live'] * (live_fx if row['Region'] == 'US' else 1.0)
        
        rows.append({
            "Ticker": t, "Region": row['Region'], "Benchmark": b, 
            "Value_INR": val_inr, "Total_Ret": s_total_ret, 
            "Day_Ret": s_day_ret, "Alpha_Day": alpha_day,
            "Type": row.get('Type', ''), "Thesis": row.get('Thesis', '')
        })
    else:
        failures.append(t)

calc_df = pd.DataFrame(rows)
calc_df['Weight'] = calc_df['Value_INR'] / calc_df['Value_INR'].sum()

# --- 4. WEIGHTED BENCHMARK LOGIC ---
in_weight = calc_df[calc_df['Region'] == 'India']['Weight'].sum()
us_weight = calc_df[calc_df['Region'] == 'US']['Weight'].sum()

nifty_day = (prices["^NSEI"]['live'] / prices["^NSEI"]['prev']) - 1
sp500_day = (prices["^GSPC"]['live'] / prices["^GSPC"]['prev']) - 1

custom_bench_day = (in_weight * nifty_day) + (us_weight * sp500_day)
port_day = (calc_df['Day_Ret'] * calc_df['Weight']).sum()
port_total = (calc_df['Total_Ret'] * calc_df['Weight']).sum()

# --- 5. UI DISPLAY ---
st.title("🏛️ Global Alpha Terminal")
st.caption(f"Sync Time: {datetime.now(timezone.utc).strftime('%H:%M')} UTC | Benchmarks: Nifty 50, S&P 500, Gold")

m1, m2, m3 = st.columns(3)
m1.metric("Strategy Total Return", f"{port_total*100:.2f}%")
m2.metric("Portfolio Day Move", f"{port_day*100:.2f}%", f"{(port_day - custom_bench_day)*100:.2f}% vs Weighted Market")
m3.metric("Live USD/INR", f"{live_fx:.2f}")

if failures:
    st.warning(f"⚠️ Pricing failed for: {', '.join(failures)}. Check Ticker symbols.")

st.divider()

tab1, tab2, tab3 = st.tabs(["🌎 Global View", "📌 Pick Analysis", "📈 5Y Macro"])

with tab1:
    c1, c2 = st.columns([2, 1])
    with c1:
        # Indexed 5Y Chart
        trend_norm = (trend_data / trend_data.iloc[0] * 100)
        fig = px.line(trend_norm, title="5-Year Macro Benchmarks (Indexed to 100)")
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        # Regional Pie
        fig_pie = px.pie(calc_df, values='Weight', names='Region', hole=0.5, title="Currency Exposure (INR Value)")
        st.plotly_chart(fig_pie, use_container_width=True)
        st.info(f"**India Allocation:** {in_weight*100:.2f}%")
        st.success(f"**US Allocation:** {us_weight*100:.2f}%")

with tab2:
    st.subheader("Performance & Alpha Matrix")
    st.dataframe(
        calc_df[['Ticker', 'Region', 'Weight', 'Total_Ret', 'Day_Ret', 'Alpha_Day']],
        column_config={
            "Weight": st.column_config.NumberColumn(format="%.2f%%"),
            "Total_Ret": st.column_config.NumberColumn("Total Ret", format="%.2f%%"),
            "Day_Ret": st.column_config.NumberColumn("Day Ret", format="%.2f%%"),
            "Alpha_Day": st.column_config.NumberColumn("Alpha (vs Bench)", format="%.2f%%"),
        },
        use_container_width=True, hide_index=True
    )
    
    st.divider()
    # Detailed Pick Breakdown
    sel_ticker = st.selectbox("View Pick Thesis", sorted(calc_df['Ticker'].tolist()))
    pick_data = calc_df[calc_df['Ticker'] == sel_ticker].iloc[0]
    st.write(f"**Type:** {pick_data['Type']} | **Benchmark:** {pick_data['Benchmark']}")
    st.text_area("Investment Thesis", value=pick_data['Thesis'], height=150, disabled=True)

with tab3:
    st.subheader("Benchmark Comparison (Raw Prices)")
    st.dataframe(trend_data.tail(10), use_container_width=True)
