import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, time as dtime
from zoneinfo import ZoneInfo
import plotly.express as px
import plotly.graph_objects as go
import time as _time

# ============================================================
# Atharva Portfolio Returns - FIXED & INTEGRATED
# ============================================================

APP_TITLE = "Atharva Portfolio Returns"
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📈")

# ========= YOUR SHEET URLS =========
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUHmE__8dpl_nKGv5F5mTXO7e3EyVRqz-PJF_4yyIrfJAa7z8XgzkIw6IdLnaotkACka2Q-PvP8P-z/pub?output=csv"
TRANSACTIONS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR-OybDEJRMpK5jvtLnMq3SOze-ZwT6hVY07w4nAnKfn1dva_E68fKSZQkn0yvzDhk217HEQ7xis77G/pub?output=csv"

REQUIRED_COLS = ["Ticker", "Region", "QTY", "AvgCost"]
OPTIONAL_COLS = ["Benchmark", "Type", "Thesis", "FirstBuyDate", "LivePrice_GS", "PrevClose_GS"]
MACRO_ASSETS = ["^GSPC", "^NSEI", "GC=F", "SI=F"]
ASSET_LABELS = {"^GSPC": "S&P 500", "^NSEI": "Nifty 50", "GC=F": "Gold", "SI=F": "Silver"}

# ============================================================
# Helpers & Cleaning
# ============================================================
def _clean_str(x):
    return str(x).strip() if pd.notna(x) else ""

def _as_float(x):
    try:
        if pd.isna(x): return None
        return float(str(x).replace(",", ""))
    except: return None

def _normalize_region(r):
    r = _clean_str(r).upper()
    return "US" if r in ["US", "USA", "UNITED STATES"] else "India"

def _is_us_ticker(tk):
    t = _clean_str(tk).upper()
    return not (t.endswith(".NS") or t.endswith(".BO") or t == "^NSEI")

def _fmt_pct(x):
    return x * 100 if x is not None else None

def _tooltip(label, help_text):
    return f'{label} <span title="{help_text}" style="cursor:default; border-bottom:1px dotted #888;">ⓘ</span>'

# ============================================================
# Data Loaders
# ============================================================
@st.cache_data(ttl=300)
def load_and_clean_data(url):
    df = pd.read_csv(url).dropna(how='all')
    df.columns = df.columns.str.strip()
    for c in OPTIONAL_COLS:
        if c not in df.columns: df[c] = ""
    df["Ticker"] = df["Ticker"].apply(_clean_str)
    df["Region"] = df["Region"].apply(_normalize_region)
    df["QTY"] = df["QTY"].apply(_as_float)
    df["AvgCost"] = df["AvgCost"].apply(_as_float)
    df = df[df["Ticker"] != ""].dropna(subset=["QTY", "AvgCost"])
    return df

@st.cache_data(ttl=300)
def load_transactions(url):
    df = pd.read_csv(url).dropna(how='all')
    df.columns = df.columns.str.strip()
    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    out["Ticker"] = df["Ticker"].apply(_clean_str)
    out["Qty"] = df["QTY"].apply(_as_float)
    # Ensure columns match for calculation
    out = out.dropna(subset=["Date", "Ticker", "Qty"])
    return out.sort_values("Date").reset_index(drop=True)

# ============================================================
# Pricing Engine
# ============================================================
@st.cache_data(ttl=600)
def build_prices(tickers, sheet_fallback):
    data = yf.download(tickers, period="5d", interval="1d", progress=False, threads=False)['Close']
    price_map = {}
    for tk in tickers:
        live, prev = None, None
        if isinstance(data, pd.DataFrame) and tk in data.columns:
            s = data[tk].dropna()
            if len(s) >= 1: live = s.iloc[-1]
            if len(s) >= 2: prev = s.iloc[-2]
        
        # Sheet Fallback if Yahoo fails
        if live is None: live = sheet_fallback.get(tk, {}).get("live")
        if prev is None: prev = sheet_fallback.get(tk, {}).get("prev")
        if prev is None: prev = live # Safety
        
        price_map[tk] = {"live": live, "prev": prev}
    return price_map

@st.cache_data(ttl=1800)
def build_equity_curve(txn):
    if txn.empty: return None
    start = txn["Date"].min()
    end = pd.Timestamp.now()
    tickers = txn["Ticker"].unique().tolist()
    
    px = yf.download(tickers + ["USDINR=X"], start=start, end=end, progress=False)["Close"].ffill()
    if px.empty: return None
    
    daily_val = []
    for d in px.index:
        t_so_far = txn[txn["Date"] <= d]
        inventory = t_so_far.groupby("Ticker")["Qty"].sum()
        total_inr = 0
        fx = px.loc[d, "USDINR=X"] if "USDINR=X" in px.columns else 83.0
        
        for tk, qty in inventory.items():
            if tk in px.columns:
                price = px.loc[d, tk]
                val = qty * price
                if _is_us_ticker(tk): val *= fx
                total_inr += val
        daily_val.append(total_inr)
    
    s = pd.Series(daily_val, index=px.index).dropna()
    return (s / s.iloc[0]) * 100 if not s.empty else None

# ============================================================
# Logic & UI
# ============================================================
try:
    df_sheet = load_and_clean_data(SHEET_URL)
    txn_df = load_transactions(TRANSACTIONS_URL)
    
    sheet_fb = {r["Ticker"]: {"live": r["LivePrice_GS"], "prev": r["PrevClose_GS"]} for _, r in df_sheet.iterrows()}
    all_tkrs = list(set(df_sheet["Ticker"].tolist() + MACRO_ASSETS))
    prices = build_prices(all_tkrs, sheet_fb)
    
    # Calculate Snapshot Metrics
    rows = []
    for _, r in df_sheet.iterrows():
        tk = r["Ticker"]
        p = prices.get(tk, {"live": None, "prev": None})
        if p["live"] and r["AvgCost"]:
            rows.append({
                "Ticker": tk, "Region": r["Region"], "Weight_Val": r["QTY"] * p["live"] * (83 if r["Region"]=="US" else 1),
                "Total_Ret": (p["live"] - r["AvgCost"]) / r["AvgCost"],
                "Day_Ret": (p["live"] - p["prev"]) / p["prev"] if p["prev"] else 0
            })
    calc_df = pd.DataFrame(rows)
    total_v = calc_df["Weight_Val"].sum()
    calc_df["W"] = calc_df["Weight_Val"] / total_v
    
    port_day = (calc_df["Day_Ret"] * calc_df["W"]).sum()
    port_total = (calc_df["Total_Ret"] * calc_df["W"]).sum()

    # --- INCEPTION ALPHA CALC (FIXED) ---
    portfolio_idx = build_equity_curve(txn_df)
    inception_alpha = None
    if portfolio_idx is not None:
        spx = yf.download("^GSPC", start=portfolio_idx.index[0], progress=False)["Close"].ffill()
        spx_idx = (spx / spx.iloc[0]) * 100
        common = pd.concat([portfolio_idx, spx_idx], axis=1).dropna()
        if not common.empty:
            p_gain = (common.iloc[-1, 0] / common.iloc[0, 0]) - 1
            s_gain = (common.iloc[-1, 1] / common.iloc[0, 1]) - 1
            inception_alpha = p_gain - s_gain

    # UI Rendering
    st.title("📈 Atharva Portfolio Returns")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Return", f"{port_total*100:.2f}%")
    m2.metric("Today's Return", f"{port_day*100:.2f}%")
    m3.metric("Alpha (vs S&P)", "TBD") # Placeholder for daily
    m4.metric("Inception Alpha", f"{inception_alpha*100:.2f}%" if inception_alpha else "—")

    # Equity Curve Plot
    st.subheader("Strategy vs Benchmarks (Base 100)")
    if portfolio_idx is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=portfolio_idx.index, y=portfolio_idx, name="Portfolio"))
        fig.add_trace(go.Scatter(x=spx_idx.index, y=spx_idx, name="S&P 500", line=dict(dash='dot')))
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # --- IMPROVED HEATMAP ---
    st.subheader("Asset Correlation Matrix")
    
    corr_data = yf.download(df_sheet["Ticker"].tolist(), period="1y", progress=False)["Close"].pct_change().corr()
    fig_hm = px.imshow(corr_data, text_auto=".2f", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig_hm, use_container_width=True)

    # Table
    st.dataframe(calc_df[["Ticker", "Region", "W", "Total_Ret", "Day_Ret"]], use_container_width=True)

except Exception as e:
    st.error(f"Initialization Error: {e}")
