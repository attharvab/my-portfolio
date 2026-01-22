# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, time as dtime
from zoneinfo import ZoneInfo
import plotly.express as px
import plotly.graph_objects as go
import time as _time

APP_TITLE = "Atharva Portfolio Returns"
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📈")

# SHEET URLS ────────────────────────────────────────────────────────────────
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUHmE__8dpl_nKGv5F5mTXO7e3EyVRqz-PJF_4yyIrfJAa7z8XgzkIw6IdLnaotkACka2Q-PvP8P-z/pub?output=csv"
TRANSACTIONS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR-OybDEJRMpK5jvtLnMq3SOze-ZwT6hVY07w4nAnKfn1dva_E68fKSZQkn0yvzDhk217HEQ7xis77G/pub?output=csv"

# Config ────────────────────────────────────────────────────────────────────
REQUIRED_COLS = ["Ticker", "Region", "QTY", "AvgCost"]
OPTIONAL_COLS = ["Benchmark", "Type", "Thesis", "FirstBuyDate", "LivePrice_GS", "PrevClose_GS"]
SUMMARY_NOISE_REGEX = r"TOTAL|PORTFOLIO|SUMMARY|CASH"
DEFAULT_BENCH = {"us": "^GSPC", "india": "^NSEI"}

# Helpers ───────────────────────────────────────────────────────────────────
def _clean_str(x):
    if pd.isna(x): return ""
    return str(x).strip()

def _as_float(x):
    try:
        if pd.isna(x): return None
        s = str(x).strip().replace(",", "")
        if s == "": return None
        return float(s)
    except:
        return None

def _normalize_region(r):
    r = _clean_str(r).upper()
    if r in ["US", "USA", "UNITED STATES", "UNITEDSTATES"]: return "US"
    if r in ["INDIA", "IN", "IND"]: return "India"
    return r

def _region_key(region: str) -> str:
    r = _clean_str(region).lower()
    return "us" if r == "us" else "india" if r == "india" else r

def _default_benchmark_for_region(region: str) -> str:
    return DEFAULT_BENCH.get(_region_key(region), "")

def _parse_date(x):
    try:
        if pd.isna(x) or str(x).strip() == "": return pd.NaT
        return pd.to_datetime(str(x).strip(), errors="coerce")
    except:
        return pd.NaT

def _fmt_pct(x):
    if x is None or pd.isna(x): return None
    return x * 100

def _bench_context(bench: str):
    b = _clean_str(bench)
    if b == "^NSEI": return "vs Nifty 50 (India)"
    if b == "^GSPC": return "vs S&P 500 (US)"
    if b in ["GC=F", "SI=F"]: return f"vs {b}"
    return f"vs {b}" if b else "—"

def _status_tag(alpha_day, bench):
    if alpha_day is None or pd.isna(alpha_day): return "—"
    short = "Nifty" if bench == "^NSEI" else ("S&P" if bench == "^GSPC" else "Index")
    return f"🔥 Beating ({short})" if alpha_day >= 0 else f"❄️ Lagging ({short})"

def _tooltip(label: str, help_text: str):
    safe = help_text.replace('"', "'")
    return f"{label} <span title='{safe}' style='cursor:default;'>ⓘ</span>"

def _is_india_ticker(tk: str) -> bool:
    t = _clean_str(tk).upper()
    return t.endswith(".NS") or t.endswith(".BO") or t in ["^NSEI"]

def _is_us_ticker(tk: str) -> bool:
    return not _is_india_ticker(tk)

# Market open logic (unchanged) ─────────────────────────────────────────────
def _is_market_open(now_utc: datetime, market: str) -> bool:
    if market == "US":
        tz = ZoneInfo("America/New_York")
        now = now_utc.astimezone(tz)
        if now.weekday() >= 5: return False
        return dtime(9, 30) <= now.time() <= dtime(16, 0)
    if market == "India":
        tz = ZoneInfo("Asia/Kolkata")
        now = now_utc.astimezone(tz)
        if now.weekday() >= 5: return False
        return dtime(9, 15) <= now.time() <= dtime(15, 30)
    return False

def _market_status_badge(now_utc: datetime, calc_df):
    us_open = _is_market_open(now_utc, "US")
    in_open = _is_market_open(now_utc, "India")
    if calc_df is None or calc_df.empty or "Day_Ret" not in calc_df.columns:
        return ("🟡 Markets closed or data unavailable.", "closed" if not us_open and not in_open else "mixed")
    # ... (rest unchanged)
    return "🟩 Markets open.", "open"  # simplified for brevity

# Load holdings ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_and_clean_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    # ... (your original cleaning logic unchanged)
    # return aggregated df
    return agg.reset_index(drop=True)  # ← keep your full logic here

# Load transactions ──────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_transactions(url: str) -> pd.DataFrame:
    if not url.strip(): return pd.DataFrame()
    df = pd.read_csv(url)
    if df.empty: return pd.DataFrame()
    
    df.columns = df.columns.astype(str).str.strip()
    keep_cols = [c for c in df.columns if not c.startswith('Unnamed')]
    df = df[keep_cols].copy()
    
    required = ["Ticker", "Date", "QTY"]
    for col in required:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            return pd.DataFrame()
    
    n = len(df)
    out = pd.DataFrame(index=range(n))
    
    # Date parsing
    date_parsed = pd.to_datetime(df["Date"].astype(str).str.strip(), format="%d-%b-%Y", errors="coerce")
    out["Date"] = date_parsed.values
    if date_parsed.isna().all():
        out["Date"] = pd.to_datetime(df["Date"], errors="coerce").values
    
    out["Ticker"] = df["Ticker"].apply(_clean_str).values
    out["Qty"]   = df["QTY"].apply(_as_float).values
    out["Region"] = df.get("Region", pd.Series([""]*n)).apply(_normalize_region).values
    out["FX_Rate"] = df.get("FX_Rate", pd.Series([None]*n)).apply(_as_float).values
    out["Type"]   = df.get("Type", pd.Series([""]*n)).apply(_clean_str).values
    
    out = out.dropna(subset=["Date", "Ticker", "Qty"]).copy()
    out = out[out["Ticker"].str.len() > 0]
    out = out.sort_values("Date").reset_index(drop=True)
    
    # Infer missing region
    mask = out["Region"].str.strip() == ""
    if mask.any():
        out.loc[mask, "Region"] = out.loc[mask, "Ticker"].apply(lambda t: "India" if _is_india_ticker(t) else "US")
    
    return out

# Pricing functions (unchanged) ─────────────────────────────────────────────
# ... keep your _fast_live_prev, fetch_history_closes_chunked, build_prices_with_sheet_fallback, fetch_fx_usdinr_snapshot

# Portfolio daily returns for calendar ──────────────────────────────────────
@st.cache_data(ttl=1800)
def build_portfolio_daily_returns():
    txn = load_transactions(TRANSACTIONS_URL)
    if txn.empty: return None

    start = pd.to_datetime(txn["Date"].min()).normalize()
    end   = pd.Timestamp.utcnow().normalize()

    tickers = sorted(txn["Ticker"].dropna().unique())
    if not tickers: return None

    px = yf.download(tickers, start=start-pd.Timedelta(10,"D"), end=end+pd.Timedelta(1,"D"),
                     interval="1d", progress=False, threads=False)["Close"]

    if px.empty: return None
    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])

    # Fill missing tickers individually
    missing = [t for t in tickers if t not in px.columns]
    for tk in missing:
        try:
            one = yf.download(tk, start=px.index.min(), end=px.index.max()+pd.Timedelta(1,"D"),
                              progress=False, threads=False)
            if "Close" in one:
                px[tk] = one["Close"]
        except:
            continue

    px = px.ffill().dropna(how="all")

    fx = yf.download("USDINR=X", start=px.index.min(), end=px.index.max()+pd.Timedelta(1,"D"),
                     progress=False, threads=False)["Close"].reindex(px.index).ffill()

    pos = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    d = txn.copy()
    d["Day"] = pd.to_datetime(d["Date"]).dt.normalize()
    deltas = d.groupby(["Day","Ticker"])["Qty"].sum()

    for tk in pos.columns:
        s = deltas.get(tk, pd.Series()).reindex(pos.index).fillna(0).cumsum()
        pos[tk] = s

    pos = pos.clip(lower=0)

    px_inr = px.copy()
    if not fx.empty:
        fx_a = fx.reindex(px.index).ffill()
        for tk in px_inr.columns:
            if _is_us_ticker(tk) and tk not in ["GC=F","SI=F"]:
                px_inr[tk] *= fx_a

    value = (pos * px_inr).sum(axis=1).replace([float("inf"),-float("inf")],pd.NA).dropna()
    value = value[value > 0]

    if len(value) < 2: return None

    ret = value.pct_change().dropna()
    ret.name = "Portfolio Daily Return"
    return ret

# Webull-style P&L Calendar ─────────────────────────────────────────────────
def render_webull_pnl_calendar():
    ret = build_portfolio_daily_returns()
    if ret is None or ret.empty:
        st.info("P&L calendar unavailable — not enough data.")
        return

    years = sorted(ret.index.year.unique())
    if not years:
        st.info("No calendar data available.")
        return

    year = st.selectbox("Year", years, index=len(years)-1)
    month_names = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    month = st.selectbox("Month", range(1,13), format_func=lambda m: month_names[m-1],
                         index=datetime.now().month-1)

    month_data = ret[(ret.index.year == year) & (ret.index.month == month)]

    if month_data.empty:
        st.info(f"No data for {month_names[month-1]} {year}")
        return

    df = pd.DataFrame({"Date": month_data.index, "Return": month_data.values})
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday  # 0=Mon ... 6=Sun
    df["Week"] = df["Date"].dt.isocalendar().week

    ret_pivot = df.pivot_table(index="Weekday", columns="Week", values="Return", aggfunc="first")
    day_pivot = df.pivot_table(index="Weekday", columns="Week", values="Day",   aggfunc="first")

    ret_pivot = ret_pivot.reindex(range(7))
    day_pivot = day_pivot.reindex(range(7))

    fig = go.Figure()

    # Background color
    fig.add_trace(go.Heatmap(
        z=ret_pivot.values,
        x=ret_pivot.columns.astype(str),
        y=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        colorscale=[[0,"#c0392b"], [0.5,"#ecf0f1"], [1,"#27ae60"]],
        zmid=0,
        zmin=-0.10,
        zmax=0.10,
        showscale=True,
        colorbar=dict(title="% Return"),
        hovertemplate="%{y} %{x}<br>Day %{text}<br>Return %{z:.2%}<extra></extra>",
        text=day_pivot.values,
        texttemplate="%{text}",
        textfont=dict(size=14, color="black")
    ))

    # Centered return text
    fig.add_trace(go.Heatmap(
        z=ret_pivot.values,
        x=ret_pivot.columns.astype(str),
        y=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        colorscale=[[0,"rgba(0,0,0,0)"],[1,"rgba(0,0,0,0)"]],
        showscale=False,
        hoverinfo="skip",
        text=(ret_pivot*100).round(1).astype(str) + "%",
        texttemplate="%{text}",
        textfont=dict(size=11, color="black", family="Arial Black")
    ))

    fig.update_layout(
        title=f"Portfolio P&L – {month_names[month-1]} {year}",
        xaxis_title="Week",
        yaxis_title="",
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
        height=520,
        margin=dict(l=20,r=20,t=60,b=20),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# Main app starts here
# ────────────────────────────────────────────────────────────────────────────

try:
    df_sheet = load_and_clean_data(SHEET_URL)
except Exception as e:
    st.error(f"Sheet error: {e}")
    st.stop()

if df_sheet.empty:
    st.warning("No holdings found.")
    st.stop()

# ... keep your sheet_fallback, prices, fx_usdinr loading ...

# Current day metrics (unchanged)
# ... your rows loop, calc_df, weights, port_day, port_total, daily_alpha_vs_spx ...

# Inception Alpha (trailing 4 years)
inception_alpha_vs_spx = None
try:
    txn = load_transactions(TRANSACTIONS_URL)
    if not txn.empty:
        ret_series = build_portfolio_daily_returns()
        if ret_series is not None and not ret_series.empty:
            cumret = (1 + ret_series).cumprod()
            cumret = cumret / cumret.iloc[0] * 100
            end_dt = cumret.index.max()
            start_4y = end_dt - pd.DateOffset(years=4)
            window = cumret[cumret.index >= start_4y]
            if len(window) >= 2:
                total_return = (window.iloc[-1] / window.iloc[0]) - 1
                # For now we show portfolio total return as proxy
                # If you want vs S&P, add spx calculation here like before
                inception_alpha_vs_spx = total_return
except Exception as e:
    st.sidebar.warning(f"Alpha calc failed: {e}")

# UI ────────────────────────────────────────────────────────────────────────
st.title("📈 Atharva Portfolio Returns")

# Status, metrics row (keep your existing code here)

# Tabs
tabs = st.tabs(["Combined", "India", "US"])

with tabs[0]:
    st.subheader("📅 Portfolio P&L Calendar (Webull Style)")
    render_webull_pnl_calendar()

    # Keep your country risk pie if you want

# Keep India/US tabs, picks table, failures, caption...

st.caption("Educational project — not investment advice.")
