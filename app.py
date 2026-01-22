# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
import plotly.graph_objects as go

APP_TITLE = "Atharva Portfolio Returns"
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📈")

# ── SHEET URLS ───────────────────────────────────────────────────────────────
SHEET_URL       = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUHmE__8dpl_nKGv5F5mTXO7e3EyVRqz-PJF_4yyIrfJAa7z8XgzkIw6IdLnaotkACka2Q-PvP8P-z/pub?output=csv"
TRANSACTIONS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR-OybDEJRMpK5jvtLnMq3SOze-ZwT6hVY07w4nAnKfn1dva_E68fKSZQkn0yvzDhk217HEQ7xis77G/pub?output=csv"

# ── CONFIG ───────────────────────────────────────────────────────────────────
REQUIRED_COLS = ["Ticker", "Region", "QTY", "AvgCost"]
OPTIONAL_COLS = ["Benchmark", "Type", "Thesis", "FirstBuyDate", "LivePrice_GS", "PrevClose_GS"]
SUMMARY_NOISE = r"TOTAL|PORTFOLIO|SUMMARY|CASH"
DEFAULT_BENCH = {"us": "^GSPC", "india": "^NSEI"}

# ── HELPERS ──────────────────────────────────────────────────────────────────
def clean_str(x):    return str(x).strip() if not pd.isna(x) else ""
def as_float(x): 
    try: return float(str(x).strip().replace(",","")) if not pd.isna(x) else None
    except: return None
def normalize_region(r):
    r = clean_str(r).upper()
    if r in ["US","USA","UNITED STATES","UNITEDSTATES"]: return "US"
    if r in ["INDIA","IN","IND"]: return "India"
    return r
def is_india_ticker(tk): return clean_str(tk).upper().endswith((".NS",".BO")) or tk.upper() in ["^NSEI"]
def is_us_ticker(tk):    return not is_india_ticker(tk)

def fmt_pct(x): return f"{x*100:.2f}%" if x is not None else "—"

# ── LOAD HOLDINGS ────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_holdings():
    df = pd.read_csv(SHEET_URL)
    df.columns = df.columns.str.strip()
    for c in REQUIRED_COLS:
        if c not in df.columns: raise ValueError(f"Missing column: {c}")
    for c in OPTIONAL_COLS:
        if c not in df.columns: df[c] = ""
    df["Ticker"] = df["Ticker"].apply(clean_str)
    df["Region"] = df["Region"].apply(normalize_region)
    df = df[df["Ticker"].str.len() > 0]
    df = df[~df["Ticker"].str.contains(SUMMARY_NOISE, case=False, na=False)]
    df["QTY"] = df["QTY"].apply(as_float)
    df["AvgCost"] = df["AvgCost"].apply(as_float)
    df = df.dropna(subset=["Ticker","Region","QTY","AvgCost"])
    df = df[df["QTY"] > 0]
    df["Benchmark"] = df.apply(lambda row: DEFAULT_BENCH.get(row["Region"].lower(), "") if not row["Benchmark"] else row["Benchmark"], axis=1)
    return df

# ── LOAD TRANSACTIONS ────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_transactions():
    df = pd.read_csv(TRANSACTIONS_URL)
    df.columns = df.columns.astype(str).str.strip()
    keep = [c for c in df.columns if not c.startswith("Unnamed")]
    df = df[keep].copy()
    req = ["Ticker", "Date", "QTY"]
    for c in req:
        if c not in df.columns: return pd.DataFrame()
    n = len(df)
    out = pd.DataFrame(index=range(n))
    # Date parsing (most reliable format first)
    out["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), format="%d-%b-%Y", errors="coerce")
    if out["Date"].isna().all():
        out["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    out["Ticker"] = df["Ticker"].apply(clean_str)
    out["Qty"]   = df["QTY"].apply(as_float)
    out["Region"] = df.get("Region", "").apply(normalize_region)
    out["FX_Rate"] = df.get("FX_Rate", None).apply(as_float)
    out = out.dropna(subset=["Date","Ticker","Qty"])
    out = out[out["Ticker"].str.len() > 0].sort_values("Date").reset_index(drop=True)
    # Infer missing region
    mask = out["Region"] == ""
    if mask.any():
        out.loc[mask, "Region"] = out.loc[mask, "Ticker"].apply(lambda t: "India" if is_india_ticker(t) else "US")
    return out

# ── PORTFOLIO DAILY RETURNS (for calendar & alpha) ───────────────────────────
@st.cache_data(ttl=1800)
def get_portfolio_daily_returns():
    txn = load_transactions()
    if txn.empty: return None
    start = txn["Date"].min().normalize()
    end   = pd.Timestamp.utcnow().normalize()
    tickers = sorted(txn["Ticker"].unique())
    if not tickers: return None
    px = yf.download(tickers, start=start-pd.Timedelta(10,"d"), end=end+pd.Timedelta(1,"d"),
                     interval="1d", progress=False, threads=False)["Close"]
    if px.empty: return None
    if isinstance(px, pd.Series): px = px.to_frame(name=tickers[0])
    # Fill missing
    for tk in [t for t in tickers if t not in px.columns]:
        try:
            one = yf.download(tk, start=px.index.min(), end=px.index.max()+pd.Timedelta(1,"d"), progress=False)
            if "Close" in one: px[tk] = one["Close"]
        except: pass
    px = px.ffill().dropna(how="all")
    fx = yf.download("USDINR=X", start=px.index.min(), end=px.index.max()+pd.Timedelta(1,"d"),
                     progress=False)["Close"].reindex(px.index).ffill()
    pos = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    deltas = txn.groupby([pd.to_datetime(txn["Date"]).dt.normalize(), "Ticker"])["Qty"].sum()
    for tk in pos.columns:
        pos[tk] = deltas.get(tk, pd.Series(0,index=pos.index)).reindex(pos.index).fillna(0).cumsum()
    pos = pos.clip(lower=0)
    px_inr = px.copy()
    if not fx.empty:
        fx_a = fx.reindex(px.index).ffill()
        for tk in px_inr.columns:
            if is_us_ticker(tk) and tk not in ["GC=F","SI=F"]:
                px_inr[tk] *= fx_a
    value = (pos * px_inr).sum(1).replace([float("inf"),-float("inf")],pd.NA).dropna()
    value = value[value > 0]
    if len(value) < 2: return None
    return value.pct_change().dropna()

# ── WEBULL-STYLE P&L CALENDAR ───────────────────────────────────────────────
def render_pnl_calendar():
    rets = get_portfolio_daily_returns()
    if rets is None or rets.empty:
        st.info("P&L calendar unavailable — insufficient data.")
        return

    years = sorted(rets.index.year.unique())
    year = st.selectbox("Year", years, index=len(years)-1)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month = st.selectbox("Month", range(1,13), format_func=lambda m: months[m-1], index=datetime.now().month-1)

    month_rets = rets[(rets.index.year == year) & (rets.index.month == month)]
    if month_rets.empty:
        st.info(f"No data for {months[month-1]} {year}")
        return

    df = pd.DataFrame({"date": month_rets.index, "ret": month_rets})
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday   # 0=Mon … 6=Sun
    df["week"] = df["date"].dt.isocalendar().week

    ret_grid = df.pivot_table(index="weekday", columns="week", values="ret", aggfunc="first").reindex(range(7))
    day_grid = df.pivot_table(index="weekday", columns="week", values="day",   aggfunc="first").reindex(range(7))

    fig = go.Figure()

    # Background heatmap (color)
    fig.add_trace(go.Heatmap(
        z=ret_grid.values,
        x=ret_grid.columns.astype(str),
        y=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        colorscale=[[0.0, "#c0392b"], [0.5, "#ffffff"], [1.0, "#27ae60"]],
        zmid=0,
        zmin=-0.08,
        zmax=0.08,
        showscale=True,
        colorbar=dict(title="% Return", len=0.6),
        hovertemplate="%{y} week %{x}<br>Day %{text}<br>Return %{z:.2%}<extra></extra>",
        text=day_grid.values,
        texttemplate="%{text}",
        textfont=dict(size=13, color="#222")
    ))

    # Centered % return text
    fig.add_trace(go.Heatmap(
        z=ret_grid.values,
        x=ret_grid.columns.astype(str),
        y=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        colorscale=[[0,"rgba(0,0,0,0)"],[1,"rgba(0,0,0,0)"]],
        showscale=False,
        hoverinfo="skip",
        text=(ret_grid*100).round(1).astype(str) + "%",
        texttemplate="%{text}",
        textfont=dict(size=11, color="#000", family="Arial Black")
    ))

    fig.update_layout(
        title=f"P&L Calendar – {months[month-1]} {year}",
        xaxis_title="Week",
        yaxis_title="",
        xaxis=dict(side="top", tickmode="array"),
        yaxis=dict(autorange="reversed"),
        height=520,
        margin=dict(l=20,r=20,t=60,b=20),
        plot_bgcolor="#fff",
        paper_bgcolor="#fff"
    )
    fig.update_xaxes(scaleanchor="y", scaleratio=1, constrain="domain")
    fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain")

    st.plotly_chart(fig, use_container_width=True)

# ── MAIN APP ─────────────────────────────────────────────────────────────────
try:
    holdings = load_holdings()
except Exception as e:
    st.error(f"Could not load holdings: {e}")
    st.stop()

if holdings.empty:
    st.warning("No valid holdings found.")
    st.stop()

# Current prices & metrics (simplified — add your full pricing logic here)
# For brevity — replace with your full prices / calc_df logic
st.title("📈 Atharva Portfolio Returns")
st.caption("Educational dashboard — not investment advice")

# Placeholder metrics (replace with your real calculation)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Return", "—")
col2.metric("Today Return", "—")
col3.metric("Daily Alpha vs S&P", "—")
with col4:
    inception_alpha = 0.00  # ← real value comes from below
    st.metric("Inception Alpha (4Y)", f"{inception_alpha:+.2f}%")

# P&L Calendar
st.divider()
st.subheader("📅 Portfolio P&L Calendar")
render_pnl_calendar()

st.divider()
st.caption("Last sync: " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
