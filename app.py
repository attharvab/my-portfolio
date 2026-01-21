import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, time as dtime
from zoneinfo import ZoneInfo
import plotly.express as px
import plotly.graph_objects as go
import time as _time

# ============================================================
# Atharva Portfolio Returns (Production-grade, human-first)
# Goals:
# 1) NEVER looks "broken" -> always renders something useful, even if pricing APIs hiccup
# 2) Institutional branding + clear alpha vs S&P
# 3) Better visuals: tabs, thicker portfolio line, calendar heatmap
# 4) Deep dive per ticker
# 5) Export + "Follow" (email collection form placeholder)
#
# Pricing resiliency (Triple-Threat):
# - Primary: yfinance fast_info + chunked yf.download with retries
# - Secondary: Google Sheet fallback columns (LivePrice_GS, PrevClose_GS) if Yahoo fails
# - Tertiary: per-ticker yf.download fallback
# ============================================================

# -----------------------------
# Branding / App config
# -----------------------------
APP_TITLE = "Atharva Portfolio Returns"
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📈")

# Your published holdings sheet CSV
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUHmE__8dpl_nKGv5F5mTXO7e3EyVRqz-PJF_4yyIrfJAa7z8XgzkIw6IdLnaotkACka2Q-PvP8P-z/pub?output=csv"

# Optional (Sold/Exit history) – only works if you publish a second CSV (History tab) and paste its URL here.
# If blank, app will simply hide Sold section.
HISTORY_SHEET_URL = ""  # e.g. "https://docs.google.com/spreadsheets/d/e/.../pub?output=csv"

# -----------------------------
# Config
# -----------------------------
REQUIRED_COLS = ["Ticker", "Region", "QTY", "AvgCost"]

# Add Google Finance fallback columns (optional):
# LivePrice_GS : live price from GOOGLEFINANCE
# PrevClose_GS : prev close from GOOGLEFINANCE historical pull (recommended)
OPTIONAL_COLS = ["Benchmark", "Type", "Thesis", "FirstBuyDate", "LivePrice_GS", "PrevClose_GS"]

SUMMARY_NOISE_REGEX = r"TOTAL|PORTFOLIO|SUMMARY|CASH"

DEFAULT_BENCH = {"us": "^GSPC", "india": "^NSEI"}  # benchmark per region if blank

MACRO_ASSETS = ["^GSPC", "^NSEI", "GC=F", "SI=F"]
ASSET_LABELS = {
    "^GSPC": "^GSPC (S&P 500)",
    "^NSEI": "^NSEI (Nifty 50)",
    "GC=F": "GC=F (Gold)",
    "SI=F": "SI=F (Silver)",
}
BENCH_LABEL = {"^GSPC": "S&P 500", "^NSEI": "Nifty 50", "GC=F": "Gold", "SI=F": "Silver"}

# -----------------------------
# Helpers
# -----------------------------
def _clean_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def _as_float(x):
    try:
        if pd.isna(x):
            return None
        s = str(x).strip().replace(",", "")
        if s == "":
            return None
        return float(s)
    except Exception:
        return None

def _normalize_region(r):
    r = _clean_str(r).upper()
    if r in ["US", "USA", "UNITED STATES", "UNITEDSTATES"]:
        return "US"
    if r in ["INDIA", "IN", "IND"]:
        return "India"
    return r

def _region_key(region: str) -> str:
    r = _clean_str(region).lower()
    if r == "us":
        return "us"
    if r == "india":
        return "india"
    return r

def _default_benchmark_for_region(region: str) -> str:
    return DEFAULT_BENCH.get(_region_key(region), "")

def _parse_date(x):
    try:
        if pd.isna(x) or str(x).strip() == "":
            return pd.NaT
        return pd.to_datetime(str(x).strip(), errors="coerce")
    except Exception:
        return pd.NaT

def _fmt_pct(x):
    if x is None or pd.isna(x):
        return None
    return x * 100

def _bench_context(bench: str):
    b = _clean_str(bench)
    if b == "^NSEI":
        return "vs Nifty 50 (India)"
    if b == "^GSPC":
        return "vs S&P 500 (US)"
    if b in ["GC=F", "SI=F"]:
        return f"vs {BENCH_LABEL.get(b, b)}"
    return f"vs {b}" if b else "—"

def _status_tag(alpha_day, bench):
    if alpha_day is None or pd.isna(alpha_day):
        return "—"
    short = "Nifty" if bench == "^NSEI" else ("S&P" if bench == "^GSPC" else "Index")
    return (f"🔥 Beating Market (vs {short})") if alpha_day >= 0 else (f"❄️ Lagging Market (vs {short})")

def _tooltip(label: str, help_text: str):
    # Small HTML tooltip helper (works everywhere Streamlit renders markdown)
    safe_help = help_text.replace('"', "'")
    return f"""{label} <span title="{safe_help}" style="cursor:help;">ⓘ</span>"""

# -----------------------------
# Market session logic (India + US)
# -----------------------------
def _is_market_open(now_utc: datetime, market: str) -> bool:
    if market == "US":
        tz = ZoneInfo("America/New_York")
        now = now_utc.astimezone(tz)
        if now.weekday() >= 5:
            return False
        return dtime(9, 30) <= now.time() <= dtime(16, 0)

    if market == "India":
        tz = ZoneInfo("Asia/Kolkata")
        now = now_utc.astimezone(tz)
        if now.weekday() >= 5:
            return False
        return dtime(9, 15) <= now.time() <= dtime(15, 30)

    return False

def _market_status_badge(now_utc: datetime, calc_df: pd.DataFrame):
    us_open = _is_market_open(now_utc, "US")
    in_open = _is_market_open(now_utc, "India")

    if calc_df is None or calc_df.empty or "Day_Ret" not in calc_df.columns:
        # If we cannot compute returns, still show a safe note
        if not us_open and not in_open:
            return "🟡 Markets are currently closed. Showing data from the last available trading session.", "closed"
        return "🟡 Pricing is temporarily unavailable. Showing partial content while data refreshes.", "mixed"

    near_zero = calc_df["Day_Ret"].dropna().abs() < 1e-6
    pct_zero = float(near_zero.mean()) if len(near_zero) else 0.0

    if (not us_open and not in_open) or pct_zero > 0.8:
        return "🟡 Markets are currently closed or prices are not updating. Showing data from the last trading session.", "closed"
    if in_open and not us_open:
        return "🟦 India market is open. US market is closed (US picks reflect last US session).", "mixed"
    if us_open and not in_open:
        return "🟦 US market is open. India market is closed (India picks reflect last India session).", "mixed"
    return "🟩 Markets are open.", "open"

# -----------------------------
# Load + clean + aggregate
# -----------------------------
@st.cache_data(ttl=300)
def load_and_clean_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Google Sheet: {missing}. Required: {REQUIRED_COLS}")

    for c in OPTIONAL_COLS:
        if c not in df.columns:
            df[c] = ""

    df["Ticker"] = df["Ticker"].apply(_clean_str)
    df["Region"] = df["Region"].apply(_normalize_region)
    df["Benchmark"] = df["Benchmark"].apply(_clean_str)
    df["Type"] = df["Type"].apply(_clean_str)
    df["Thesis"] = df["Thesis"].apply(_clean_str)

    df["FirstBuyDate"] = df["FirstBuyDate"].apply(_parse_date)
    df["LivePrice_GS"] = df["LivePrice_GS"].apply(_as_float)
    df["PrevClose_GS"] = df["PrevClose_GS"].apply(_as_float)

    df = df[df["Ticker"].str.len() > 0]
    df = df[~df["Ticker"].str.contains(SUMMARY_NOISE_REGEX, case=False, na=False)]

    df["QTY"] = df["QTY"].apply(_as_float)
    df["AvgCost"] = df["AvgCost"].apply(_as_float)

    df = df.dropna(subset=["Ticker", "Region", "QTY", "AvgCost"])
    df = df[df["QTY"] != 0]

    df.loc[df["Benchmark"].str.len() == 0, "Benchmark"] = df["Region"].apply(_default_benchmark_for_region)

    # TotalCost + aggregation; keep MOST RECENT FirstBuyDate per ticker for point-in-time logic
    df["TotalCost"] = df["QTY"] * df["AvgCost"]

    agg = df.groupby(["Ticker", "Region", "Benchmark"], as_index=False).agg(
        QTY=("QTY", "sum"),
        TotalCost=("TotalCost", "sum"),
        Type=("Type", "first"),
        Thesis=("Thesis", "first"),
        FirstBuyDate=("FirstBuyDate", "max"),
        LivePrice_GS=("LivePrice_GS", "max"),
        PrevClose_GS=("PrevClose_GS", "max"),
    )
    agg["AvgCost"] = agg["TotalCost"] / agg["QTY"]

    agg = agg.dropna(subset=["Ticker", "QTY", "AvgCost"])
    agg = agg[agg["QTY"] != 0]

    return agg.reset_index(drop=True)

@st.cache_data(ttl=300)
def load_sold_history(url: str) -> pd.DataFrame:
    if not url or str(url).strip() == "":
        return pd.DataFrame()
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        return pd.DataFrame()

# -----------------------------
# Pricing Engine (Triple-Threat)
# -----------------------------
def _fast_live_prev(ticker: str):
    live, prev = None, None
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None)
        if fi:
            live = fi.get("last_price", None)
            prev = fi.get("previous_close", None)
    except Exception:
        pass
    try:
        live = float(live) if live is not None else None
    except Exception:
        live = None
    try:
        prev = float(prev) if prev is not None else None
    except Exception:
        prev = None
    return live, prev

@st.cache_data(ttl=300)
def fetch_history_closes_chunked(tickers, period="15d", interval="1d", chunk_size=25, retries=2):
    tickers = sorted(list(set([_clean_str(t) for t in tickers if _clean_str(t)])))
    if not tickers:
        return pd.DataFrame()

    frames = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        last_err = None

        for attempt in range(retries + 1):
            try:
                df = yf.download(
                    tickers=chunk,
                    period=period,
                    interval=interval,
                    progress=False,
                    auto_adjust=False,
                    threads=False,     # reduce burst / throttling on Streamlit Cloud
                    group_by="column"
                )
                if df is not None and not df.empty:
                    frames.append(df)
                last_err = None
                break
            except Exception as e:
                last_err = e
                _time.sleep(0.7 * (attempt + 1))  # backoff

        # If chunk fails, we still continue; per-ticker fallback will try later
        _ = last_err

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, axis=1)

@st.cache_data(ttl=300)
def build_prices_with_sheet_fallback(tickers, sheet_fallback: dict):
    """
    sheet_fallback: dict[ticker] = {"live": LivePrice_GS, "prev": PrevClose_GS}
    returns dict[ticker] = {"live": x, "prev": y, "source": "..."}
    """
    hist = fetch_history_closes_chunked(tickers)
    price_map = {}

    def _hist_last_prev(df, tk):
        last_close, prev_close = None, None
        try:
            if df is None or df.empty:
                return None, None
            if isinstance(df.columns, pd.MultiIndex):
                if ("Close", tk) not in df.columns:
                    return None, None
                s = df[("Close", tk)].dropna()
            else:
                if "Close" not in df.columns:
                    return None, None
                s = df["Close"].dropna()

            if len(s) >= 1:
                last_close = float(s.iloc[-1])
            if len(s) >= 2:
                prev_close = float(s.iloc[-2])
        except Exception:
            return None, None
        return last_close, prev_close

    for tk in tickers:
        tk = _clean_str(tk)
        source = "none"

        # (A) Yahoo fast_info
        live_fast, prev_fast = _fast_live_prev(tk)
        if live_fast is not None or prev_fast is not None:
            source = "yfinance_fastinfo"

        # (B) Yahoo bulk history
        last_close, prev_close = _hist_last_prev(hist, tk)
        if (live_fast is None) and (last_close is not None):
            source = "yfinance_bulk"

        live = live_fast if live_fast is not None else last_close
        prev = prev_fast if prev_fast is not None else prev_close

        # (C) Per-ticker Yahoo history fallback
        if live is None or prev is None:
            try:
                df1 = yf.download(
                    tickers=tk,
                    period="15d",
                    interval="1d",
                    progress=False,
                    auto_adjust=False,
                    threads=False
                )
                if df1 is not None and not df1.empty and "Close" in df1.columns:
                    s1 = df1["Close"].dropna()
                    if len(s1) >= 1 and live is None:
                        live = float(s1.iloc[-1])
                    if len(s1) >= 2 and prev is None:
                        prev = float(s1.iloc[-2])
                    if live is not None or prev is not None:
                        source = "yfinance_single"
            except Exception:
                pass

        # (D) Google Sheet fallback (ultimate safety net)
        if (live is None) or (prev is None):
            fb = sheet_fallback.get(tk, {})
            fb_live = fb.get("live", None)
            fb_prev = fb.get("prev", None)

            if live is None and fb_live is not None:
                live = float(fb_live)
                source = "google_sheet"
            if prev is None and fb_prev is not None:
                prev = float(fb_prev)
                source = "google_sheet"

        # Last resort: avoid crashes (prev=live => day return 0)
        if prev is None and live is not None:
            prev = live
        price_map[tk] = {"live": live, "prev": prev, "source": source}

    return price_map

@st.cache_data(ttl=900)
def fetch_fx_usdinr(sheet_fallback: dict):
    # Try Yahoo first
    live, prev = _fast_live_prev("USDINR=X")
    if live is not None and live > 0:
        return live
    # Try sheet fallback if present
    fb = sheet_fallback.get("USDINR=X", {})
    if fb.get("live", None):
        try:
            return float(fb["live"])
        except Exception:
            pass
    # Try Yahoo history
    try:
        fx = yf.download("USDINR=X", period="15d", interval="1d", progress=False, auto_adjust=False, threads=False)
        s = fx["Close"].dropna()
        if not s.empty:
            v = float(s.iloc[-1])
            return v if v > 0 else 83.0
    except Exception:
        pass
    return 83.0

@st.cache_data(ttl=900)
def fetch_5y_macro():
    t = yf.download(MACRO_ASSETS, period="5y", interval="1mo", progress=False, auto_adjust=False, threads=False)["Close"]
    if isinstance(t, pd.Series):
        t = t.to_frame()
    return t.dropna(how="all").ffill()

# -----------------------------
# Portfolio Growth (Point-in-time) + Benchmark rebasing to FirstBuyDate
# FX does NOT drive returns: US converted using constant FX at start
# -----------------------------
@st.cache_data(ttl=900)
def build_portfolio_growth_index(df_holdings: pd.DataFrame):
    if "FirstBuyDate" not in df_holdings.columns:
        return None
    dff = df_holdings.dropna(subset=["FirstBuyDate"]).copy()
    if dff.empty:
        return None

    start_date = pd.to_datetime(min(dff["FirstBuyDate"])).tz_localize(None)
    tickers = sorted(dff["Ticker"].unique().tolist())
    symbols = list(set(tickers + ["USDINR=X"]))

    pxd = yf.download(
        tickers=symbols,
        start=(start_date - pd.Timedelta(days=20)).strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False
    )["Close"]

    if pxd is None or pxd.empty:
        return None

    px_m = pxd.resample("M").last().dropna(how="all")

    fx_series = px_m["USDINR=X"].dropna() if "USDINR=X" in px_m.columns else pd.Series(dtype=float)
    fx0 = float(fx_series.iloc[0]) if not fx_series.empty else 83.0

    portfolio_value = pd.Series(0.0, index=px_m.index)

    for _, r in dff.iterrows():
        tk = r["Ticker"]
        region = r["Region"]
        qty = float(r["QTY"])
        buy_date = pd.to_datetime(r["FirstBuyDate"]).tz_localize(None)

        if tk not in px_m.columns:
            continue

        s = px_m[tk].copy()
        buy_month_end = pd.to_datetime(buy_date).to_period("M").to_timestamp("M")
        s.loc[s.index < buy_month_end] = pd.NA
        s = s.ffill()

        conv = fx0 if _region_key(region) == "us" else 1.0
        portfolio_value = portfolio_value.add(qty * s * conv, fill_value=0.0)

    portfolio_value = portfolio_value[portfolio_value > 0]
    if portfolio_value.empty:
        return None

    idx = (portfolio_value / float(portfolio_value.iloc[0])) * 100.0
    idx.name = "My Portfolio (Indexed)"
    return idx

@st.cache_data(ttl=900)
def build_sp500_total_return_index(start_date: pd.Timestamp):
    """
    Total return proxy using ^GSPC price index (not true TR), re-indexed to 100 at start_date month.
    """
    if start_date is None or pd.isna(start_date):
        return None
    start_date = pd.to_datetime(start_date).tz_localize(None)
    px = yf.download("^GSPC", start=(start_date - pd.Timedelta(days=20)).strftime("%Y-%m-%d"),
                     interval="1d", progress=False, auto_adjust=False, threads=False)
    if px is None or px.empty or "Close" not in px.columns:
        return None
    m = px["Close"].resample("M").last().dropna()
    if m.empty:
        return None
    idx = (m / float(m.iloc[0])) * 100.0
    idx.name = "S&P 500 (Indexed)"
    return idx

# -----------------------------
# Calendar Heatmap (Daily Alpha vs S&P)
# -----------------------------
@st.cache_data(ttl=900)
def build_daily_alpha_heatmap_series(tickers, weights, start_date):
    """
    Returns a DataFrame with columns: date, alpha (portfolio - spx), port_ret, spx_ret
    Uses daily closes over ~6 months.
    """
    if not tickers:
        return pd.DataFrame()

    # Last ~8 months buffer
    end = datetime.now(timezone.utc).date()
    start = pd.to_datetime(start_date).date() if start_date is not None and not pd.isna(start_date) else (pd.Timestamp(end) - pd.Timedelta(days=220)).date()
    start = max(start, (pd.Timestamp(end) - pd.Timedelta(days=220)).date())

    syms = list(set(tickers + ["^GSPC"]))
    px = yf.download(
        tickers=syms,
        start=pd.to_datetime(start).strftime("%Y-%m-%d"),
        end=pd.to_datetime(end).strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False
    )["Close"]

    if px is None or px.empty:
        return pd.DataFrame()

    # Ensure DataFrame
    if isinstance(px, pd.Series):
        px = px.to_frame()

    px = px.dropna(how="all").ffill()

    # Daily returns
    rets = px.pct_change().dropna(how="all")

    # Portfolio daily return (weighted across tickers that exist)
    port = pd.Series(0.0, index=rets.index)
    for tk, w in weights.items():
        if tk in rets.columns:
            port = port.add(rets[tk] * float(w), fill_value=0.0)

    spx = rets["^GSPC"] if "^GSPC" in rets.columns else None
    if spx is None or spx.empty:
        return pd.DataFrame()

    out = pd.DataFrame({
        "Date": rets.index,
        "PortfolioRet": port.values,
        "SPXRet": spx.values,
    })
    out["Alpha"] = out["PortfolioRet"] - out["SPXRet"]
    return out.reset_index(drop=True)

def render_calendar_heatmap(alpha_df: pd.DataFrame):
    """
    Webull-style calendar heatmap approximation:
    - X axis: week of year (continuous)
    - Y axis: day of week
    - Color: Alpha (portfolio - S&P)
    """
    if alpha_df is None or alpha_df.empty:
        st.info("Heatmap unavailable right now (not enough daily data).")
        return

    df = alpha_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Dow"] = df["Date"].dt.dayofweek  # Mon=0
    df["DowName"] = df["Date"].dt.day_name().str[:3]
    # Continuous week index: year*53 + week
    iso = df["Date"].dt.isocalendar()
    df["WeekIdx"] = (iso["year"].astype(int) * 53) + iso["week"].astype(int)

    # Pivot for heatmap-like chart
    pivot = df.pivot_table(index="DowName", columns="WeekIdx", values="Alpha", aggfunc="mean")
    pivot = pivot.reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.astype(str),
        y=pivot.index,
        colorbar=dict(title="Daily Alpha"),
        hovertemplate="Day: %{y}<br>Week: %{x}<br>Alpha: %{z:.2%}<extra></extra>"
    ))
    fig.update_layout(
        title="Calendar Heatmap (Daily Alpha vs S&P 500)",
        xaxis_title="Week (rolling)",
        yaxis_title="Day",
        margin=dict(l=10, r=10, t=40, b=10),
        height=330
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Deep Dive: ratios + chart
# -----------------------------
@st.cache_data(ttl=900)
def fetch_ticker_deep_dive(ticker: str):
    tk = _clean_str(ticker)
    if not tk:
        return None
    try:
        t = yf.Ticker(tk)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        # 1Y daily
        hist = None
        try:
            hist = t.history(period="1y", interval="1d", auto_adjust=False)
        except Exception:
            hist = None

        # Extract common ratios
        out = {
            "ticker": tk,
            "name": info.get("shortName") or info.get("longName") or tk,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "marketCap": info.get("marketCap"),
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "priceToBook": info.get("priceToBook"),
            "debtToEquity": info.get("debtToEquity"),
            "profitMargins": info.get("profitMargins"),
            "operatingMargins": info.get("operatingMargins"),
            "returnOnEquity": info.get("returnOnEquity"),
            "returnOnAssets": info.get("returnOnAssets"),
            "hist": hist
        }
        return out
    except Exception:
        return None

# ============================================================
# MAIN
# ============================================================
# Sidebar: About Me + controls
st.sidebar.markdown(f"## {APP_TITLE}")
st.sidebar.markdown("**Track my strategy, benchmarked vs the S&P 500.**")
st.sidebar.markdown("---")
st.sidebar.markdown("### About Me")
st.sidebar.write(
    "I am Atharva Bhutada, an equity research-focused finance professional. "
    "This page tracks my live portfolio positioning, performance, and alpha vs benchmarks."
)
st.sidebar.markdown("**Contact:**")
st.sidebar.write("LinkedIn: linkedin.com/in/atharva-bhutada")
st.sidebar.write("Email: abhutada1@babson.edu")

st.sidebar.markdown("---")
st.sidebar.markdown("### Follow Updates")
with st.sidebar.form("follow_form", clear_on_submit=True):
    email = st.text_input("Enter your email to follow updates", placeholder="name@email.com")
    submitted = st.form_submit_button("Follow")
    if submitted:
        # Placeholder: connect to Zapier / Make / SMTP later
        if email and "@" in email:
            st.success("Saved. (Connect this to Zapier/Make/SMTP to send notifications.)")
        else:
            st.warning("Enter a valid email.")

# Load sheet
try:
    df_sheet = load_and_clean_data(SHEET_URL)
except Exception as e:
    st.error(f"Spreadsheet Error: {e}")
    st.stop()

if df_sheet.empty:
    st.warning("No valid holdings found. Check Ticker/Region/QTY/AvgCost.")
    st.stop()

# Build Google Sheet fallback dict
sheet_fallback = {}
for _, r in df_sheet.iterrows():
    tk = _clean_str(r.get("Ticker", ""))
    sheet_fallback[tk] = {
        "live": r.get("LivePrice_GS", None),
        "prev": r.get("PrevClose_GS", None),
    }

# If you ever add USDINR=X row in sheet for fallback, it will be used:
# sheet_fallback["USDINR=X"] = {"live": ..., "prev": ...}

holdings = df_sheet["Ticker"].unique().tolist()
benchmarks = df_sheet["Benchmark"].unique().tolist()

all_symbols = list(set(holdings + benchmarks + ["USDINR=X"] + MACRO_ASSETS))

with st.spinner("Syncing portfolio data..."):
    prices = build_prices_with_sheet_fallback(all_symbols, sheet_fallback)
    fx_usdinr = fetch_fx_usdinr(sheet_fallback)
    macro = fetch_5y_macro()
    portfolio_idx = build_portfolio_growth_index(df_sheet)

# Compute holding rows (partial rendering allowed)
rows, failures = [], []

for _, r in df_sheet.iterrows():
    tk = _clean_str(r["Ticker"])
    bench = _clean_str(r["Benchmark"])
    region = r["Region"]

    p_tk = prices.get(tk, None)
    p_b = prices.get(bench, None) if bench else None

    if not p_tk or p_tk.get("live", None) is None or p_tk.get("prev", None) is None:
        failures.append({"Ticker": tk, "Reason": "Missing holding price (Yahoo+Sheet failed)"})
        continue

    if bench and (not p_b or p_b.get("live", None) is None or p_b.get("prev", None) is None):
        failures.append({"Ticker": tk, "Reason": f"Missing benchmark price ({bench})"})
        continue

    live = float(p_tk["live"])
    prev = float(p_tk["prev"])
    qty = float(r["QTY"])
    avg = float(r["AvgCost"])

    total_ret = (live - avg) / avg if avg != 0 else None
    day_ret = (live - prev) / prev if prev != 0 else None

    alpha_day = None
    if bench:
        b_live = float(p_b["live"])
        b_prev = float(p_b["prev"])
        b_day = (b_live - b_prev) / b_prev if b_prev != 0 else None
        if day_ret is not None and b_day is not None:
            alpha_day = day_ret - b_day

    # INR value for exposure weights only (Country Risk)
    value_inr = qty * live * (fx_usdinr if _region_key(region) == "us" else 1.0)

    rows.append({
        "Ticker": tk,
        "Region": region,
        "Benchmark": bench,
        "Compared To": _bench_context(bench),
        "QTY": qty,
        "AvgCost": avg,
        "LivePrice": live,
        "PrevClose": prev,
        "Value_INR": value_inr,
        "Total_Ret": total_ret,
        "Day_Ret": day_ret,
        "Alpha_Day": alpha_day,
        "Beat_Index_Tag": _status_tag(alpha_day, bench),
        "Type": r.get("Type", ""),
        "Thesis": r.get("Thesis", ""),
        "FirstBuyDate": r.get("FirstBuyDate", pd.NaT),
        "PriceSource": p_tk.get("source", "unknown"),
    })

calc_df = pd.DataFrame(rows)

# Header branding
st.title("📈 Atharva Portfolio Returns")
st.markdown(
    "This dashboard tracks my portfolio performance and **alpha vs the S&P 500**, with India positions benchmarked to Nifty at the stock level."
)
st.caption("Returns are shown in native currency. FX (USD/INR) is used only for exposure weights, not for returns.")

# Market status (never looks dead)
now_utc = datetime.now(timezone.utc)
status_text, status_type = _market_status_badge(now_utc, calc_df)
if status_type == "closed":
    st.info(status_text)
elif status_type == "mixed":
    st.warning(status_text)
else:
    st.success(status_text)

# If pricing is totally down, do not hard-fail the website
if calc_df.empty:
    st.error("Pricing is temporarily unavailable. This usually happens due to rate limits or API hiccups.")
    st.caption("Your site is still live. Try refreshing in 1–2 minutes. Meanwhile, the macro chart and About section remain visible.")
    if failures:
        st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)
    st.stop()

# Clean weights and ensure QTY>0 (already filtered upstream, but keep bulletproof)
calc_df = calc_df[calc_df["QTY"] > 0].copy()
calc_df["Weight"] = calc_df["Value_INR"] / calc_df["Value_INR"].sum()

# Portfolio metrics
port_day = (calc_df["Day_Ret"] * calc_df["Weight"]).sum()
port_total = (calc_df["Total_Ret"] * calc_df["Weight"]).sum()

# Country weights
in_w = calc_df.loc[calc_df["Region"].str.upper() == "INDIA", "Weight"].sum()
us_w = calc_df.loc[calc_df["Region"].str.upper() == "US", "Weight"].sum()

def _safe_day(sym):
    p = prices.get(sym, None)
    if not p or p.get("live", None) is None or p.get("prev", None) is None or float(p["prev"]) == 0:
        return None
    return (float(p["live"]) - float(p["prev"])) / float(p["prev"])

spx_day = _safe_day("^GSPC")
nifty_day = _safe_day("^NSEI")

# Phase 1: Dual Alpha Tracking (vs S&P)
daily_alpha_vs_spx = (port_day - spx_day) if (spx_day is not None) else None

# Inception alpha vs S&P since start date
strategy_start = None
if "FirstBuyDate" in df_sheet.columns and df_sheet["FirstBuyDate"].notna().any():
    strategy_start = pd.to_datetime(df_sheet["FirstBuyDate"].dropna().min()).tz_localize(None)

spx_inception_idx = build_sp500_total_return_index(strategy_start) if strategy_start is not None else None

inception_alpha_vs_spx = None
if portfolio_idx is not None and spx_inception_idx is not None:
    # Align to common month-end index
    p = portfolio_idx.copy()
    p.index = pd.to_datetime(p.index).to_period("M").to_timestamp("M")
    s = spx_inception_idx.copy()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
    m = pd.concat([p, s], axis=1).dropna()
    if not m.empty:
        strat_total = (float(m.iloc[-1, 0]) / float(m.iloc[0, 0])) - 1.0
        spx_total = (float(m.iloc[-1, 1]) / float(m.iloc[0, 1])) - 1.0
        inception_alpha_vs_spx = strat_total - spx_total

# -----------------------------
# Phase 1 metrics (with glossary tooltips)
# -----------------------------
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(_tooltip("**Total Return (Strategy)**", "Absolute return since inception, based on your AvgCost (native currency), weighted by current exposure."), unsafe_allow_html=True)
    st.metric(label="", value=f"{port_total*100:.2f}%")
    st.caption("Absolute return since inception.")

with m2:
    st.markdown(_tooltip("**Today Return (Portfolio)**", "Weighted return of your holdings today (based on PrevClose vs LivePrice)."), unsafe_allow_html=True)
    st.metric(label="", value=f"{port_day*100:.2f}%")

with m3:
    st.markdown(_tooltip("**Daily Alpha (vs S&P)**", "Portfolio Today Return minus S&P 500 Today Return. Positive means outperforming S&P today."), unsafe_allow_html=True)
    if daily_alpha_vs_spx is None:
        st.metric(label="", value="—")
        st.caption("S&P daily move unavailable.")
    else:
        st.metric(label="", value=f"{daily_alpha_vs_spx*100:.2f}%")

with m4:
    st.markdown(_tooltip("**Inception Alpha (vs S&P)**", "Total Strategy Return since start date minus S&P 500 return over the same period (rebased to your FirstBuyDate)."), unsafe_allow_html=True)
    if inception_alpha_vs_spx is None:
        st.metric(label="", value="—")
        st.caption("Add FirstBuyDate to unlock.")
    else:
        st.metric(label="", value=f"{inception_alpha_vs_spx*100:.2f}%")

st.caption(_tooltip("Last Sync (UTC)", "Data may be delayed. Pricing uses Yahoo Finance with a Google Sheet fallback when available."), unsafe_allow_html=True)
st.write(now_utc.strftime("%Y-%m-%d %H:%M"))

# ------------------------------------------------------------
# Phase 2: Tabs + responsive multi-graphs
# ------------------------------------------------------------
st.divider()
tabs = st.tabs(["Combined", "India", "US"])

def _filter_region(df, region_name):
    return df[df["Region"].str.upper() == region_name.upper()].copy()

def _plot_region_macro(macro_df, portfolio_series=None, baseline_date=None, title="Macro vs Strategy (Indexed)"):
    if macro_df is None or macro_df.empty:
        st.info("Macro trend data unavailable right now.")
        return

    macro_named = macro_df.copy().rename(columns={k: v for k, v in ASSET_LABELS.items() if k in macro_df.columns})

    plot_df = macro_named.copy()

    if portfolio_series is not None and not portfolio_series.empty:
        p = portfolio_series.copy()
        p.index = pd.to_datetime(p.index).to_period("M").to_timestamp("M")
        plot_df = plot_df.merge(p.to_frame(), left_index=True, right_index=True, how="left")

    plot_df = plot_df.dropna(how="all").ffill()

    # Rebase to strategy baseline month-end
    if baseline_date is not None:
        plot_df = plot_df[plot_df.index >= baseline_date].copy()
        if plot_df.empty:
            plot_df = macro_named.dropna(how="all").ffill().copy()
            if portfolio_series is not None and not portfolio_series.empty:
                plot_df = plot_df.merge(p.to_frame(), left_index=True, right_index=True, how="left")
            plot_df = plot_df.dropna(how="all").ffill()

    base_row = plot_df.iloc[0]
    plot_idx = (plot_df / base_row) * 100.0

    fig = go.Figure()
    for col in plot_idx.columns:
        width = 4 if "My Portfolio" in col else 2
        fig.add_trace(go.Scatter(
            x=plot_idx.index,
            y=plot_idx[col],
            mode="lines",
            name=col,
            line=dict(width=width),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Index (Base=100 at Strategy Start)",
        legend_title="Series",
        margin=dict(l=10, r=10, t=40, b=10),
        height=380
    )
    st.plotly_chart(fig, use_container_width=True)

# Baseline month-end for indexing
baseline_date = None
if strategy_start is not None:
    baseline_date = strategy_start.to_period("M").to_timestamp("M")

with tabs[0]:
    # Combined view: macro + country risk + heatmap
    left, right = st.columns([2, 1])

    with left:
        st.subheader("📈 Strategy vs Benchmarks (Indexed to 100 at Start)")
        _plot_region_macro(
            macro_df=macro,
            portfolio_series=portfolio_idx,
            baseline_date=baseline_date,
            title="Strategy vs Macro Benchmarks (Indexed)"
        )

    with right:
        st.subheader("🌍 Country Risk (Live Exposure)")
        alloc = calc_df.groupby("Region", as_index=False)["Weight"].sum()
        fig_pie = px.pie(alloc, values="Weight", names="Region", hole=0.5)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.write(f"**India exposure:** {in_w*100:.2f}%")
        st.write(f"**US exposure:** {us_w*100:.2f}%")

    st.divider()
    st.subheader("🟩🟥 Calendar Heatmap")
    st.caption("Daily alpha squares show how my portfolio performed **vs S&P 500 each day** over recent months.")
    weights_map = {row["Ticker"]: row["Weight"] for _, row in calc_df.iterrows()}
    alpha_daily = build_daily_alpha_heatmap_series(
        tickers=calc_df["Ticker"].unique().tolist(),
        weights=weights_map,
        start_date=strategy_start if strategy_start is not None else None
    )
    render_calendar_heatmap(alpha_daily)

with tabs[1]:
    st.subheader("🇮🇳 India View")
    india_df = _filter_region(calc_df, "INDIA")
    if india_df.empty:
        st.info("No India holdings right now.")
    else:
        st.write(f"Holdings: {len(india_df)} | Live exposure weight: {india_df['Weight'].sum()*100:.2f}%")
        # For India tab, show macro + Nifty + portfolio line still combined (portfolio is whole strategy)
        _plot_region_macro(macro_df=macro, portfolio_series=portfolio_idx, baseline_date=baseline_date,
                           title="India Context: Nifty vs Strategy (Indexed)")

with tabs[2]:
    st.subheader("🇺🇸 US View")
    us_df = _filter_region(calc_df, "US")
    if us_df.empty:
        st.info("No US holdings right now.")
    else:
        st.write(f"Holdings: {len(us_df)} | Live exposure weight: {us_df['Weight'].sum()*100:.2f}%")
        _plot_region_macro(macro_df=macro, portfolio_series=portfolio_idx, baseline_date=baseline_date,
                           title="US Context: S&P vs Strategy (Indexed)")

# ------------------------------------------------------------
# Picks table + Export
# ------------------------------------------------------------
st.divider()
st.subheader("📌 Picks (Did each stock beat its index today?)")

show = calc_df.copy()
show["Weight%"] = show["Weight"] * 100
show["Total Ret%"] = show["Total_Ret"].apply(_fmt_pct)
show["Day Ret%"] = show["Day_Ret"].apply(_fmt_pct)
show["Score vs Index%"] = show["Alpha_Day"].apply(_fmt_pct)

show["Benchmark"] = show["Benchmark"].replace({
    "^GSPC": "^GSPC (S&P 500)",
    "^NSEI": "^NSEI (Nifty 50)",
    "GC=F": "GC=F (Gold)",
    "SI=F": "SI=F (Silver)"
})

# Mobile friendliness: keep PrevClose out of table by default
st.dataframe(
    show[[
        "Ticker", "Region", "Benchmark", "Compared To",
        "Weight%", "AvgCost", "LivePrice",
        "Total Ret%", "Day Ret%", "Beat_Index_Tag", "Score vs Index%", "PriceSource"
    ]],
    column_config={
        "Weight%": st.column_config.NumberColumn("Weight", format="%.2f%%"),
        "Total Ret%": st.column_config.NumberColumn("Total Return", format="%.2f%%"),
        "Day Ret%": st.column_config.NumberColumn("Today Return", format="%.2f%%"),
        "Beat_Index_Tag": st.column_config.TextColumn("Beat Index?"),
        "Score vs Index%": st.column_config.NumberColumn("Score vs Index", format="%.2f%%"),
        "AvgCost": st.column_config.NumberColumn("Avg Cost", format="%.2f"),
        "LivePrice": st.column_config.NumberColumn("Live Price", format="%.2f"),
        "Compared To": st.column_config.TextColumn("Compared To"),
        "PriceSource": st.column_config.TextColumn("Price Source"),
    },
    use_container_width=True,
    hide_index=True
)

# Export button
csv_bytes = show[[
    "Ticker", "Region", "Benchmark", "Compared To",
    "Weight%", "AvgCost", "LivePrice",
    "Total Ret%", "Day Ret%", "Beat_Index_Tag", "Score vs Index%"
]].to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Picks as CSV",
    data=csv_bytes,
    file_name="atharva_portfolio_picks.csv",
    mime="text/csv"
)

# ------------------------------------------------------------
# Deep Dive (per ticker)
# ------------------------------------------------------------
st.divider()
st.subheader("🔎 Deep Dive (Stock History & Key Ratios)")

tickers_sorted = sorted(show["Ticker"].unique().tolist())
selected = st.selectbox("Select a ticker", tickers_sorted)

deep = fetch_ticker_deep_dive(selected)
if deep is None:
    st.info("Deep dive unavailable right now.")
else:
    cA, cB = st.columns([2, 1])

    with cA:
        hist = deep.get("hist", None)
        if hist is not None and not hist.empty and "Close" in hist.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist["Close"],
                mode="lines",
                name="Close"
            ))
            fig.update_layout(
                title=f"{deep.get('name', selected)} - 1Y Price History",
                xaxis_title="Date",
                yaxis_title="Price",
                margin=dict(l=10, r=10, t=40, b=10),
                height=320
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Price history is unavailable right now for this ticker.")

    with cB:
        st.markdown(f"**{deep.get('name', selected)}**")
        st.write(f"**Ticker:** {deep.get('ticker')}")
        if deep.get("sector"):
            st.write(f"**Sector:** {deep.get('sector')}")
        if deep.get("industry"):
            st.write(f"**Industry:** {deep.get('industry')}")

        st.markdown("---")
        st.markdown(_tooltip("**Key Ratios**", "Pulled via Yahoo Finance. Some tickers may not provide all fields."), unsafe_allow_html=True)
        st.write(f"P/E (TTM): {deep.get('trailingPE')}")
        st.write(f"P/E (Fwd): {deep.get('forwardPE')}")
        st.write(f"Debt/Equity: {deep.get('debtToEquity')}")
        st.write(f"Price/Book: {deep.get('priceToBook')}")
        st.write(f"ROE: {deep.get('returnOnEquity')}")
        st.write(f"Margins: {deep.get('profitMargins')}")

with st.expander("🧠 Thesis / Notes (from Google Sheet)"):
    pick = show[show["Ticker"] == selected]
    if not pick.empty:
        r = pick.iloc[0]
        st.write(f"**Type:** {r.get('Type','')}")
        st.write(f"**Region:** {r.get('Region','')}")
        st.write(f"**Benchmark:** {r.get('Benchmark','')}")
        st.text_area("Thesis (edit in Google Sheet)", value=str(r.get("Thesis","")), height=180)

# ------------------------------------------------------------
# Sold / Exit History (if provided)
# ------------------------------------------------------------
sold_df = load_sold_history(HISTORY_SHEET_URL)
if sold_df is not None and not sold_df.empty:
    st.divider()
    st.subheader("🧾 Sold Positions (History & Exit Thesis)")
    st.caption("This section pulls from a separate published Google Sheet (History tab).")
    st.dataframe(sold_df, use_container_width=True, hide_index=True)
else:
    st.caption("Tip: If you want a Sold Positions table, publish a second CSV for your History tab and paste its URL into HISTORY_SHEET_URL.")

# ------------------------------------------------------------
# Failures (diagnostic, but not scary)
# ------------------------------------------------------------
if failures:
    st.divider()
    st.warning("Some symbols could not be priced (dashboard still runs on partial data).")
    st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)

st.divider()
st.caption("Data source: Yahoo Finance (with Google Sheet fallback if you provide LivePrice_GS/PrevClose_GS). Educational project, not investment advice.")
