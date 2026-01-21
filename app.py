# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, time as dtime
from zoneinfo import ZoneInfo
import plotly.express as px
import plotly.graph_objects as go
import time as _time

# ============================================================
# Atharva Portfolio Returns (Privacy-first, resilient)
#
# FIXES INCLUDED:
# ✅ Crash fix: _plot_indexed_strategy no longer calls .to_frame() on DataFrames
# ✅ Alpha fix: Inception equity curve is built in ONE currency (INR)
#    - India tickers valued in INR
#    - US tickers valued in USD * USDINR daily series (FX normalized per day)
# ✅ No summing daily values (uses daily market value only, then rebases to Base=100)
# ✅ No absolute ₹ values shown anywhere (only Base=100 indices and weights/returns)
# ============================================================

APP_TITLE = "Atharva Portfolio Returns"
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📈")

# ========= YOUR SHEET URLS =========
# Holdings (Current portfolio snapshot)
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUHmE__8dpl_nKGv5F5mTXO7e3EyVRqz-PJF_4yyIrfJAa7z8XgzkIw6IdLnaotkACka2Q-PvP8P-z/pub?output=csv"

# Transactions (Ledger) - CSV published
TRANSACTIONS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR-OybDEJRMpK5jvtLnMq3SOze-ZwT6hVY07w4nAnKfn1dva_E68fKSZQkn0yvzDhk217HEQ7xis77G/pub?output=csv"

# =============================
# Columns & config
# =============================
REQUIRED_COLS = ["Ticker", "Region", "QTY", "AvgCost"]
OPTIONAL_COLS = ["Benchmark", "Type", "Thesis", "FirstBuyDate", "LivePrice_GS", "PrevClose_GS"]

SUMMARY_NOISE_REGEX = r"TOTAL|PORTFOLIO|SUMMARY|CASH"
DEFAULT_BENCH = {"us": "^GSPC", "india": "^NSEI"}

MACRO_ASSETS = ["^GSPC", "^NSEI", "GC=F", "SI=F"]
ASSET_LABELS = {
    "^GSPC": "^GSPC (S&P 500)",
    "^NSEI": "^NSEI (Nifty 50)",
    "GC=F": "GC=F (Gold)",
    "SI=F": "SI=F (Silver)",
}
BENCH_LABEL = {"^GSPC": "S&P 500", "^NSEI": "Nifty 50", "GC=F": "Gold", "SI=F": "Silver"}

# ============================================================
# Helpers
# ============================================================
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
    safe_help = help_text.replace('"', "'")
    return f"""{label} <span title="{safe_help}" style="cursor:default;">ⓘ</span>"""

def _ensure_series(obj, preferred_col=None):
    """Convert Series/DataFrame into a Series safely."""
    if obj is None:
        return None
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame):
        if preferred_col and preferred_col in obj.columns:
            return obj[preferred_col]
        if obj.shape[1] == 1:
            return obj.iloc[:, 0]
        return obj.iloc[:, 0]
    return None

# ============================================================
# Market session logic
# ============================================================
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

# ============================================================
# Load + clean (Holdings)
# ============================================================
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

# ============================================================
# Load transactions ledger (for equity curve)
# Expected columns (flexible names):
# Date, Ticker, Qty (positive buy, negative sell), Region(optional), FX_Rate(optional)
# ============================================================
@st.cache_data(ttl=300)
def load_transactions(url: str) -> pd.DataFrame:
    if not url or str(url).strip() == "":
        return pd.DataFrame()
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()

    col_map = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n.lower() in col_map:
                return col_map[n.lower()]
        return None

    c_date = pick("Date", "TradeDate", "TxnDate", "TransactionDate")
    c_tkr = pick("Ticker", "Symbol")
    c_qty = pick("QTY", "Qty", "Quantity", "Shares")
    c_region = pick("Region", "Market")
    c_fx = pick("FX_Rate", "FX", "FxRate", "USDINR", "Fx")

    if c_date is None or c_tkr is None or c_qty is None:
        raise ValueError("Transactions sheet must have at least Date, Ticker/Symbol, and Qty columns.")

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df[c_date], errors="coerce")
    out["Ticker"] = df[c_tkr].apply(_clean_str)
    out["Qty"] = df[c_qty].apply(_as_float)
    out["Region"] = df[c_region].apply(_normalize_region) if c_region else ""
    out["FX_Rate"] = df[c_fx].apply(_as_float) if c_fx else None

    out = out.dropna(subset=["Date", "Ticker", "Qty"])
    out = out[out["Ticker"].str.len() > 0]
    out = out.sort_values("Date").reset_index(drop=True)

    # Infer region if missing
    def infer_region(tk):
        t = _clean_str(tk).upper()
        if t.endswith(".NS") or t.endswith(".BO") or t in ["^NSEI"]:
            return "India"
        return "US"

    out.loc[out["Region"].astype(str).str.strip() == "", "Region"] = out["Ticker"].apply(infer_region)

    return out

# ============================================================
# Pricing Engine (Yahoo primary + Sheet fallback ALWAYS)
# ============================================================
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
        for attempt in range(retries + 1):
            try:
                df = yf.download(
                    tickers=chunk,
                    period=period,
                    interval=interval,
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                    group_by="column"
                )
                if df is not None and not df.empty:
                    frames.append(df)
                break
            except Exception:
                _time.sleep(0.7 * (attempt + 1))
                continue

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)

@st.cache_data(ttl=300)
def build_prices_with_sheet_fallback(tickers, sheet_fallback: dict):
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

        live_fast, prev_fast = _fast_live_prev(tk)
        if live_fast is not None or prev_fast is not None:
            source = "yfinance_fastinfo"

        last_close, prev_close = _hist_last_prev(hist, tk)
        live = live_fast if live_fast is not None else last_close
        prev = prev_fast if prev_fast is not None else prev_close
        if source == "none" and (live is not None or prev is not None):
            source = "yfinance_bulk"

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

        # Sheet fallback ALWAYS
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

        if prev is None and live is not None:
            prev = live

        price_map[tk] = {"live": live, "prev": prev, "source": source}

    return price_map

@st.cache_data(ttl=900)
def fetch_5y_macro():
    t = yf.download(MACRO_ASSETS, period="5y", interval="1mo", progress=False, auto_adjust=False, threads=False)["Close"]
    if isinstance(t, pd.Series):
        t = t.to_frame()
    return t.dropna(how="all").ffill()

# ============================================================
# PRIVACY-FIRST Inception Equity Curve (Base=100, INR-normalized)
# - Builds daily shares from transactions
# - Values India positions in INR
# - Values US positions in USD * USDINR daily series
# ============================================================
@st.cache_data(ttl=1800)
def build_equity_curve_index_from_ledger(txn: pd.DataFrame):
    if txn is None or txn.empty:
        return None

    start = pd.to_datetime(txn["Date"].min()).normalize()
    end = pd.Timestamp.utcnow().normalize()

    tickers = sorted(txn["Ticker"].unique().tolist())

    # Download close prices for tickers (daily)
    px = yf.download(
        tickers=tickers,
        start=(start - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False
    )["Close"]

    if px is None or (isinstance(px, pd.DataFrame) and px.empty):
        return None
    if isinstance(px, pd.Series):
        px = px.to_frame()

    px = px.dropna(how="all").ffill()

    # FX series (USDINR daily close)
    fx = yf.download(
        "USDINR=X",
        start=(start - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False
    )
    fx = fx["Close"] if fx is not None and not fx.empty and "Close" in fx.columns else None
    if fx is None or fx.dropna().empty:
        # fallback constant (safe)
        fx = pd.Series(83.0, index=px.index)
    fx = fx.reindex(px.index).ffill().bfill()

    # Daily positions (shares) from ledger
    days = px.index
    pos = pd.DataFrame(0.0, index=days, columns=tickers)

    d = txn.copy()
    d["Day"] = pd.to_datetime(d["Date"]).dt.normalize()
    daily_deltas = d.groupby(["Day", "Ticker"], as_index=False)["Qty"].sum()

    for tk in tickers:
        s = daily_deltas[daily_deltas["Ticker"] == tk].set_index("Day")["Qty"]
        s = s.reindex(days).fillna(0.0)
        pos[tk] = s.cumsum()

    # Ignore sold / short: no negative holdings
    pos = pos.clip(lower=0.0)

    # Region map (India vs US) from txn sheet (latest known per ticker)
    last_region = (
        txn.sort_values("Date")
           .groupby("Ticker")["Region"]
           .last()
           .to_dict()
    )

    # Value daily in INR units (internal only)
    v = pd.Series(0.0, index=days)

    for tk in tickers:
        if tk not in px.columns:
            continue
        region = _normalize_region(last_region.get(tk, "US"))
        price_series = px[tk].astype(float)
        shares = pos[tk].astype(float)

        if _region_key(region) == "us":
            # USD -> INR via daily FX
            v = v.add(shares * price_series * fx, fill_value=0.0)
        else:
            # India already INR
            v = v.add(shares * price_series, fill_value=0.0)

    v = v.replace([float("inf"), float("-inf")], pd.NA).dropna()
    v = v[v > 0]

    if v.empty:
        return None

    # Base=100 index (no INR values exposed)
    idx = (v / float(v.iloc[0])) * 100.0
    idx.name = "My Portfolio (Indexed)"
    return idx

@st.cache_data(ttl=1800)
def build_benchmark_index(symbol: str, start_date: pd.Timestamp):
    if start_date is None or pd.isna(start_date):
        return None
    start_date = pd.to_datetime(start_date).tz_localize(None)

    px = yf.download(
        symbol,
        start=(start_date - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False
    )

    if px is None or px.empty or "Close" not in px.columns:
        return None

    s = px["Close"].dropna()
    if s.empty:
        return None

    idx = (s / float(s.iloc[0])) * 100.0
    idx.name = f"{symbol} (Indexed)"
    return idx

# ============================================================
# Calendar Heatmap (unchanged)
# ============================================================
@st.cache_data(ttl=900)
def build_daily_alpha_heatmap_series(tickers, weights, start_date):
    if not tickers:
        return pd.DataFrame()

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

    if isinstance(px, pd.Series):
        px = px.to_frame()

    px = px.dropna(how="all").ffill()
    rets = px.pct_change().dropna(how="all")

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
    if alpha_df is None or alpha_df.empty:
        st.info("Heatmap unavailable right now (not enough daily data).")
        return

    df = alpha_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Dow"] = df["Date"].dt.dayofweek
    df["DowName"] = df["Date"].dt.day_name().str[:3]
    df["WeekStart"] = df["Date"] - pd.to_timedelta(df["Dow"], unit="D")
    df["WeekLabel"] = df["WeekStart"].dt.strftime("%b %d")

    pivot = df.pivot_table(index="DowName", columns="WeekLabel", values="Alpha", aggfunc="mean")
    pivot = pivot.reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorbar=dict(title="Daily Alpha"),
        hovertemplate="Day: %{y}<br>Week of: %{x}<br>Alpha: %{z:.2%}<extra></extra>"
    ))
    fig.update_layout(
        title="Calendar Heatmap (Daily Alpha vs S&P 500)",
        xaxis_title="Week (rolling)",
        yaxis_title="Day",
        margin=dict(l=10, r=10, t=40, b=10),
        height=330
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Deep Dive (unchanged)
# ============================================================
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

        hist = None
        try:
            hist = t.history(period="1y", interval="1d", auto_adjust=False)
        except Exception:
            hist = None

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
        has_any = any([out.get("sector"), out.get("industry"), out.get("trailingPE"), out.get("hist") is not None])
        return out if has_any else None
    except Exception:
        return None

# ============================================================
# Sidebar
# ============================================================
st.sidebar.markdown(f"## {APP_TITLE}")
st.sidebar.markdown("**Institutional tracking of my portfolio and alpha.**")
st.sidebar.markdown("---")
st.sidebar.markdown("### About Me")
st.sidebar.write(
    "I am Atharva Bhutada, an equity research-focused finance professional. "
    "This dashboard tracks my portfolio positioning, performance, and alpha vs benchmarks."
)
st.sidebar.markdown("**Contact:**")
st.sidebar.markdown("[LinkedIn: linkedin.com/in/atharva-bhutada](https://linkedin.com/in/atharva-bhutada)")
st.sidebar.markdown("[Email: abhutada1@babson.edu](mailto:abhutada1@babson.edu)")
st.sidebar.markdown("[Substack: atharvabhutada1.substack.com](https://atharvabhutada1.substack.com/)")

st.sidebar.markdown("---")
st.sidebar.markdown("### Follow Updates")
with st.sidebar.form("follow_form", clear_on_submit=True):
    email = st.text_input("Enter your email to follow updates", placeholder="name@email.com")
    submitted = st.form_submit_button("Follow")
    if submitted:
        if email and "@" in email:
            st.success("Saved. (Connect this to Zapier/Make/SMTP to send notifications.)")
        else:
            st.warning("Enter a valid email.")

# ============================================================
# Load holdings sheet
# ============================================================
try:
    df_sheet = load_and_clean_data(SHEET_URL)
except Exception as e:
    st.error(f"Spreadsheet Error: {e}")
    st.stop()

if df_sheet.empty:
    st.warning("No valid holdings found. Check Ticker/Region/QTY/AvgCost.")
    st.stop()

# Google Sheet fallback dict
sheet_fallback = {}
for _, r in df_sheet.iterrows():
    tk = _clean_str(r.get("Ticker", ""))
    sheet_fallback[tk] = {"live": r.get("LivePrice_GS", None), "prev": r.get("PrevClose_GS", None)}

holdings = df_sheet["Ticker"].unique().tolist()
benchmarks = df_sheet["Benchmark"].unique().tolist()
all_symbols = list(set(holdings + benchmarks + ["USDINR=X"] + MACRO_ASSETS))

with st.spinner("Syncing portfolio data..."):
    prices = build_prices_with_sheet_fallback(all_symbols, sheet_fallback)
    macro = fetch_5y_macro()

# ============================================================
# Compute per-holding metrics
# ============================================================
rows, failures = [], []
for _, r in df_sheet.iterrows():
    tk = _clean_str(r["Ticker"])
    bench = _clean_str(r["Benchmark"])
    region = r["Region"]

    p_tk = prices.get(tk, None)
    p_b = prices.get(bench, None) if bench else None

    if not p_tk or p_tk.get("live", None) is None or p_tk.get("prev", None) is None:
        failures.append({"Ticker": tk, "Reason": "Missing holding price (Yahoo + Sheet failed)"})
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

    # Exposure weights only (do NOT show value)
    # USDINR not needed for returns, only weights, we use last price map for USDINR=X
    fx_live = prices.get("USDINR=X", {}).get("live", None)
    fx_live = float(fx_live) if fx_live else 83.0
    value_inr = qty * live * (fx_live if _region_key(region) == "us" else 1.0)

    rows.append({
        "Ticker": tk,
        "Region": region,
        "Benchmark": bench,
        "Compared To": _bench_context(bench),
        "QTY": qty,
        "AvgCost": avg,
        "LivePrice": live,
        "PrevClose": prev,
        "Value_INR": value_inr,  # internal only
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

# ============================================================
# Header + status
# ============================================================
st.title("📈 Atharva Portfolio Returns")
st.markdown(
    "This dashboard tracks my portfolio performance and **alpha vs the S&P 500**, with India positions benchmarked to Nifty at the stock level."
)
st.caption("Returns are shown in native currency. FX (USD/INR) is used only for exposure weights, not for returns.")

now_utc = datetime.now(timezone.utc)
status_text, status_type = _market_status_badge(now_utc, calc_df)
if status_type == "closed":
    st.info(status_text)
elif status_type == "mixed":
    st.warning(status_text)
else:
    st.success(status_text)

if calc_df.empty:
    st.error("Pricing is temporarily unavailable. Your page is still live, please refresh shortly.")
    if failures:
        st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)
    st.stop()

# ============================================================
# Weights
# ============================================================
calc_df = calc_df[calc_df["QTY"] > 0].copy()
den = calc_df["Value_INR"].sum()
calc_df["Weight"] = (calc_df["Value_INR"] / den) if den and den > 0 else 0.0

# Portfolio metrics
port_day = (calc_df["Day_Ret"] * calc_df["Weight"]).sum()
port_total = (calc_df["Total_Ret"] * calc_df["Weight"]).sum()

def _safe_day(sym):
    p = prices.get(sym, None)
    if not p or p.get("live", None) is None or p.get("prev", None) is None or float(p["prev"]) == 0:
        return None
    return (float(p["live"]) - float(p["prev"])) / float(p["prev"])

spx_day = _safe_day("^GSPC")
daily_alpha_vs_spx = (port_day - spx_day) if (spx_day is not None) else None

# ============================================================
# Inception Index (Portfolio vs S&P) from Transactions Ledger
# ============================================================
txn_df = pd.DataFrame()
portfolio_idx = None
spx_idx = None
inception_alpha_vs_spx = None

try:
    txn_df = load_transactions(TRANSACTIONS_URL)
    portfolio_idx = build_equity_curve_index_from_ledger(txn_df)

    if portfolio_idx is not None and not portfolio_idx.empty:
        start_dt = pd.to_datetime(portfolio_idx.index.min()).tz_localize(None)
        spx_idx = build_benchmark_index("^GSPC", start_dt)

        if spx_idx is not None and not spx_idx.empty:
            m = pd.concat([portfolio_idx, spx_idx], axis=1).dropna()
            if not m.empty:
                strat_total = (float(m.iloc[-1, 0]) / float(m.iloc[0, 0])) - 1.0
                spx_total = (float(m.iloc[-1, 1]) / float(m.iloc[0, 1])) - 1.0
                inception_alpha_vs_spx = strat_total - spx_total
except Exception as e:
    st.warning(f"Transactions ledger loaded with issues: {e}")

# ============================================================
# Top metrics row
# ============================================================
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(_tooltip("**Total Return (Strategy)**", "Absolute return since inception, based on AvgCost, weighted by current exposure."), unsafe_allow_html=True)
    st.metric(label="", value=f"{port_total*100:.2f}%")

with m2:
    st.markdown(_tooltip("**Today Return (Portfolio)**", "Weighted return of holdings today (PrevClose vs LivePrice)."), unsafe_allow_html=True)
    st.metric(label="", value=f"{port_day*100:.2f}%")

with m3:
    st.markdown(_tooltip("**Daily Alpha (vs S&P)**", "Portfolio Today Return minus S&P 500 Today Return. Positive means outperforming S&P today."), unsafe_allow_html=True)
    st.metric(label="", value="—" if daily_alpha_vs_spx is None else f"{daily_alpha_vs_spx*100:.2f}%")

with m4:
    st.markdown(_tooltip("**Inception Alpha (vs S&P)**", "Uses Transactions ledger to reconstruct daily holdings, values everything in INR (US via USDINR=X), then rebases to Base=100. Alpha is strategy total return minus S&P return."), unsafe_allow_html=True)
    st.metric(label="", value="—" if inception_alpha_vs_spx is None else f"{inception_alpha_vs_spx*100:.2f}%")

st.caption(_tooltip("Last Sync (UTC)", "Pricing uses Yahoo Finance with a Google Sheet fallback when available."), unsafe_allow_html=True)
st.write(now_utc.strftime("%Y-%m-%d %H:%M"))

# ============================================================
# Tabs
# ============================================================
st.divider()
tabs = st.tabs(["Combined", "India", "US"])

def _filter_region(df, region_name):
    return df[df["Region"].str.upper() == region_name.upper()].copy()

def _rebase_to_100(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        s = out[c].dropna()
        if s.empty:
            continue
        base = float(s.iloc[0])
        if base == 0:
            continue
        out[c] = (out[c] / base) * 100.0
    return out

def _plot_indexed_strategy(macro_df, portfolio_series=None, spx_series=None, title="Strategy vs Benchmarks (Indexed to 100 at Start)"):
    if macro_df is None or macro_df.empty:
        st.info("Macro trend data unavailable right now.")
        return

    macro_named = macro_df.copy().rename(columns={k: v for k, v in ASSET_LABELS.items() if k in macro_df.columns})
    plot_df = macro_named.copy()

    # Convert portfolio/spx to Series safely, then resample to month-end to match macro
    if portfolio_series is not None and not pd.Series(portfolio_series).empty:
        p = _ensure_series(portfolio_series, preferred_col="My Portfolio (Indexed)")
        if p is not None and not p.dropna().empty:
            p = p.resample("M").last()
            p.name = "My Portfolio (Indexed)"
            plot_df = plot_df.join(p, how="left")

    if spx_series is not None:
        s = _ensure_series(spx_series)
        if s is not None and not s.dropna().empty:
            s = s.resample("M").last()
            s.name = "S&P 500 (Indexed)"
            plot_df = plot_df.join(s, how="left")

    plot_df = plot_df.dropna(how="all").ffill()
    if plot_df.empty:
        st.info("Not enough data to plot right now.")
        return

    plot_idx = _rebase_to_100(plot_df)

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
        yaxis_title="Return Index (Base=100)",
        legend_title="Series",
        margin=dict(l=10, r=10, t=40, b=10),
        height=380
    )
    st.plotly_chart(fig, use_container_width=True)

with tabs[0]:
    left, right = st.columns([2, 1])

    with left:
        st.subheader("📈 Strategy vs Benchmarks (Return Index, Base=100)")
        if portfolio_idx is None:
            st.info("Inception equity curve unavailable (transactions/prices missing).")
        _plot_indexed_strategy(
            macro_df=macro,
            portfolio_series=portfolio_idx,
            spx_series=spx_idx,
            title="Strategy vs Macro Benchmarks (Indexed)"
        )

    with right:
        st.subheader("🌍 Country Risk (Live Exposure)")
        alloc = calc_df.groupby("Region", as_index=False)["Weight"].sum()
        fig_pie = px.pie(alloc, values="Weight", names="Region", hole=0.5)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()
    st.subheader("🟩🟥 Calendar Heatmap")
    st.caption("Daily alpha squares show how my portfolio performed **vs S&P 500 each day** over recent months.")
    weights_map = {row["Ticker"]: row["Weight"] for _, row in calc_df.iterrows()}
    alpha_daily = build_daily_alpha_heatmap_series(
        tickers=calc_df["Ticker"].unique().tolist(),
        weights=weights_map,
        start_date=txn_df["Date"].min() if (txn_df is not None and not txn_df.empty) else None
    )
    render_calendar_heatmap(alpha_daily)

with tabs[1]:
    st.subheader("🇮🇳 India View")
    india_df = _filter_region(calc_df, "INDIA")
    if india_df.empty:
        st.info("No India holdings right now.")
    else:
        st.write(f"Holdings: {len(india_df)} | Live exposure weight: {india_df['Weight'].sum()*100:.2f}%")

with tabs[2]:
    st.subheader("🇺🇸 US View")
    us_df = _filter_region(calc_df, "US")
    if us_df.empty:
        st.info("No US holdings right now.")
    else:
        st.write(f"Holdings: {len(us_df)} | Live exposure weight: {us_df['Weight'].sum()*100:.2f}%")

# ============================================================
# Picks table + Export (privacy)
# ============================================================
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

show = show.reset_index(drop=True)
show.index = show.index + 1

st.dataframe(
    show[[
        "Ticker", "Region", "Benchmark", "Compared To",
        "Weight%", "Total Ret%", "Day Ret%",
        "Beat_Index_Tag", "Score vs Index%", "PriceSource"
    ]],
    column_config={
        "Weight%": st.column_config.NumberColumn("Weight", format="%.2f%%"),
        "Total Ret%": st.column_config.NumberColumn("Total Return", format="%.2f%%"),
        "Day Ret%": st.column_config.NumberColumn("Today Return", format="%.2f%%"),
        "Beat_Index_Tag": st.column_config.TextColumn("Beat Index?"),
        "Score vs Index%": st.column_config.NumberColumn("Score vs Index", format="%.2f%%"),
        "Compared To": st.column_config.TextColumn("Compared To"),
        "PriceSource": st.column_config.TextColumn("Price Source"),
    },
    use_container_width=True,
    hide_index=False
)

csv_bytes = show[[
    "Ticker", "Region", "Benchmark", "Compared To",
    "Weight%", "Total Ret%", "Day Ret%", "Beat_Index_Tag", "Score vs Index%", "PriceSource"
]].reset_index(names="Row").to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Picks as CSV",
    data=csv_bytes,
    file_name="atharva_portfolio_picks.csv",
    mime="text/csv",
    key="download_picks_csv_main"
)

# ============================================================
# Deep Dive
# ============================================================
st.divider()
st.subheader("🔎 Deep Dive (Stock History & Key Ratios)")

tickers_sorted = sorted(show["Ticker"].unique().tolist())
selected = st.selectbox("Select a ticker", tickers_sorted)

deep = fetch_ticker_deep_dive(selected)
if deep is None:
    st.info("Deep dive is temporarily unavailable for this ticker (Yahoo Finance did not return fundamentals).")
else:
    cA, cB = st.columns([2, 1])

    with cA:
        hist = deep.get("hist", None)
        if hist is not None and not hist.empty and "Close" in hist.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close"))
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
        st.write(f"**Type:** {r.get('Type','') if 'Type' in r else ''}")
        st.write(f"**Region:** {r.get('Region','')}")
        st.write(f"**Benchmark:** {r.get('Benchmark','')}")
        raw = calc_df[calc_df["Ticker"] == selected]
        if not raw.empty:
            st.text_area("Thesis (edit in Google Sheet)", value=str(raw.iloc[0].get("Thesis", "")), height=180)

# ============================================================
# Failures
# ============================================================
if failures:
    st.divider()
    st.warning("Some symbols could not be priced (dashboard still runs on partial data).")
    st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)

st.divider()
st.caption("Data source: Yahoo Finance (with Google Sheet LivePrice_GS/PrevClose_GS fallback). Educational project, not investment advice.")
