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
# - NO ₹ value shown anywhere, only Base=100 index + weights
# - Google Sheet LivePrice_GS/PrevClose_GS always acts as fallback for snapshot
# - Removes Transactions-ledger equity curve (you asked to drop it for now)
# - Adds:
#   1) Webull-style P&L Calendar (monthly grid, day+P&L%, red/green, 0-centered)
#   2) Realistic 4Y Blended Alpha:
#        India sleeve vs Nifty (INR) + US sleeve vs S&P (USD), blended by exposure
# ============================================================

APP_TITLE = "Atharva Portfolio Returns"
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📈")

# ========= YOUR SHEET URLS =========
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUHmE__8dpl_nKGv5F5mTXO7e3EyVRqz-PJF_4yyIrfJAa7z8XgzkIw6IdLnaotkACka2Q-PvP8P-z/pub?output=csv"

# =============================
# Columns & config
# =============================
REQUIRED_COLS = ["Ticker", "Region", "QTY", "AvgCost"]
OPTIONAL_COLS = ["Benchmark", "Type", "Thesis", "FirstBuyDate", "LivePrice_GS", "PrevClose_GS"]

SUMMARY_NOISE_REGEX = r"TOTAL|PORTFOLIO|SUMMARY|CASH"
DEFAULT_BENCH = {"us": "^GSPC", "india": "^NSEI"}  # per region if Benchmark blank

MACRO_ASSETS = ["^GSPC", "^NSEI", "GC=F", "SI=F"]
ASSET_LABELS = {
    "^GSPC": "^GSPC (S&P 500)",
    "^NSEI": "^NSEI (Nifty 50)",
    "GC=F": "GC=F (Gold)",
    "SI=F": "SI=F (Silver)",
}

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
        return f"vs {b}"
    return f"vs {b}" if b else "—"

def _status_tag(alpha_day, bench):
    if alpha_day is None or pd.isna(alpha_day):
        return "—"
    short = "Nifty" if bench == "^NSEI" else ("S&P" if bench == "^GSPC" else "Index")
    return (f"🔥 Beating Market (vs {short})") if alpha_day >= 0 else (f"❄️ Lagging Market (vs {short})")

def _tooltip(label: str, help_text: str):
    safe_help = help_text.replace('"', "'")
    return f"""{label} <span title="{safe_help}" style="cursor:default;">ⓘ</span>"""

def _is_india_ticker(tk: str) -> bool:
    t = _clean_str(tk).upper()
    return t.endswith(".NS") or t.endswith(".BO") or t in ["^NSEI"]

def _is_us_ticker(tk: str) -> bool:
    t = _clean_str(tk).upper()
    if t in ["^GSPC"]:
        return True
    return (not _is_india_ticker(t))  # simple rule for this project

# ============================================================
# Market session logic (India + US)
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
# Pricing Engine (Yahoo primary + Sheet fallback ALWAYS for snapshot)
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
                    group_by="column",
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

        # Per-ticker fallback (single download)
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

        # Sheet fallback ALWAYS (snapshot)
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
def fetch_fx_usdinr_snapshot():
    live, _ = _fast_live_prev("USDINR=X")
    if live is not None and live > 0:
        return float(live)
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
    t = yf.download(MACRO_ASSETS, period="5y", interval="1mo", progress=False, auto_adjust=False, threads=False)
    if t is None or t.empty:
        return pd.DataFrame()
    t = t["Close"] if "Close" in t.columns else t
    if isinstance(t, pd.Series):
        t = t.to_frame()
    return t.dropna(how="all").ffill()

# ============================================================
# 4Y Blended Alpha (realistic)
# ============================================================
@st.cache_data(ttl=1800)
def _download_close_daily(symbols, start, end):
    if not symbols:
        return pd.DataFrame()
    df = yf.download(
        tickers=sorted(list(set(symbols))),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    close = df["Close"] if isinstance(df, pd.DataFrame) and "Close" in df.columns else df
    if isinstance(close, pd.Series):
        close = close.to_frame()
    return close.dropna(how="all").ffill()

def _total_return_from_close(close: pd.Series):
    s = close.dropna()
    if len(s) < 2:
        return None
    return float(s.iloc[-1] / s.iloc[0] - 1.0)

def _weighted_total_return_from_close(close_df: pd.DataFrame, weights: dict):
    cols = [c for c in close_df.columns if c in weights]
    if not cols:
        return None
    px = close_df[cols].dropna(how="all").ffill()
    if px.empty or len(px) < 2:
        return None
    rets = px.pct_change().fillna(0.0)
    w = pd.Series({k: float(v) for k, v in weights.items() if k in cols})
    if w.sum() == 0:
        return None
    w = (w / w.sum()).reindex(cols).fillna(0.0)
    port_daily = (rets * w).sum(axis=1)
    port_index = (1.0 + port_daily).cumprod()
    return float(port_index.iloc[-1] / port_index.iloc[0] - 1.0)

@st.cache_data(ttl=1800)
def compute_blended_alpha_4y(calc_df: pd.DataFrame):
    if calc_df is None or calc_df.empty:
        return None

    end = pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)
    start = (pd.Timestamp.utcnow().normalize() - pd.DateOffset(years=4) - pd.Timedelta(days=10))

    india = calc_df[calc_df["Region"].astype(str).str.upper().eq("INDIA")].copy()
    us = calc_df[calc_df["Region"].astype(str).str.upper().eq("US")].copy()

    india_exposure = float(india["Value_INR"].sum()) if not india.empty else 0.0
    us_exposure = float(us["Value_INR"].sum()) if not us.empty else 0.0
    total_exposure = india_exposure + us_exposure
    if total_exposure <= 0:
        return None

    w_india_blend = india_exposure / total_exposure
    w_us_blend = us_exposure / total_exposure

    out = {
        "alpha_blended": None,
        "alpha_india": None,
        "alpha_us": None,
        "w_india": w_india_blend,
        "w_us": w_us_blend,
    }

    # India sleeve vs Nifty (INR)
    if not india.empty:
        india_tk = sorted(list(set(india["Ticker"].dropna().tolist())))
        px_india = _download_close_daily(india_tk + ["^NSEI"], start, end)
        if "^NSEI" in px_india.columns:
            w_india = {row["Ticker"]: float(row["Weight"]) for _, row in india.iterrows()}
            india_port = _weighted_total_return_from_close(px_india, w_india)
            india_bench = _total_return_from_close(px_india["^NSEI"])
            if india_port is not None and india_bench is not None:
                out["alpha_india"] = india_port - india_bench

    # US sleeve vs S&P (USD)
    if not us.empty:
        us_tk = sorted(list(set(us["Ticker"].dropna().tolist())))
        px_us = _download_close_daily(us_tk + ["^GSPC"], start, end)
        if "^GSPC" in px_us.columns:
            us = us.copy()
            us["Value_USD"] = us["QTY"].astype(float) * us["LivePrice"].astype(float)
            denom = float(us["Value_USD"].sum())
            w_us = {}
            if denom > 0:
                for _, r in us.iterrows():
                    w_us[r["Ticker"]] = float(r["Value_USD"] / denom)
            us_port = _weighted_total_return_from_close(px_us, w_us)
            us_bench = _total_return_from_close(px_us["^GSPC"])
            if us_port is not None and us_bench is not None:
                out["alpha_us"] = us_port - us_bench

    parts, weights = [], []
    if out["alpha_india"] is not None and w_india_blend > 0:
        parts.append(out["alpha_india"]); weights.append(w_india_blend)
    if out["alpha_us"] is not None and w_us_blend > 0:
        parts.append(out["alpha_us"]); weights.append(w_us_blend)

    if not parts:
        return out

    wsum = sum(weights)
    out["alpha_blended"] = sum(p * (w / wsum) for p, w in zip(parts, weights))
    return out

# ============================================================
# Webull-style P&L Calendar (monthly grid)
# - Uses a fixed-weight daily return series (privacy-safe)
# - Converts US tickers to INR using USDINR=X each day so one series exists
# ============================================================
@st.cache_data(ttl=1800)
def build_portfolio_daily_returns_series_5y(calc_df: pd.DataFrame):
    if calc_df is None or calc_df.empty:
        return pd.DataFrame()

    end = pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)
    start = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=365 * 5 + 15)

    tickers = sorted(list(set(calc_df["Ticker"].dropna().tolist())))
    if not tickers:
        return pd.DataFrame()

    need = tickers + ["USDINR=X"]
    px = _download_close_daily(need, start, end)
    if px is None or px.empty:
        return pd.DataFrame()
    px = px.dropna(how="all").ffill()

    fx = px["USDINR=X"].dropna().ffill() if "USDINR=X" in px.columns else None

    # Build INR-consistent price panel
    px_inr = px.copy()
    if fx is not None and not fx.empty:
        fx_a = fx.reindex(px_inr.index).ffill()
        for tk in tickers:
            if tk in px_inr.columns and _is_us_ticker(tk) and tk not in ["GC=F", "SI=F"]:
                px_inr[tk] = px_inr[tk] * fx_a

    # Fixed weights based on current exposure (already computed)
    w = {row["Ticker"]: float(row["Weight"]) for _, row in calc_df.iterrows()}
    cols = [c for c in tickers if c in px_inr.columns and c in w]
    if not cols:
        return pd.DataFrame()

    px_use = px_inr[cols].dropna(how="all").ffill()
    rets = px_use.pct_change().fillna(0.0)
    ws = pd.Series({k: float(v) for k, v in w.items() if k in cols})
    if ws.sum() == 0:
        return pd.DataFrame()
    ws = (ws / ws.sum()).reindex(cols).fillna(0.0)

    port_ret = (rets * ws).sum(axis=1)
    out = pd.DataFrame({"Date": port_ret.index, "PortRet": port_ret.values})
    out = out.dropna()
    return out.reset_index(drop=True)

def render_webull_calendar(port_daily: pd.DataFrame):
    if port_daily is None or port_daily.empty:
        st.info("Calendar unavailable (not enough daily data).")
        return

    df = port_daily.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    months = df[["Year", "Month"]].drop_duplicates().sort_values(["Year", "Month"])
    if months.empty:
        st.info("Calendar unavailable.")
        return

    # Default to latest month
    latest = months.iloc[-1]
    year_options = sorted(months["Year"].unique().tolist())
    y = st.selectbox("Year", year_options, index=year_options.index(int(latest["Year"])), key="webull_year")

    month_options = months[months["Year"] == y]["Month"].tolist()
    month_labels = {m: pd.Timestamp(2000, m, 1).strftime("%B") for m in range(1, 13)}
    m_default = int(latest["Month"]) if int(latest["Year"]) == int(y) else month_options[-1]
    m = st.selectbox(
        "Month",
        month_options,
        index=month_options.index(m_default),
        format_func=lambda mm: month_labels.get(mm, str(mm)),
        key="webull_month"
    )

    sub = df[(df["Year"] == y) & (df["Month"] == m)].copy()
    if sub.empty:
        st.info("No daily data for this month.")
        return

    # Webull-like monthly grid:
    # rows: Mon..Sun (0..6), cols: week-of-month (integer)
    sub["Day"] = sub["Date"].dt.day
    sub["Dow"] = sub["Date"].dt.dayofweek  # Mon=0
    first = pd.Timestamp(int(y), int(m), 1)
    first_dow = first.dayofweek
    sub["WeekOfMonth"] = ((sub["Day"] + first_dow - 1) // 7) + 1

    # Make full grid to keep square layout stable
    max_week = int(sub["WeekOfMonth"].max())
    grid = pd.MultiIndex.from_product([range(0, 7), range(1, max_week + 1)], names=["Dow", "WeekOfMonth"]).to_frame(index=False)
    sub2 = grid.merge(sub[["Dow", "WeekOfMonth", "Day", "PortRet"]], on=["Dow", "WeekOfMonth"], how="left")

    # Heat values (0-centered)
    z = sub2.pivot(index="Dow", columns="WeekOfMonth", values="PortRet").reindex(range(0, 7))
    z = z.fillna(0.0)

    # Text: day + P&L%
    def _cell_text(d, r):
        if pd.isna(d):
            return ""
        if pd.isna(r):
            return f"{int(d)}"
        return f"{int(d)}<br>{r*100:+.2f}%"

    text = sub2.pivot(index="Dow", columns="WeekOfMonth", values="Day").reindex(range(0, 7))
    retp = sub2.pivot(index="Dow", columns="WeekOfMonth", values="PortRet").reindex(range(0, 7))

    text_out = []
    for i in range(text.shape[0]):
        row = []
        for j in range(text.shape[1]):
            d = text.iloc[i, j]
            r = retp.iloc[i, j]
            if pd.isna(d):
                row.append("")
            else:
                row.append(_cell_text(d, r))
        text_out.append(row)

    # Symmetric range around 0
    vmax = float(pd.Series(sub["PortRet"]).abs().quantile(0.98)) if len(sub) else 0.02
    vmax = max(vmax, 0.01)
    vmin = -vmax

    y_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    x_labels = [f"W{k}" for k in range(1, max_week + 1)]

    fig = go.Figure(data=go.Heatmap(
        z=z.values,
        x=x_labels,
        y=y_labels,
        zmin=vmin,
        zmax=vmax,
        colorscale="RdYlGn",  # red negative, green positive
        showscale=True,
        colorbar=dict(title="Daily P&L %", tickformat=".0%"),
        text=text_out,
        texttemplate="%{text}",
        hovertemplate="Week: %{x}<br>Day: %{y}<br>P&L: %{z:.2%}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Webull-Style P&L Calendar ({pd.Timestamp(int(y), int(m), 1).strftime('%B %Y')})",
        margin=dict(l=10, r=10, t=50, b=10),
        height=520,
    )

    # Force square-ish cells (visual 1:1 feel)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    st.plotly_chart(fig, use_container_width=True)

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

st.sidebar.markdown("**Contact**")
st.sidebar.markdown("LinkedIn:")
st.sidebar.markdown("[linkedin.com/in/atharva-bhutada](https://linkedin.com/in/atharva-bhutada)")
st.sidebar.markdown("Email:")
st.sidebar.markdown("[abhutada1@babson.edu](mailto:abhutada1@babson.edu)")
st.sidebar.markdown("Substack:")
st.sidebar.markdown("[atharvabhutada1.substack.com](https://atharvabhutada1.substack.com/)")

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

# Build Google Sheet fallback dict (per ticker)
sheet_fallback = {}
for _, r in df_sheet.iterrows():
    tk = _clean_str(r.get("Ticker", ""))
    sheet_fallback[tk] = {
        "live": r.get("LivePrice_GS", None),
        "prev": r.get("PrevClose_GS", None),
    }

holdings = df_sheet["Ticker"].unique().tolist()
benchmarks = df_sheet["Benchmark"].unique().tolist()
all_symbols = list(set(holdings + benchmarks + ["USDINR=X"] + MACRO_ASSETS))

with st.spinner("Syncing portfolio data..."):
    prices = build_prices_with_sheet_fallback(all_symbols, sheet_fallback)
    fx_usdinr = fetch_fx_usdinr_snapshot()
    macro = fetch_5y_macro()

# ============================================================
# Compute per-holding metrics (NO absolute portfolio value displayed)
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

    # exposure weights only (internal); NO display of value
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
st.caption("Returns are shown in native currency. FX (USD/INR) is used only for exposure weights, not displayed.")

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
den = float(calc_df["Value_INR"].sum())
calc_df["Weight"] = (calc_df["Value_INR"] / den) if den > 0 else 0.0

# Snapshot metrics
port_day = float((calc_df["Day_Ret"] * calc_df["Weight"]).sum())
port_total = float((calc_df["Total_Ret"] * calc_df["Weight"]).sum())

def _safe_day(sym):
    p = prices.get(sym, None)
    if not p or p.get("live", None) is None or p.get("prev", None) is None or float(p["prev"]) == 0:
        return None
    return (float(p["live"]) - float(p["prev"])) / float(p["prev"])

spx_day = _safe_day("^GSPC")
daily_alpha_vs_spx = (port_day - spx_day) if (spx_day is not None) else None

# 4Y blended alpha (realistic)
alpha_detail = None
alpha_4y_blended = None
try:
    alpha_detail = compute_blended_alpha_4y(calc_df)
    if alpha_detail and alpha_detail.get("alpha_blended", None) is not None:
        alpha_4y_blended = float(alpha_detail["alpha_blended"])
except Exception:
    alpha_4y_blended = None

# ============================================================
# Top metrics row
# ============================================================
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(_tooltip("**Total Return (Strategy)**", "Absolute return vs AvgCost, weighted by current exposure."), unsafe_allow_html=True)
    st.metric(label="", value=f"{port_total*100:.2f}%")

with m2:
    st.markdown(_tooltip("**Today Return (Portfolio)**", "Weighted return today (PrevClose vs LivePrice)."), unsafe_allow_html=True)
    st.metric(label="", value=f"{port_day*100:.2f}%")

with m3:
    st.markdown(_tooltip("**Daily Alpha (vs S&P)**", "Portfolio Today Return minus S&P 500 Today Return."), unsafe_allow_html=True)
    st.metric(label="", value="—" if daily_alpha_vs_spx is None else f"{daily_alpha_vs_spx*100:.2f}%")

with m4:
    st.markdown(_tooltip(
        "**4Y Alpha (India vs Nifty + US vs S&P)**",
        "Fair alpha: India sleeve is compared to Nifty in INR, US sleeve is compared to S&P in USD. Sleeve alphas are blended using current exposure weights."
    ), unsafe_allow_html=True)
    st.metric(label="", value="—" if alpha_4y_blended is None else f"{alpha_4y_blended*100:.2f}%")

if alpha_detail:
    st.caption(f"Blend weights: India {alpha_detail.get('w_india',0)*100:.1f}%, US {alpha_detail.get('w_us',0)*100:.1f}%")

st.caption(_tooltip("Last Sync (UTC)", "Pricing uses Yahoo Finance with a Google Sheet LivePrice_GS/PrevClose_GS fallback for snapshot."), unsafe_allow_html=True)
st.write(now_utc.strftime("%Y-%m-%d %H:%M"))

# ============================================================
# Tabs
# ============================================================
st.divider()
tabs = st.tabs(["Combined", "India", "US"])

def _filter_region(df, region_name):
    return df[df["Region"].astype(str).str.upper() == region_name.upper()].copy()

with tabs[0]:
    left, right = st.columns([2, 1])

    with left:
        st.subheader("🗓️ Webull-Style P&L Calendar")
        st.caption("Monthly P&L calendar uses a **fixed-weight daily return series** (privacy-safe). Red is negative, green is positive, white is ~0%.")
        daily = build_portfolio_daily_returns_series_5y(calc_df)
        render_webull_calendar(daily)

    with right:
        st.subheader("🌍 Country Risk (Live Exposure)")
        alloc = calc_df.groupby("Region", as_index=False)["Weight"].sum()
        fig_pie = px.pie(alloc, values="Weight", names="Region", hole=0.5)
        st.plotly_chart(fig_pie, use_container_width=True)

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
# Failures
# ============================================================
if failures:
    st.divider()
    st.warning("Some symbols could not be priced (dashboard still runs on partial data).")
    st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)

st.divider()
st.caption("Data source: Yahoo Finance (with Google Sheet LivePrice_GS/PrevClose_GS fallback for snapshot). Educational project, not investment advice.")
