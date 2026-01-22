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
# - Inception equity curve uses Transactions ledger + daily prices
#   and converts US holdings to INR using USDINR=X daily series
# - Fixes StreamlitDuplicateElementId, yfinance threading, and plot crash
# - Calendar heatmap uses last 5 years
# - Inception Alpha uses trailing 4Y window to avoid data ghosts
# ============================================================

APP_TITLE = "Atharva Portfolio Returns"
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📈")

# ========= YOUR SHEET URLS =========
# Holdings (Current portfolio snapshot)
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUHmE__8dpl_nKGv5F5mTXO7e3EyVRqz-PJF_4yyIrfJAa7z8XgzkIw6IdLnaotkACka2Q-PvP8P-z/pub?output=csv"

# Transactions (Ledger) - published CSV
TRANSACTIONS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR-OybDEJRMpK5jvtLnMq3SOze-ZwT6hVY07w4nAnKfn1dva_E68fKSZQkn0yvzDhk217HEQ7xis77G/pub?output=csv"

# =============================
# Columns & config
# =============================
REQUIRED_COLS = ["Ticker", "Region", "QTY", "AvgCost"]
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
# Load transactions ledger (for equity curve)
# Expected columns in your sheet (exact, case-sensitive):
# Ticker, Date, QTY, Region (optional), FX_Rate (optional), Type (optional)
# ============================================================
def _col_series(frame: pd.DataFrame, colname: str):
    """Return a 1D Series safely."""
    if colname is None:
        return None
    x = frame[colname]
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return x

@st.cache_data(ttl=300)
def load_transactions(url: str) -> pd.DataFrame:
    if not url or str(url).strip() == "":
        return pd.DataFrame()

    df = pd.read_csv(url)
    if df is None or df.empty:
        return pd.DataFrame()

    # Normalize headers + drop empty export columns
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~pd.Series(df.columns).astype(str).str.match(r"^Unnamed")].copy()

    # Defensive: duplicated headers in Google Sheets exports
    if pd.Index(df.columns).duplicated().any():
        df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
        df.columns = [str(c).strip() for c in df.columns]

    # Exact schema (you said labels will never change)
    required_txn_cols = ["Ticker", "Date", "QTY"]
    missing = [c for c in required_txn_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Transactions CSV missing required columns: {missing}. Found: {list(df.columns)}")

    c_tkr = "Ticker"
    c_date = "Date"
    c_qty = "QTY"
    c_region = "Region" if "Region" in df.columns else None
    c_fx = "FX_Rate" if "FX_Rate" in df.columns else None
    c_type = "Type" if "Type" in df.columns else None

    n = len(df)
    out = pd.DataFrame(index=range(n))
    out["Date"] = pd.to_datetime(_col_series(df, c_date), errors="coerce")
    out["Ticker"] = _col_series(df, c_tkr).apply(_clean_str)
    out["Qty"] = _col_series(df, c_qty).apply(_as_float)

    out["Region"] = _col_series(df, c_region).apply(_normalize_region) if c_region else ""
    out["FX_Rate"] = _col_series(df, c_fx).apply(_as_float) if c_fx else None
    out["Type"] = _col_series(df, c_type).apply(_clean_str) if c_type else ""

    out = out.dropna(subset=["Date", "Ticker", "Qty"]).copy()
    out = out[out["Ticker"].str.len() > 0]
    out = out.sort_values("Date").reset_index(drop=True)

    # If Region is missing, infer from ticker
    def infer_region(tk):
        return "India" if _is_india_ticker(tk) else "US"

    mask_missing_region = out["Region"].astype(str).str.strip().eq("")
    if mask_missing_region.any():
        out.loc[mask_missing_region, "Region"] = out.loc[mask_missing_region, "Ticker"].apply(infer_region)

    return out

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
    t = yf.download(MACRO_ASSETS, period="5y", interval="1mo", progress=False, auto_adjust=False, threads=False)["Close"]
    if isinstance(t, pd.Series):
        t = t.to_frame()
    return t.dropna(how="all").ffill()

# ============================================================
# PRIVACY-FIRST Equity Curve (Index only, Base=100)
# - Values US holdings in INR using daily USDINR=X
# ============================================================
@st.cache_data(ttl=1800)
def build_equity_curve_index_from_ledger(txn: pd.DataFrame):
    if txn is None or txn.empty:
        return None

    start = pd.to_datetime(txn["Date"].min()).normalize()
    end = pd.Timestamp.utcnow().normalize()

    tickers = sorted(txn["Ticker"].dropna().unique().tolist())
    if not tickers:
        return None

    # Pull daily closes (bulk first)
    px_close = yf.download(
        tickers=tickers,
        start=(start - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False
    )["Close"]

    if px_close is None:
        return None

    if isinstance(px_close, pd.Series):
        # If only 1 ticker, yfinance returns Series; name it properly
        px_close = px_close.to_frame(name=tickers[0])

    # If any ticker missing, try single pulls
    got = set([str(c) for c in px_close.columns])
    missing = [t for t in tickers if t not in got]
    for tk in missing:
        try:
            one = yf.download(
                tickers=tk,
                start=(start - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False
            )
            if one is not None and (not one.empty) and "Close" in one.columns:
                px_close = px_close.join(one["Close"].rename(tk), how="outer")
        except Exception:
            continue

    if px_close.empty:
        return None

    px_close = px_close.dropna(how="all").ffill()

    # FX series for converting US tickers to INR
    fx = yf.download(
        "USDINR=X",
        start=(start - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False
    )

    fx_close = None
    if fx is not None and (not fx.empty) and "Close" in fx.columns:
        fx_close = fx["Close"].dropna().ffill()

    # Build daily positions
    days = px_close.index
    pos = pd.DataFrame(0.0, index=days, columns=px_close.columns)

    d = txn.copy()
    d["Day"] = pd.to_datetime(d["Date"]).dt.normalize()
    daily_deltas = d.groupby(["Day", "Ticker"], as_index=False)["Qty"].sum()

    for tk in pos.columns:
        s = daily_deltas[daily_deltas["Ticker"] == tk].set_index("Day")["Qty"]
        s = s.reindex(days).fillna(0.0)
        pos[tk] = s.cumsum()

    # Ignore sold/closed positions
    pos = pos.clip(lower=0.0)

    # Convert prices to INR where needed
    px_inr = px_close.copy()
    if fx_close is not None and not fx_close.empty:
        fx_aligned = fx_close.reindex(days).ffill()
        for tk in px_inr.columns:
            if _is_us_ticker(tk) and tk not in ["GC=F", "SI=F"]:
                px_inr[tk] = px_inr[tk] * fx_aligned

    # Daily portfolio value (never displayed), then index
    v = (pos * px_inr).sum(axis=1)
    v = v.replace([float("inf"), float("-inf")], pd.NA).dropna()
    v = v[v > 0]
    if v.empty:
        return None

    idx = (v / float(v.iloc[0])) * 100.0
    idx.name = "My Portfolio (Indexed)"
    return idx

@st.cache_data(ttl=1800)
def build_spx_benchmark_in_inr(start_date: pd.Timestamp):
    if start_date is None or pd.isna(start_date):
        return None

    start_date = pd.to_datetime(start_date).tz_localize(None)
    end = pd.Timestamp.utcnow().normalize()

    spx = yf.download(
        "^GSPC",
        start=(start_date - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False
    )

    if spx is None or spx.empty or "Close" not in spx.columns:
        return None

    fx = yf.download(
        "USDINR=X",
        start=(start_date - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False
    )

    if fx is None or fx.empty or "Close" not in fx.columns:
        s = spx["Close"].dropna()
        if s.empty:
            return None
        idx = (s / float(s.iloc[0])) * 100.0
        idx.name = "S&P 500 (Indexed)"
        return idx

    s = spx["Close"].dropna()
    f = fx["Close"].dropna()
    m = pd.concat([s, f], axis=1).dropna()
    if m.empty:
        return None

    spx_inr = m.iloc[:, 0] * m.iloc[:, 1]
    idx = (spx_inr / float(spx_inr.iloc[0])) * 100.0
    idx.name = "S&P 500 (Indexed)"
    return idx

# ============================================================
# Calendar Heatmap (last 5 years)
# ============================================================
@st.cache_data(ttl=900)
def build_daily_alpha_heatmap_series_5y(tickers, weights, region_map):
    if not tickers:
        return pd.DataFrame()

    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=365 * 5)

    syms = list(set(tickers + ["^GSPC", "USDINR=X"]))
    px_close = yf.download(
        tickers=syms,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False
    )["Close"]

    if px_close is None:
        return pd.DataFrame()

    if isinstance(px_close, pd.Series):
        px_close = px_close.to_frame()

    if px_close.empty:
        return pd.DataFrame()

    px_close = px_close.dropna(how="all").ffill()

    fx = px_close["USDINR=X"].dropna().ffill() if "USDINR=X" in px_close.columns else None

    rets = pd.DataFrame(index=px_close.index)
    for tk in tickers:
        if tk not in px_close.columns:
            continue
        s = px_close[tk].dropna().ffill()
        if s.empty:
            continue
        if region_map.get(tk, "") == "US" and fx is not None and not fx.empty:
            s = (s * fx.reindex(s.index).ffill())
        rets[tk] = s.pct_change()

    if "^GSPC" not in px_close.columns:
        return pd.DataFrame()

    spx = px_close["^GSPC"].dropna().ffill()
    if fx is not None and not fx.empty:
        spx = spx * fx.reindex(spx.index).ffill()
    spx_ret = spx.pct_change()

    rets = rets.join(spx_ret.rename("^GSPC_INR"), how="inner").dropna(how="all")
    if rets.empty or "^GSPC_INR" not in rets.columns:
        return pd.DataFrame()

    port = pd.Series(0.0, index=rets.index)
    for tk, w in weights.items():
        if tk in rets.columns:
            port = port.add(rets[tk].fillna(0.0) * float(w), fill_value=0.0)

    out = pd.DataFrame({
        "Date": rets.index,
        "PortfolioRet": port.values,
        "SPXRet": rets["^GSPC_INR"].values
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
        title="Calendar Heatmap (Daily Alpha vs S&P 500, last 5 years)",
        xaxis_title="Week (rolling)",
        yaxis_title="Day",
        margin=dict(l=10, r=10, t=40, b=10),
        height=330
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Deep Dive
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
        has_any = any([out.get("sector"), out.get("industry"), out.get("trailingPE"), (out.get("hist") is not None)])
        return out if has_any else None
    except Exception:
        return None

# ============================================================
# Sidebar (labels not blue; only URLs are blue)
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

    # Exposure weights only (internal); NO display of value
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
        "Value_INR": value_inr,         # internal only
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
# Weights + no sold positions
# ============================================================
calc_df = calc_df[calc_df["QTY"] > 0].copy()
den = calc_df["Value_INR"].sum()
calc_df["Weight"] = (calc_df["Value_INR"] / den) if den and den > 0 else 0.0

# Portfolio snapshot metrics
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
# - Trailing 4Y window for alpha + plot
# ============================================================
txn_df = pd.DataFrame()
portfolio_idx = None
spx_idx = None
inception_alpha_vs_spx = None
ledger_warning = None

if TRANSACTIONS_URL and str(TRANSACTIONS_URL).strip():
    try:
        txn_df = load_transactions(TRANSACTIONS_URL)

        # Sidebar debug
        with st.sidebar.expander("Transactions Debug", expanded=False):
            if txn_df is not None and not txn_df.empty:
                st.success(f"✅ Loaded {len(txn_df)} rows")
                st.write(f"Date range: {txn_df['Date'].min()} → {txn_df['Date'].max()}")
                st.write(f"Unique tickers: {txn_df['Ticker'].nunique()}")
                st.write("Columns:", list(txn_df.columns))
            else:
                st.warning("⚠️ Transactions loaded but empty")

        if txn_df is not None and (not txn_df.empty):
            portfolio_idx = build_equity_curve_index_from_ledger(txn_df)

            if portfolio_idx is not None and (not portfolio_idx.empty):
                start_dt = pd.to_datetime(portfolio_idx.index.min()).tz_localize(None)
                spx_idx = build_spx_benchmark_in_inr(start_dt)

                if spx_idx is not None and (not spx_idx.empty):
                    m = pd.concat([portfolio_idx, spx_idx], axis=1).dropna()
                    if not m.empty:
                        end_dt = pd.to_datetime(m.index.max()).tz_localize(None)
                        cutoff = end_dt - pd.DateOffset(years=4)
                        m4 = m[m.index >= cutoff]
                        if len(m4) < 2:
                            m4 = m

                        strat_total = (float(m4.iloc[-1, 0]) / float(m4.iloc[0, 0])) - 1.0
                        spx_total = (float(m4.iloc[-1, 1]) / float(m4.iloc[0, 1])) - 1.0
                        inception_alpha_vs_spx = strat_total - spx_total

                        portfolio_idx = m4.iloc[:, 0].copy()
                        spx_idx = m4.iloc[:, 1].copy()
        else:
            ledger_warning = "Transactions ledger is empty or not published correctly."
    except Exception as e:
        ledger_warning = f"Transactions ledger issue: {e}"

# ============================================================
# Top metrics row
# ============================================================
if ledger_warning:
    st.warning(ledger_warning)

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
    st.markdown(_tooltip("**Inception Alpha (vs S&P)**", "Trailing 4Y alpha using Base=100 curves from transactions. US holdings valued as USD×USDINR. Benchmark is S&P×USDINR."), unsafe_allow_html=True)
    st.metric(label="", value="—" if inception_alpha_vs_spx is None else f"{inception_alpha_vs_spx*100:.2f}%")

st.caption(_tooltip("Last Sync (UTC)", "Pricing uses Yahoo Finance with a Google Sheet LivePrice_GS/PrevClose_GS fallback for snapshot."), unsafe_allow_html=True)
st.write(now_utc.strftime("%Y-%m-%d %H:%M"))

# ============================================================
# Tabs: Combined / India / US
# ============================================================
st.divider()
tabs = st.tabs(["Combined", "India", "US"])

def _filter_region(df, region_name):
    return df[df["Region"].str.upper() == region_name.upper()].copy()

def _ensure_series(x, name=None):
    if x is None:
        return None
    if isinstance(x, pd.Series):
        y = x.copy()
        if name:
            y.name = name
        return y
    if isinstance(x, pd.DataFrame):
        if x.empty:
            return None
        s = x.iloc[:, 0].copy()
        if name:
            s.name = name
        elif s.name is None:
            s.name = "Series"
        return s
    return None

def _plot_indexed_strategy(macro_df, portfolio_series=None, spx_series=None, title="Strategy vs Benchmarks (Indexed to 100 at Start)"):
    if macro_df is None or macro_df.empty:
        st.info("Macro trend data unavailable right now.")
        return

    macro_named = macro_df.copy().rename(columns={k: v for k, v in ASSET_LABELS.items() if k in macro_df.columns})
    plot_df = macro_named.copy()

    p = _ensure_series(portfolio_series, "My Portfolio (Indexed)")
    s = _ensure_series(spx_series, "S&P 500 (Indexed)")

    if p is not None and not p.empty:
        p_m = p.resample("M").last()
        plot_df = plot_df.merge(p_m.to_frame(), left_index=True, right_index=True, how="left")

    if s is not None and not s.empty:
        s_m = s.resample("M").last()
        plot_df = plot_df.merge(s_m.to_frame(), left_index=True, right_index=True, how="left")

    plot_df = plot_df.dropna(how="all").ffill()
    if plot_df.empty:
        st.info("Not enough data to plot right now.")
        return

    base = plot_df.iloc[0]
    plot_idx = (plot_df / base) * 100.0

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
        if portfolio_idx is None or (isinstance(portfolio_idx, pd.Series) and portfolio_idx.empty):
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
    st.subheader("🟩🟥 Calendar Heatmap (last 5 years)")
    st.caption("Daily alpha squares show how my portfolio performed **vs S&P 500 each day** over the last 5 years (INR-consistent).")
    weights_map = {row["Ticker"]: row["Weight"] for _, row in calc_df.iterrows()}
    region_map = {row["Ticker"]: row["Region"] for _, row in calc_df.iterrows()}
    alpha_daily = build_daily_alpha_heatmap_series_5y(
        tickers=calc_df["Ticker"].unique().tolist(),
        weights=weights_map,
        region_map=region_map
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
    raw = calc_df[calc_df["Ticker"] == selected]
    if not raw.empty:
        st.write(f"**Type:** {raw.iloc[0].get('Type','')}")
        st.write(f"**Region:** {raw.iloc[0].get('Region','')}")
        st.write(f"**Benchmark:** {raw.iloc[0].get('Benchmark','')}")
        st.text_area("Thesis (edit in Google Sheet)", value=str(raw.iloc[0].get("Thesis", "")), height=180)

# ============================================================
# Failures (diagnostic)
# ============================================================
if failures:
    st.divider()
    st.warning("Some symbols could not be priced (dashboard still runs on partial data).")
    st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)

st.divider()
st.caption("Data source: Yahoo Finance (with Google Sheet LivePrice_GS/PrevClose_GS fallback for snapshot). Educational project, not investment advice.")
