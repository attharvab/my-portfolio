# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, time as dtime
from zoneinfo import ZoneInfo
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import calendar as _cal
import time as _time

APP_TITLE = "Atharva Portfolio Returns"
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📈")

# ========= YOUR SHEET URLS =========
# Holdings (Current portfolio snapshot)
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUHmE__8dpl_nKGv5F5mTXO7e3EyVRqz-PJF_4yyIrfJAa7z8XgzkIw6IdLnaotkACka2Q-PvP8P-z/pub?output=csv"

# Transactions Ledger (Published CSV)
# MUST contain at least: Ticker, Date, QTY
# Optional but recommended: Type (BUY/SELL), Buy Price (or Trade Price/Price)
TRANSACTIONS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR-OybDEJRMpK5jvtLnMq3SOze-ZwT6hVY07w4nAnKfn1dva_E68fKSZQkn0yvzDhk217HEQ7xis77G/pub?output=csv"

# =============================
# Columns & config
# =============================
REQUIRED_COLS = ["Ticker", "Region", "QTY", "AvgCost"]
OPTIONAL_COLS = ["Benchmark", "Type", "Thesis", "FirstBuyDate", "LivePrice_GS", "PrevClose_GS"]

SUMMARY_NOISE_REGEX = r"TOTAL|PORTFOLIO|SUMMARY|CASH"
DEFAULT_BENCH = {"us": "^GSPC", "india": "^NSEI"}

# Used only for labels in the older macro chart (not required for alpha/calendar)
MACRO_ASSETS = ["^GSPC", "^NSEI", "GC=F", "SI=F"]
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
    return (not _is_india_ticker(t))

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
# Transactions loader (robust, exact labels, no guessing)
# ============================================================
@st.cache_data(ttl=300)
def load_transactions(url: str) -> pd.DataFrame:
    if not url or str(url).strip() == "":
        return pd.DataFrame()

    df = pd.read_csv(url)
    if df is None or df.empty:
        return pd.DataFrame()

    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")].copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()

    required_txn_cols = ["Ticker", "Date", "QTY"]
    for col in required_txn_cols:
        if col not in df.columns:
            raise ValueError(f"Transactions CSV missing required column: {col}")

    # Normalize types
    out = pd.DataFrame()
    out["Ticker"] = df["Ticker"].astype(str).str.strip()
    out["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    out["QTY"] = pd.to_numeric(df["QTY"], errors="coerce")

    # Optional
    out["Type"] = df["Type"].astype(str).str.strip() if "Type" in df.columns else ""
    out["Region"] = df["Region"].astype(str).str.strip() if "Region" in df.columns else ""
    if "FX_Rate" in df.columns:
        out["FX_Rate"] = pd.to_numeric(df["FX_Rate"], errors="coerce")
    else:
        out["FX_Rate"] = np.nan

    # Try to keep a trade price column if present
    price_cols = ["Buy Price", "Trade Price", "Price", "Avg Price", "Execution Price", "Fill Price"]
    found_price = None
    for c in price_cols:
        if c in df.columns:
            found_price = c
            break
    if found_price:
        out["TradePrice"] = pd.to_numeric(df[found_price], errors="coerce")
    else:
        out["TradePrice"] = np.nan

    out = out.dropna(subset=["Date", "Ticker", "QTY"]).copy()
    out = out[out["Ticker"].str.len() > 0]
    out = out[out["QTY"] != 0]

    # Infer Region if missing
    def infer_region(tk):
        return "India" if _is_india_ticker(tk) else "US"

    if "Region" in out.columns:
        mask = out["Region"].astype(str).str.strip().eq("")
        if mask.any():
            out.loc[mask, "Region"] = out.loc[mask, "Ticker"].apply(infer_region)
    else:
        out["Region"] = out["Ticker"].apply(infer_region)

    out = out.sort_values("Date").reset_index(drop=True)
    return out

# ============================================================
# Pricing Engine (Yahoo primary + Sheet fallback for snapshot)
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
        if fx is not None and not fx.empty and "Close" in fx.columns:
            s = fx["Close"].dropna()
            if not s.empty:
                v = float(s.iloc[-1])
                return v if v > 0 else 83.0
    except Exception:
        pass
    return 83.0

# ============================================================
# TWR Alpha (regional split, correct cashflow handling)
# ============================================================
def _infer_side_sign(row):
    """
    +1 BUY, -1 SELL
    Priority:
      1) Type column (BUY/SELL)
      2) Sign of QTY (negative means SELL)
      3) Default BUY
    """
    t = str(row.get("Type", "")).strip().upper()
    if t in ["BUY", "B", "LONG"]:
        return 1
    if t in ["SELL", "S", "SHORT"]:
        return -1

    try:
        q = float(row.get("QTY", row.get("Qty", 0.0)))
        if q < 0:
            return -1
    except Exception:
        pass
    return 1

@st.cache_data(ttl=1800)
def build_regional_twr_alpha(txn_df: pd.DataFrame, region: str, benchmark: str, years_back: int = 4):
    """
    Returns: alpha, port_idx, bench_idx, daily_twr_returns
    - India: compares to ^NSEI, currency INR
    - US: compares to ^GSPC, currency USD
    """
    if txn_df is None or txn_df.empty:
        return None, None, None, None

    region = region.strip().upper()
    d = txn_df.copy()

    d["Ticker"] = d["Ticker"].astype(str).str.strip()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce").dt.normalize()
    d["QTY"] = pd.to_numeric(d["QTY"], errors="coerce")
    d = d.dropna(subset=["Date", "Ticker", "QTY"]).copy()
    d = d[d["QTY"] != 0].copy()

    # Filter region by ticker (more reliable than user-entered Region)
    if region == "INDIA":
        d = d[d["Ticker"].apply(_is_india_ticker)].copy()
    else:
        d = d[~d["Ticker"].apply(_is_india_ticker)].copy()

    if d.empty:
        return None, None, None, None

    # Use last N years window (but keep earlier transactions if needed for positions)
    end = pd.Timestamp.utcnow().normalize()
    cutoff = end - pd.DateOffset(years=years_back)
    # We still need earlier txns to build correct positions at cutoff:
    d_all = d.copy()
    d_window = d[d["Date"] >= cutoff].copy()
    if d_window.empty:
        # still compute on all
        d_window = d.copy()
        cutoff = d_window["Date"].min()

    tickers = sorted(d_all["Ticker"].unique().tolist())
    start = pd.to_datetime(cutoff) - pd.Timedelta(days=10)
    end_dl = end + pd.Timedelta(days=1)

    px = yf.download(
        tickers=tickers + [benchmark],
        start=start.strftime("%Y-%m-%d"),
        end=end_dl.strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False
    )

    if px is None or px.empty or "Close" not in px.columns:
        return None, None, None, None

    close = px["Close"].copy()
    if isinstance(close, pd.Series):
        close = close.to_frame()

    close = close.dropna(how="all").ffill()
    if close.empty or benchmark not in close.columns:
        return None, None, None, None

    # Build signed delta shares
    d_all["Type"] = d_all.get("Type", "")
    deltas = []
    for _, row in d_all.iterrows():
        sign = _infer_side_sign(row)
        qty = abs(float(row["QTY"]))
        deltas.append(sign * qty)
    d_all["DeltaShares"] = deltas

    # Trade price for cashflows: prefer TradePrice column, else close on trade date
    if "TradePrice" in d_all.columns:
        d_all["TradePrice"] = pd.to_numeric(d_all["TradePrice"], errors="coerce")
    else:
        d_all["TradePrice"] = np.nan

    def _row_cf(row):
        sign = _infer_side_sign(row)  # +1 buy, -1 sell
        qty = abs(float(row["QTY"]))
        dt = row["Date"]
        tk = row["Ticker"]

        p = row.get("TradePrice", np.nan)
        if pd.isna(p):
            if (dt in close.index) and (tk in close.columns):
                p = close.loc[dt, tk]
        if pd.isna(p):
            return np.nan

        amt = qty * float(p)
        # BUY is deposit (positive), SELL is withdrawal (negative)
        return amt if sign == 1 else -amt

    d_all["CF"] = d_all.apply(_row_cf, axis=1)
    cf_daily = d_all.groupby("Date")["CF"].sum().sort_index()

    # Positions matrix for tickers (exclude benchmark)
    days = close.index
    pos_cols = [c for c in close.columns if c != benchmark]
    pos = pd.DataFrame(0.0, index=days, columns=pos_cols)

    delta_tbl = d_all.groupby(["Date", "Ticker"])["DeltaShares"].sum().reset_index()

    for tk in pos_cols:
        s = delta_tbl[delta_tbl["Ticker"] == tk].set_index("Date")["DeltaShares"]
        s = s.reindex(days).fillna(0.0)
        pos[tk] = s.cumsum()

    pos = pos.clip(lower=0.0)

    # Sleeve value
    sleeve_prices = close[pos_cols].copy()
    sleeve_val = (pos * sleeve_prices).sum(axis=1)
    sleeve_val = sleeve_val.replace([np.inf, -np.inf], np.nan).dropna()

    # Restrict to the window for performance
    sleeve_val = sleeve_val[sleeve_val.index >= start].copy()
    cf = cf_daily.reindex(sleeve_val.index).fillna(0.0)

    v_prev = sleeve_val.shift(1)
    valid = (v_prev > 0) & sleeve_val.notna()
    sleeve_val = sleeve_val[valid]
    cf = cf[valid]
    v_prev = v_prev[valid]

    if sleeve_val.empty:
        return None, None, None, None

    twr = (sleeve_val - v_prev - cf) / v_prev
    twr = twr.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Build indices
    port_idx = (1.0 + twr).cumprod() * 100.0
    port_idx.name = f"{region.title()} Strategy (TWR, Base=100)"

    b = close[benchmark].reindex(port_idx.index).ffill().dropna()
    if b.empty:
        return None, None, None, None

    bench_idx = (b / float(b.iloc[0])) * 100.0
    bench_idx.name = f"{benchmark} (Base=100)"

    alpha = (float(port_idx.iloc[-1]) / 100.0 - 1.0) - (float(bench_idx.iloc[-1]) / 100.0 - 1.0)
    return float(alpha), port_idx, bench_idx, twr

# ============================================================
# Portfolio TWR daily returns in INR for Webull Calendar (optional)
# - Converts US tickers to INR using USDINR=X, keeps India in INR
# - Uses cashflow-adjusted TWR logic so buys do not create fake gains
# ============================================================
@st.cache_data(ttl=1800)
def build_total_portfolio_twr_daily_returns_inr(txn_df: pd.DataFrame, years_back: int = 4):
    if txn_df is None or txn_df.empty:
        return None

    d = txn_df.copy()
    d["Ticker"] = d["Ticker"].astype(str).str.strip()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce").dt.normalize()
    d["QTY"] = pd.to_numeric(d["QTY"], errors="coerce")
    d = d.dropna(subset=["Date", "Ticker", "QTY"]).copy()
    d = d[d["QTY"] != 0].copy()
    if d.empty:
        return None

    end = pd.Timestamp.utcnow().normalize()
    cutoff = end - pd.DateOffset(years=years_back)

    # Keep earlier txns for correct positions at cutoff, but performance window is cutoff
    tickers = sorted(d["Ticker"].unique().tolist())
    start = pd.to_datetime(cutoff) - pd.Timedelta(days=10)
    end_dl = end + pd.Timedelta(days=1)

    need = tickers + ["USDINR=X"]
    px = yf.download(
        tickers=need,
        start=start.strftime("%Y-%m-%d"),
        end=end_dl.strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False
    )
    if px is None or px.empty or "Close" not in px.columns:
        return None

    close = px["Close"].copy()
    if isinstance(close, pd.Series):
        close = close.to_frame()

    close = close.dropna(how="all").ffill()
    if close.empty or "USDINR=X" not in close.columns:
        return None

    fx = close["USDINR=X"].dropna().ffill()

    # Convert all ticker prices into INR:
    close_inr = close.copy()
    for tk in tickers:
        if tk not in close_inr.columns:
            continue
        if _is_us_ticker(tk) and tk not in ["GC=F", "SI=F"]:
            close_inr[tk] = close_inr[tk] * fx.reindex(close_inr.index).ffill()

    # Signed delta shares
    d["Type"] = d.get("Type", "")
    deltas = []
    for _, row in d.iterrows():
        sign = _infer_side_sign(row)
        qty = abs(float(row["QTY"]))
        deltas.append(sign * qty)
    d["DeltaShares"] = deltas

    # Cashflows in INR (BUY deposit +, SELL withdrawal -)
    if "TradePrice" in d.columns:
        d["TradePrice"] = pd.to_numeric(d["TradePrice"], errors="coerce")
    else:
        d["TradePrice"] = np.nan

    def _row_cf_inr(row):
        sign = _infer_side_sign(row)
        qty = abs(float(row["QTY"]))
        dt = row["Date"]
        tk = row["Ticker"]

        p = row.get("TradePrice", np.nan)
        if pd.isna(p):
            if (dt in close_inr.index) and (tk in close_inr.columns):
                p = close_inr.loc[dt, tk]
        else:
            # if tradeprice was recorded in native currency, convert US tradeprice to INR
            if _is_us_ticker(tk) and tk not in ["GC=F", "SI=F"]:
                if dt in fx.index and pd.notna(fx.loc[dt]):
                    p = float(p) * float(fx.loc[dt])

        if pd.isna(p):
            return np.nan

        amt = qty * float(p)
        return amt if sign == 1 else -amt

    d["CF_INR"] = d.apply(_row_cf_inr, axis=1)
    cf_daily = d.groupby("Date")["CF_INR"].sum().sort_index()

    # Build positions
    days = close_inr.index
    pos = pd.DataFrame(0.0, index=days, columns=[t for t in tickers if t in close_inr.columns])

    delta_tbl = d.groupby(["Date", "Ticker"])["DeltaShares"].sum().reset_index()
    for tk in pos.columns:
        s = delta_tbl[delta_tbl["Ticker"] == tk].set_index("Date")["DeltaShares"]
        s = s.reindex(days).fillna(0.0)
        pos[tk] = s.cumsum()

    pos = pos.clip(lower=0.0)

    # Portfolio value (INR) and TWR returns
    prices_inr = close_inr[pos.columns].copy()
    v = (pos * prices_inr).sum(axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    v = v[v.index >= start].copy()

    cf = cf_daily.reindex(v.index).fillna(0.0)
    v_prev = v.shift(1)
    valid = (v_prev > 0) & v.notna()
    v = v[valid]
    cf = cf[valid]
    v_prev = v_prev[valid]
    if v.empty:
        return None

    twr = (v - v_prev - cf) / v_prev
    twr = twr.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Restrict to last N years (true window)
    twr = twr[twr.index >= cutoff]
    if twr.empty:
        return None
    return twr

# ============================================================
# Webull-style P&L Calendar
# ============================================================
def build_calendar_grid(daily_returns: pd.Series, year: int, month: int):
    idx = pd.to_datetime(daily_returns.index).tz_localize(None)
    s = pd.Series(daily_returns.values, index=idx).sort_index()

    month_start = pd.Timestamp(year=year, month=month, day=1)
    month_end = month_start + pd.offsets.MonthEnd(1)

    sm = s[(s.index >= month_start) & (s.index <= month_end)]

    cal = _cal.Calendar(firstweekday=0)  # Monday
    weeks = cal.monthdatescalendar(year, month)

    y_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    w = len(weeks)
    z = np.full((7, w), np.nan, dtype=float)
    text = np.full((7, w), "", dtype=object)

    for wi, week in enumerate(weeks):
        for di, day in enumerate(week):
            if day.month != month:
                continue
            dts = pd.Timestamp(day)
            ret = sm.get(dts, np.nan)
            z[di, wi] = ret if pd.notna(ret) else np.nan
            if pd.isna(ret):
                text[di, wi] = f"{day.day}"
            else:
                text[di, wi] = f"{day.day}<br>{ret*100:.2f}%"

    x_labels = [f"W{(i+1)}" for i in range(w)]
    return z, text, x_labels, y_labels

def render_webull_calendar(daily_returns: pd.Series, title_prefix="Webull-Style P&L Calendar"):
    if daily_returns is None or daily_returns.empty:
        st.info("Calendar unavailable (not enough daily return data).")
        return

    idx = pd.to_datetime(daily_returns.index).tz_localize(None)
    min_dt, max_dt = idx.min(), idx.max()

    years = list(range(min_dt.year, max_dt.year + 1))
    sel_year = st.selectbox("Year", years, index=len(years) - 1, key=f"cal_year_{title_prefix}")

    months = list(range(1, 13))
    if sel_year == min_dt.year:
        months = [m for m in months if m >= min_dt.month]
    if sel_year == max_dt.year:
        months = [m for m in months if m <= max_dt.month]

    month_names = {m: _cal.month_name[m] for m in months}
    sel_month_name = st.selectbox("Month", [month_names[m] for m in months], index=len(months) - 1, key=f"cal_month_{title_prefix}")
    sel_month = [m for m in months if month_names[m] == sel_month_name][0]

    z, text, x_labels, y_labels = build_calendar_grid(daily_returns, sel_year, sel_month)

    finite = np.isfinite(z)
    max_abs = float(np.nanmax(np.abs(z[finite]))) if finite.any() else 0.01
    max_abs = max(max_abs, 0.01)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 12},
            colorscale=[
                [0.0, "#d62728"],   # red
                [0.5, "#ffffff"],   # white at 0
                [1.0, "#2ca02c"],   # green
            ],
            zmin=-max_abs,
            zmax=max_abs,
            hovertemplate="%{text}<extra></extra>",
            showscale=True,
            colorbar=dict(title="Daily P&L %", tickformat=".1%")
        )
    )

    fig.update_layout(
        title=f"{title_prefix} ({_cal.month_name[sel_month]} {sel_year})",
        margin=dict(l=10, r=10, t=50, b=10),
        height=520,
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")

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

# Build Google Sheet fallback dict
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

# ============================================================
# Compute per-holding snapshot metrics (table + weights)
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

    # exposure weights: convert US to INR only for weight sizing
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

# ============================================================
# Header + status
# ============================================================
st.title("📈 Atharva Portfolio Returns")
st.markdown("This dashboard tracks my portfolio performance and alpha vs benchmarks.")
st.caption("Snapshot metrics use Yahoo Finance with Google Sheet fallback. Regional alpha uses cashflow-adjusted Time-Weighted Return (TWR) from your transactions ledger.")

now_utc = datetime.now(timezone.utc)
status_text, status_type = _market_status_badge(now_utc, calc_df)
if status_type == "closed":
    st.info(status_text)
elif status_type == "mixed":
    st.warning(status_text)
else:
    st.success(status_text)

if calc_df.empty:
    st.error("Pricing is temporarily unavailable. Please refresh shortly.")
    if failures:
        st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)
    st.stop()

# ============================================================
# Weights + portfolio daily snapshot metrics
# ============================================================
calc_df = calc_df[calc_df["QTY"] > 0].copy()
den = calc_df["Value_INR"].sum()
calc_df["Weight"] = (calc_df["Value_INR"] / den) if den and den > 0 else 0.0

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
# Load transactions + compute regional alpha + TWR returns for calendar
# ============================================================
txn_df = pd.DataFrame()
ledger_warning = None
india_alpha = us_alpha = None
india_idx = nifty_idx = None
us_idx = spx_idx = None
calendar_twr_inr = None

if TRANSACTIONS_URL and str(TRANSACTIONS_URL).strip():
    try:
        txn_df = load_transactions(TRANSACTIONS_URL)
        if txn_df is None or txn_df.empty:
            ledger_warning = "Transactions ledger loaded but empty."
        else:
            st.sidebar.success(f"✅ Loaded {len(txn_df)} transactions ({txn_df['Date'].min().date()} → {txn_df['Date'].max().date()})")
            st.sidebar.write(f"Unique tickers: {txn_df['Ticker'].nunique()}")

            india_alpha, india_idx, nifty_idx, india_twr = build_regional_twr_alpha(txn_df, region="India", benchmark="^NSEI", years_back=4)
            us_alpha, us_idx, spx_idx, us_twr = build_regional_twr_alpha(txn_df, region="US", benchmark="^GSPC", years_back=4)

            # Webull calendar uses total portfolio TWR in INR (cashflow-adjusted)
            calendar_twr_inr = build_total_portfolio_twr_daily_returns_inr(txn_df, years_back=4)

    except Exception as e:
        ledger_warning = f"Transactions ledger issue: {e}"
else:
    ledger_warning = "Transactions URL is not set."

if ledger_warning:
    st.warning(ledger_warning)

# ============================================================
# Top metrics row
# ============================================================
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(_tooltip("**Total Return (Strategy)**", "Weighted return vs AvgCost using current exposure weights (snapshot)."), unsafe_allow_html=True)
    st.metric(label="", value=f"{port_total*100:.2f}%")

with m2:
    st.markdown(_tooltip("**Today Return (Portfolio)**", "Weighted daily return using PrevClose vs LivePrice (snapshot)."), unsafe_allow_html=True)
    st.metric(label="", value=f"{port_day*100:.2f}%")

with m3:
    st.markdown(_tooltip("**Daily Alpha (vs S&P)**", "Portfolio Today Return minus S&P 500 Today Return (snapshot)."), unsafe_allow_html=True)
    st.metric(label="", value="—" if daily_alpha_vs_spx is None else f"{daily_alpha_vs_spx*100:.2f}%")

with m4:
    st.markdown(_tooltip("**Regional Alpha Ready**", "See Regional Alpha section below: India vs Nifty and US vs S&P using cashflow-adjusted TWR."), unsafe_allow_html=True)
    st.metric(label="", value="Split below")

st.caption(_tooltip("Last Sync (UTC)", "Pricing uses Yahoo Finance with Google Sheet LivePrice_GS/PrevClose_GS fallback for snapshot."), unsafe_allow_html=True)
st.write(now_utc.strftime("%Y-%m-%d %H:%M"))

# ============================================================
# Regional Alpha (split: India vs Nifty, US vs S&P)
# ============================================================
st.divider()
st.subheader("📍 Regional Alpha (4Y, cashflow-adjusted TWR)")

c1, c2 = st.columns(2)
with c1:
    st.markdown(_tooltip("**India Strategy vs Nifty 50**", "Uses your transactions ledger. Buys treated as deposits, sells as withdrawals. Benchmark is ^NSEI in INR."), unsafe_allow_html=True)
    st.metric(label="", value="—" if india_alpha is None else f"{india_alpha*100:.2f}%")

with c2:
    st.markdown(_tooltip("**US Strategy vs S&P 500**", "Uses your transactions ledger. Buys treated as deposits, sells as withdrawals. Benchmark is ^GSPC in USD."), unsafe_allow_html=True)
    st.metric(label="", value="—" if us_alpha is None else f"{us_alpha*100:.2f}%")

# ============================================================
# Tabs
# ============================================================
st.divider()
tabs = st.tabs(["Combined", "India", "US"])

def _filter_region(df, region_name):
    return df[df["Region"].str.upper() == region_name.upper()].copy()

with tabs[0]:
    left, right = st.columns([2, 1])

    with left:
        st.subheader("🟩🟥 Webull Calendar (Portfolio Daily P&L %, 4Y)")
        st.caption("This uses cashflow-adjusted Time-Weighted Return (TWR) in INR so buys do not create fake green days.")
        if calendar_twr_inr is None or (isinstance(calendar_twr_inr, pd.Series) and calendar_twr_inr.empty):
            st.info("Calendar unavailable (could not build daily TWR returns from ledger).")
        else:
            render_webull_calendar(calendar_twr_inr, title_prefix="Webull-Style P&L Calendar (TWR, INR)")

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
    if india_idx is not None and nifty_idx is not None:
        st.caption("India TWR index vs Nifty (Base=100).")
        m = pd.concat([india_idx, nifty_idx], axis=1).dropna()
        if not m.empty:
            fig = go.Figure()
            for col in m.columns:
                fig.add_trace(go.Scatter(x=m.index, y=m[col], mode="lines", name=col))
            fig.update_layout(title="India Strategy vs Nifty (TWR, Base=100)", yaxis_title="Index", height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("🇺🇸 US View")
    us_df = _filter_region(calc_df, "US")
    if us_df.empty:
        st.info("No US holdings right now.")
    else:
        st.write(f"Holdings: {len(us_df)} | Live exposure weight: {us_df['Weight'].sum()*100:.2f}%")
    if us_idx is not None and spx_idx is not None:
        st.caption("US TWR index vs S&P 500 (Base=100).")
        m = pd.concat([us_idx, spx_idx], axis=1).dropna()
        if not m.empty:
            fig = go.Figure()
            for col in m.columns:
                fig.add_trace(go.Scatter(x=m.index, y=m[col], mode="lines", name=col))
            fig.update_layout(title="US Strategy vs S&P 500 (TWR, Base=100)", yaxis_title="Index", height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Picks table
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
# Failures (diagnostic)
# ============================================================
if failures:
    st.divider()
    st.warning("Some symbols could not be priced (dashboard still runs on partial data).")
    st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)

st.divider()
st.caption("Data source: Yahoo Finance (with Google Sheet fallback for snapshot). Educational project, not investment advice.")
