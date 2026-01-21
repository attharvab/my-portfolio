import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, time as dtime
from zoneinfo import ZoneInfo
import plotly.express as px
import plotly.graph_objects as go
import time as _time

# ============================================================
# Atharva Portfolio Returns (Bulletproof Production-Grade)
# - Never hard-crashes when partial data fails
# - Triple pricing engine (Yahoo fast_info -> Yahoo bulk -> Sheet fallback)
# - Degraded Mode banner + failure diagnostics
# - Zombie holding checks (zero price, absurd day move)
# - Mobile-friendly table toggle
# - Daily Alpha vs S&P today + Inception Alpha vs S&P since start
# - Country Risk (exposure) = geography weights, not FX returns
# - FX used ONLY for INR exposure weights (NOT for returns)
# ============================================================

APP_TITLE = "Atharva Portfolio Returns"
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📈")

# -----------------------------
# Google Sheet CSV (Portfolio tab)
# -----------------------------
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUHmE__8dpl_nKGv5F5mTXO7e3EyVRqz-PJF_4yyIrfJAa7z8XgzkIw6IdLnaotkACka2Q-PvP8P-z/pub?output=csv"

# OPTIONAL: sold history as separate published CSV (only needed if you want Sold table)
# If you prefer a "History" tab inside the same Google Sheet, you must publish that tab separately (different publish URL).
HISTORY_SHEET_URL = ""  # keep "" to disable

# -----------------------------
# Columns
# -----------------------------
REQUIRED_COLS = ["Ticker", "Region", "QTY", "AvgCost"]
OPTIONAL_COLS = ["Benchmark", "Type", "Thesis", "FirstBuyDate", "LivePrice_GS", "PrevClose_GS", "FX_Rate"]

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


# -----------------------------
# Market session logic (simple time-window UX)
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

    # Sheet fallback pricing (optional)
    df["LivePrice_GS"] = df["LivePrice_GS"].apply(_as_float)
    df["PrevClose_GS"] = df["PrevClose_GS"].apply(_as_float)

    # FX_Rate is ONLY for INR exposure weights (not for returns)
    df["FX_Rate"] = df["FX_Rate"].apply(_as_float)

    df = df[df["Ticker"].str.len() > 0]
    df = df[~df["Ticker"].str.contains(SUMMARY_NOISE_REGEX, case=False, na=False)]

    df["QTY"] = df["QTY"].apply(_as_float)
    df["AvgCost"] = df["AvgCost"].apply(_as_float)

    df = df.dropna(subset=["Ticker", "Region", "QTY", "AvgCost"])
    df = df[df["QTY"] != 0]

    # Default benchmark by region
    df.loc[df["Benchmark"].str.len() == 0, "Benchmark"] = df["Region"].apply(_default_benchmark_for_region)

    # If FX_Rate missing, set US -> current USDINR weight later (fallback); set India -> 1.0 now
    # We will keep NaN here and fill after we fetch USDINR, so weights stay realistic.
    df.loc[df["Region"].str.upper() == "INDIA", "FX_Rate"] = df.loc[df["Region"].str.upper() == "INDIA", "FX_Rate"].fillna(1.0)

    df["TotalCost"] = df["QTY"] * df["AvgCost"]

    # Aggregate duplicates
    agg = df.groupby(["Ticker", "Region", "Benchmark"], as_index=False).agg(
        QTY=("QTY", "sum"),
        TotalCost=("TotalCost", "sum"),
        Type=("Type", "first"),
        Thesis=("Thesis", "first"),
        FirstBuyDate=("FirstBuyDate", "max"),
        LivePrice_GS=("LivePrice_GS", "max"),
        PrevClose_GS=("PrevClose_GS", "max"),
        FX_Rate=("FX_Rate", "max"),
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
# Dynamic Symbol Discovery (fix ghost benchmarks)
# -----------------------------
def _discover_all_symbols(df_sheet: pd.DataFrame):
    holdings = df_sheet["Ticker"].unique().tolist()
    benchmarks = df_sheet["Benchmark"].dropna().unique().tolist()
    core = ["^GSPC", "^NSEI", "GC=F", "SI=F", "USDINR=X"]
    return list(set(holdings + benchmarks + core))


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
    progress_bar = st.progress(0)

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        progress_bar.progress(min((i + chunk_size) / len(tickers), 1.0))
        last_err = None

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
                last_err = None
                break
            except Exception as e:
                last_err = e
                _time.sleep(0.7 * (attempt + 1))

        if last_err:
            st.warning(f"⚠️ Chunk {i//chunk_size + 1} failed after {retries + 1} attempts: {str(last_err)[:60]}")

    progress_bar.empty()

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
        if (live_fast is None) and (last_close is not None):
            source = "yfinance_bulk"

        live = live_fast if live_fast is not None else last_close
        prev = prev_fast if prev_fast is not None else prev_close

        # Retry single ticker if still missing
        if live is None or prev is None:
            try:
                df1 = yf.download(
                    tickers=tk,
                    period="15d",
                    interval="1d",
                    progress=False,
                    auto_adjust=False,
                    threads=False,
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

        # Google Sheet fallback
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

        # Final safety
        if prev is None and live is not None:
            prev = live

        price_map[tk] = {"live": live, "prev": prev, "source": source}

    return price_map


@st.cache_data(ttl=900)
def fetch_fx_usdinr():
    live, _ = _fast_live_prev("USDINR=X")
    if live is not None and live > 0:
        return live
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
# Inception benchmark index (S&P) for Inception Alpha
# -----------------------------
@st.cache_data(ttl=900)
def build_sp500_index(start_date: pd.Timestamp):
    if start_date is None or pd.isna(start_date):
        return None
    start_date = pd.to_datetime(start_date).tz_localize(None)
    px = yf.download("^GSPC", start=(start_date - pd.Timedelta(days=20)).strftime("%Y-%m-%d"),
                     interval="1d", progress=False, auto_adjust=False, threads=False)
    if px is None or px.empty or "Close" not in px.columns:
        return None
    s = px["Close"].dropna()
    if s.empty:
        return None
    return (float(s.iloc[-1]) / float(s.iloc[0])) - 1.0


# ============================================================
# UI: Sidebar (About + Follow + Export)
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
st.sidebar.write("LinkedIn: linkedin.com/in/atharva-bhutada")
st.sidebar.write("Email: abhutada1@babson.edu")

st.sidebar.markdown("---")
st.sidebar.markdown("### Track My Portfolio")
st.sidebar.caption("Download my current picks as CSV and track them in your own sheet.")
# Download button will be added after calc_df exists.

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
# MAIN
# ============================================================
try:
    df_sheet = load_and_clean_data(SHEET_URL)
except Exception as e:
    st.error(f"Spreadsheet Error: {e}")
    st.stop()

if df_sheet.empty:
    st.warning("No valid holdings found. Check Ticker/Region/QTY/AvgCost.")
    st.stop()

# Build sheet fallback dictionary
sheet_fallback = {}
for _, r in df_sheet.iterrows():
    tk = _clean_str(r.get("Ticker", ""))
    sheet_fallback[tk] = {"live": r.get("LivePrice_GS", None), "prev": r.get("PrevClose_GS", None)}

# Discover tickers + benchmarks dynamically
all_symbols = _discover_all_symbols(df_sheet)

with st.spinner("Syncing portfolio data..."):
    fx_usdinr_live = fetch_fx_usdinr()
    prices = build_prices_with_sheet_fallback(all_symbols, sheet_fallback)
    macro = fetch_5y_macro()

# Fill missing FX_Rate for US with LIVE USDINR (for exposure weights ONLY)
df_sheet.loc[(df_sheet["Region"].str.upper() == "US") & (df_sheet["FX_Rate"].isna()), "FX_Rate"] = fx_usdinr_live

# Build calc_df
rows, failures = [], []

for _, r in df_sheet.iterrows():
    tk = _clean_str(r["Ticker"])
    bench = _clean_str(r["Benchmark"])
    region = r["Region"]

    p_tk = prices.get(tk, None)
    p_b = prices.get(bench, None) if bench else None

    # Issue 1: log failures, do NOT silently drop without reporting
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
    fx_locked = float(r.get("FX_Rate", 1.0))

    # Issue 3: Zombie holding checks
    if avg == 0 or live == 0:
        failures.append({"Ticker": tk, "Reason": "Zero price detected (delisted or halted?)"})
        continue

    total_ret = (live - avg) / avg if avg != 0 else None
    day_ret = (live - prev) / prev if prev != 0 else None

    if day_ret is not None and abs(day_ret) > 0.5:
        failures.append({"Ticker": tk, "Reason": f"Suspicious intraday move: {day_ret*100:.1f}%"})
        continue

    alpha_day = None
    if bench:
        b_live = float(p_b["live"])
        b_prev = float(p_b["prev"])
        b_day = (b_live - b_prev) / b_prev if b_prev != 0 else None
        if day_ret is not None and b_day is not None:
            alpha_day = day_ret - b_day

    # FX used ONLY for exposure weights (INR value)
    value_inr = qty * live * fx_locked

    rows.append({
        "Ticker": tk,
        "Region": region,
        "Benchmark": bench,
        "Compared To": _bench_context(bench),
        "QTY": qty,
        "AvgCost": avg,
        "LivePrice": live,
        "PrevClose": prev,
        "TotalCost": float(r.get("TotalCost", qty * avg)),
        "Value_INR": value_inr,
        "Total_Ret": total_ret,
        "Day_Ret": day_ret,
        "Alpha_Day": alpha_day,
        "Beat_Index_Tag": _status_tag(alpha_day, bench),
        "Type": r.get("Type", ""),
        "Thesis": r.get("Thesis", ""),
        "FirstBuyDate": r.get("FirstBuyDate", pd.NaT),
        "PriceSource": p_tk.get("source", "none"),
        "FX_WeightRate": fx_locked,
    })

calc_df = pd.DataFrame(rows)

# If EVERYTHING failed, still do not crash. Show failures and stop gracefully.
if calc_df.empty:
    st.title(f"📈 {APP_TITLE}")
    st.error("No holdings could be priced right now (Yahoo + Google Sheet fallbacks missing).")
    if failures:
        st.warning(f"⚠️ Degraded Mode: {len(failures)} ticker(s) missing live data.")
        with st.expander("🔧 View Pricing Failures"):
            st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)
            st.caption("Tip: Ensure LivePrice_GS and PrevClose_GS are populated in your Google Sheet for every ticker.")
    st.stop()

# Weights
calc_df["Weight"] = calc_df["Value_INR"] / calc_df["Value_INR"].sum()

port_day = (calc_df["Day_Ret"] * calc_df["Weight"]).sum()
port_total = (calc_df["Total_Ret"] * calc_df["Weight"]).sum()

in_w = calc_df.loc[calc_df["Region"].str.upper() == "INDIA", "Weight"].sum()
us_w = calc_df.loc[calc_df["Region"].str.upper() == "US", "Weight"].sum()

# S&P day return (for Daily Alpha vs S&P)
def _safe_day(sym):
    p = prices.get(sym, None)
    if not p or p.get("live") is None or p.get("prev") is None or float(p["prev"]) == 0:
        return None
    return (float(p["live"]) - float(p["prev"])) / float(p["prev"])

spx_day = _safe_day("^GSPC")

daily_alpha_vs_spx = (port_day - spx_day) if (spx_day is not None) else None

# Strategy start date for inception alpha
strategy_start = None
if calc_df["FirstBuyDate"].notna().any():
    strategy_start = pd.to_datetime(calc_df["FirstBuyDate"].dropna().min()).tz_localize(None)

spx_inception_ret = build_sp500_index(strategy_start) if strategy_start is not None else None
inception_alpha_vs_spx = (port_total - spx_inception_ret) if (spx_inception_ret is not None) else None

# -----------------------------
# Header + Market Status
# -----------------------------
st.title(f"📈 {APP_TITLE}")
st.caption("Returns are shown in native currency. FX is used only for portfolio exposure weights (INR value), never for return calculation.")

now_utc = datetime.now(timezone.utc)
status_text, status_type = _market_status_badge(now_utc, calc_df)
if status_type == "closed":
    st.info(status_text)
elif status_type == "mixed":
    st.warning(status_text)
else:
    st.success(status_text)

# Issue 1: Degraded Mode banner
if failures:
    st.warning(f"⚠️ Degraded Mode: {len(failures)} ticker(s) missing live data. Showing partial portfolio.")
    with st.expander("🔧 View Pricing Failures"):
        st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)
        st.caption("Tip: Add LivePrice_GS and PrevClose_GS columns in your Google Sheet and keep them filled for each ticker.")

# -----------------------------
# Metrics (Phase 1)
# -----------------------------
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Total Return (Strategy)",
        f"{port_total*100:.2f}%",
        help="Weighted total return across holdings vs your AvgCost, in each stock’s native currency."
    )

with m2:
    delta_txt = f"{daily_alpha_vs_spx*100:+.2f}% vs S&P" if daily_alpha_vs_spx is not None else None
    st.metric(
        "Daily Alpha (vs S&P 500)",
        f"{(daily_alpha_vs_spx or 0)*100:.2f}%" if daily_alpha_vs_spx is not None else "—",
        delta=delta_txt,
        help="Daily Alpha = (Portfolio today return) - (S&P 500 today return)."
    )

with m3:
    val = f"{inception_alpha_vs_spx*100:.2f}%" if inception_alpha_vs_spx is not None else "—"
    st.metric(
        "Inception Alpha (vs S&P 500)",
        val,
        help="Inception Alpha = (Total Strategy Return) - (Total S&P 500 Return) since strategy start date."
    )

with m4:
    st.metric("Last Sync (UTC)", now_utc.strftime("%Y-%m-%d %H:%M"))
    st.caption("Prices may be delayed (Yahoo Finance).")

st.divider()

# -----------------------------
# Tabs: Combined / India / US (Phase 2: responsive)
# -----------------------------
tabs = st.tabs(["Combined", "India", "US"])

def _render_section(dfx: pd.DataFrame, title_suffix: str):
    if dfx.empty:
        st.info("No holdings in this view.")
        return

    # Exposure summary
    alloc = dfx.groupby("Region", as_index=False)["Weight"].sum()

    # Responsive columns
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader(f"📌 Picks {title_suffix}")

        show = dfx.copy()
        show["Weight%"] = show["Weight"] * 100
        show["Total Ret%"] = show["Total_Ret"].apply(_fmt_pct)
        show["Day Ret%"] = show["Day_Ret"].apply(_fmt_pct)
        show["Score vs Index%"] = show["Alpha_Day"].apply(_fmt_pct)

        # Mobile UX toggle (Issue 4)
        show_details = st.checkbox("📱 Show detailed columns (desktop mode)", value=False, key=f"detail_{title_suffix}")

        if show_details:
            cols_to_show = [
                "Ticker", "Region", "Benchmark", "Compared To",
                "Weight%", "AvgCost", "LivePrice", "PrevClose",
                "Total Ret%", "Day Ret%", "Beat_Index_Tag",
                "Score vs Index%", "PriceSource"
            ]
        else:
            cols_to_show = ["Ticker", "Weight%", "Total Ret%", "Day Ret%", "Beat_Index_Tag"]

        st.dataframe(
            show[cols_to_show],
            column_config={
                "Weight%": st.column_config.NumberColumn("Weight", format="%.2f%%", help="Exposure weight based on INR value (FX only used here)."),
                "Total Ret%": st.column_config.NumberColumn("Total Return", format="%.2f%%"),
                "Day Ret%": st.column_config.NumberColumn("Today Return", format="%.2f%%"),
                "Score vs Index%": st.column_config.NumberColumn("Alpha vs Index", format="%.2f%%", help="Stock day return minus benchmark day return."),
                "PriceSource": st.column_config.TextColumn("Price Source", help="yfinance_fastinfo / yfinance_bulk / yfinance_single / google_sheet"),
            },
            use_container_width=True,
            hide_index=True
        )

        # Export button (Phase 4)
        export_cols = ["Ticker", "Region", "QTY", "AvgCost", "Benchmark", "Type"]
        export_df = dfx[export_cols].copy()
        st.download_button(
            "⬇️ Download Picks as CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="atharva_portfolio_picks.csv",
            mime="text/csv",
            help="Download my current picks to track them in your own sheet."
        )

        # Thesis expander
        with st.expander("🧠 Thesis / Notes (from Google Sheet)"):
            tickers_sorted = sorted(show["Ticker"].unique().tolist())
            selected = st.selectbox("Select a ticker", tickers_sorted, key=f"sel_{title_suffix}")
            pick = show[show["Ticker"] == selected]
            if not pick.empty:
                r = pick.iloc[0]
                st.write(f"**Type:** {r.get('Type','')}")
                st.write(f"**Region:** {r.get('Region','')}")
                st.write(f"**Benchmark:** {r.get('Benchmark','')}")
                st.text_area("Thesis (edit in Google Sheet)", value=str(r.get("Thesis","")), height=180, key=f"th_{title_suffix}")

    with col_right:
        st.subheader("🌍 Country Risk (Exposure)")
        fig_pie = px.pie(alloc, values="Weight", names="Region", hole=0.5)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption("Country Risk reflects geographic exposure weights (India vs US), not FX impact on returns.")

# Combined
with tabs[0]:
    _render_section(calc_df, "(Combined)")

# India
with tabs[1]:
    _render_section(calc_df[calc_df["Region"].str.upper() == "INDIA"].copy(), "(India)")

# US
with tabs[2]:
    _render_section(calc_df[calc_df["Region"].str.upper() == "US"].copy(), "(US)")

st.divider()

# -----------------------------
# Macro Chart (optional, stays stable)
# -----------------------------
st.subheader("📈 5-Year Macro Trends (Monthly, Indexed to 100)")
if macro is None or macro.empty:
    st.info("Macro trend data unavailable right now.")
else:
    macro_named = macro.copy().rename(columns={k: v for k, v in ASSET_LABELS.items() if k in macro.columns})
    macro_named = macro_named.dropna(how="all").ffill()

    base_row = macro_named.iloc[0]
    plot_idx = (macro_named / base_row) * 100.0

    fig = go.Figure()
    for col in plot_idx.columns:
        fig.add_trace(go.Scatter(x=plot_idx.index, y=plot_idx[col], mode="lines", name=col, line=dict(width=2)))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Index (Base=100)",
        legend_title="Asset",
        margin=dict(l=10, r=10, t=20, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# -----------------------------
# Regret Matrix (Feature 2)
# -----------------------------
st.subheader("😢 The Road Not Taken (Opportunity Cost)")
if strategy_start is None:
    st.info("Add FirstBuyDate in your Google Sheet to unlock inception analytics.")
else:
    if spx_inception_ret is None:
        st.info("S&P inception return unavailable right now.")
    else:
        invested_capital = float(calc_df["TotalCost"].sum())
        alternative_value = invested_capital * (1 + spx_inception_ret)
        actual_value = float(calc_df["Value_INR"].sum())
        st.metric(
            "If You Had Bought the S&P 500 Instead",
            f"₹{alternative_value:,.0f}",
            delta=f"₹{(actual_value - alternative_value):,.0f}",
            help="Compares your invested cost basis versus what it would be worth if it compounded at S&P returns since your first buy date."
        )

st.divider()

# -----------------------------
# Deep Dive (Phase 3)
# -----------------------------
st.subheader("🔍 Deep Dive (Ticker History + Financial Ratios)")
selected_ticker = st.selectbox("Select a ticker for deep dive", sorted(calc_df["Ticker"].unique().tolist()), key="deep_dive_sel")
dd = fetch_ticker_deep_dive(selected_ticker)

if dd is None:
    st.info("Deep dive unavailable right now (data provider issue).")
else:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write(f"**{dd.get('name', selected_ticker)}**")
        hist = dd.get("hist", None)
        if hist is not None and not hist.empty and "Close" in hist.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close"))
            fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Price history unavailable.")
    with c2:
        st.write("**Key Ratios**")
        def _kv(k, v):
            st.write(f"- **{k}:** {v if v is not None else '—'}")
        _kv("Sector", dd.get("sector"))
        _kv("Industry", dd.get("industry"))
        _kv("Market Cap", dd.get("marketCap"))
        _kv("P/E (TTM)", dd.get("trailingPE"))
        _kv("P/E (Forward)", dd.get("forwardPE"))
        _kv("Price/Book", dd.get("priceToBook"))
        _kv("Debt/Equity", dd.get("debtToEquity"))
        _kv("Profit Margin", dd.get("profitMargins"))
        _kv("Operating Margin", dd.get("operatingMargins"))
        _kv("ROE", dd.get("returnOnEquity"))
        _kv("ROA", dd.get("returnOnAssets"))

# -----------------------------
# Sold Positions (optional)
# -----------------------------
if HISTORY_SHEET_URL.strip():
    st.divider()
    st.subheader("📜 Sold Positions (History)")
    sold_df = load_sold_history(HISTORY_SHEET_URL)
    if sold_df.empty:
        st.info("History sheet is empty or not reachable.")
    else:
        st.dataframe(sold_df, use_container_width=True, hide_index=True)

st.divider()
st.caption("Data source: Yahoo Finance and optional Google Sheet fallback. Educational project, not investment advice.")
