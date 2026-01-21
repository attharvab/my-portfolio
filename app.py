import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, time as dtime
from zoneinfo import ZoneInfo
import plotly.graph_objects as go
import time as _time

# ============================================================
# Atharva Portfolio Returns (Bulletproof Production-Grade)
# ============================================================

APP_TITLE = "Atharva Portfolio Returns"
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📈")

SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUHmE__8dpl_nKGv5F5mTXO7e3EyVRqz-PJF_4yyIrfJAa7z8XgzkIw6IdLnaotkACka2Q-PvP8P-z/pub?output=csv"

# If you want a Sold Positions tab, publish it separately (its own CSV URL)
HISTORY_SHEET_URL = ""  # optional

REQUIRED_COLS = ["Ticker", "Region", "QTY", "AvgCost"]
OPTIONAL_COLS = [
    "Benchmark", "Type", "Thesis", "FirstBuyDate",
    "LivePrice_GS", "PrevClose_GS", "FX_Rate"
]

SUMMARY_NOISE_REGEX = r"TOTAL|PORTFOLIO|SUMMARY|CASH"
DEFAULT_BENCH = {"us": "^GSPC", "india": "^NSEI"}

# Core macro assets used for charts; benchmarks are discovered dynamically too
CORE_ASSETS = ["^GSPC", "^NSEI", "GC=F", "SI=F", "USDINR=X"]
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

def _safe_pct(x):
    if x is None or pd.isna(x):
        return None
    return float(x) * 100.0

# -----------------------------
# Market session logic
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

    near_zero = calc_df["Day_Ret"].dropna().abs() < 1e-9
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
    df["FX_Rate"] = df["FX_Rate"].apply(_as_float)

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
        FX_Rate=("FX_Rate", "max"),
    )
    agg["AvgCost"] = agg["TotalCost"] / agg["QTY"]

    # FX_Rate: ONLY for INR exposure/weights. Default 1.0 if missing.
    agg.loc[agg["FX_Rate"].isna(), "FX_Rate"] = 1.0

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
# Dynamic Symbol Discovery (Fix Ghost Benchmarks)
# -----------------------------
def _discover_all_symbols(df_sheet: pd.DataFrame):
    holdings = df_sheet["Ticker"].dropna().unique().tolist()
    benchmarks = df_sheet["Benchmark"].dropna().unique().tolist()
    core = CORE_ASSETS[:]
    return sorted(list(set(holdings + benchmarks + core)))

# -----------------------------
# Pricing Engine (Triple fallback)
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
def fetch_history_closes_chunked(
    tickers,
    period="15d",
    interval="1d",
    chunk_size=25,
    retries=2,
    timeout_per_chunk=15,
    skip_timeouts=True,
):
    """
    Bulletproof bulk fetch:
    - chunking to avoid 100-ticker wall
    - progress bar
    - timeout alarm on Unix (Streamlit Cloud is Linux)
    """
    import signal

    tickers = sorted(list(set([_clean_str(t) for t in tickers if _clean_str(t)])))
    if not tickers:
        return pd.DataFrame()

    def timeout_handler(signum, frame):
        raise TimeoutError("Chunk took too long")

    frames = []
    progress = st.progress(0.0)

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        progress.progress(min((i + chunk_size) / len(tickers), 1.0))

        last_err = None
        for attempt in range(retries + 1):
            try:
                if hasattr(signal, "SIGALRM"):
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(timeout_per_chunk))

                df = yf.download(
                    tickers=chunk,
                    period=period,
                    interval=interval,
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                    group_by="column"
                )

                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)

                if df is not None and not df.empty:
                    frames.append(df)
                last_err = None
                break
            except TimeoutError:
                if skip_timeouts:
                    st.warning(f"⏱️ Chunk {i//chunk_size + 1} timed out. Skipping {len(chunk)} tickers.")
                    last_err = None
                    break
                last_err = TimeoutError("timeout")
            except Exception as e:
                last_err = e
                _time.sleep(0.6 * (attempt + 1))

        if last_err:
            st.warning(f"⚠️ Chunk {i//chunk_size + 1} failed: {str(last_err)[:80]}")

    progress.empty()

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, axis=1)

@st.cache_data(ttl=300)
def build_prices_with_sheet_fallback(tickers, sheet_fallback: dict, timeout_per_chunk=15, skip_timeouts=True):
    hist = fetch_history_closes_chunked(
        tickers,
        period="15d",
        interval="1d",
        chunk_size=25,
        retries=2,
        timeout_per_chunk=timeout_per_chunk,
        skip_timeouts=skip_timeouts,
    )

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

        # 1) fast_info
        live_fast, prev_fast = _fast_live_prev(tk)
        if live_fast is not None or prev_fast is not None:
            source = "yfinance_fastinfo"

        # 2) bulk history
        last_close, prev_close = _hist_last_prev(hist, tk)
        if (live_fast is None) and (last_close is not None):
            source = "yfinance_bulk"

        live = live_fast if live_fast is not None else last_close
        prev = prev_fast if prev_fast is not None else prev_close

        # 3) single-ticker fallback
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

        # 4) google sheet fallback
        if (live is None) or (prev is None):
            fb = sheet_fallback.get(tk, {})
            fb_live = fb.get("live", None)
            fb_prev = fb.get("prev", None)

            if live is None and fb_live is not None:
                try:
                    live = float(fb_live)
                    source = "google_sheet"
                except Exception:
                    pass
            if prev is None and fb_prev is not None:
                try:
                    prev = float(fb_prev)
                    source = "google_sheet"
                except Exception:
                    pass

        # if still missing prev, mirror live (prevents division crash)
        if prev is None and live is not None:
            prev = live

        price_map[tk] = {"live": live, "prev": prev, "source": source}

    return price_map

# -----------------------------
# Benchmark inception index (for regret matrix)
# -----------------------------
@st.cache_data(ttl=900)
def build_index_monthly(symbol: str, start_date: pd.Timestamp):
    if start_date is None or pd.isna(start_date):
        return None
    start_date = pd.to_datetime(start_date).tz_localize(None)
    px = yf.download(
        symbol,
        start=(start_date - pd.Timedelta(days=20)).strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False
    )
    if px is None or px.empty or "Close" not in px.columns:
        return None
    m = px["Close"].resample("M").last().dropna()
    if m.empty:
        return None
    idx = (m / float(m.iloc[0])) * 100.0
    idx.name = symbol
    return idx

# ============================================================
# UI: Sidebar
# ============================================================
st.sidebar.markdown(f"## {APP_TITLE}")
st.sidebar.markdown("Institutional tracking of my portfolio and alpha.")
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
st.sidebar.markdown("### Performance Controls")

show_details = st.sidebar.checkbox("📱 Desktop mode (show all columns)", value=False, key="show_details_toggle")

skip_timeouts = st.sidebar.checkbox("⏱️ Skip slow tickers (recommended)", value=True, key="skip_timeouts_toggle")
timeout_per_chunk = st.sidebar.slider("Chunk timeout (seconds)", 5, 30, 15, 1, key="timeout_per_chunk_slider")

st.sidebar.markdown("---")
with st.sidebar.form("follow_form", clear_on_submit=True):
    email = st.text_input("Enter your email to follow updates", placeholder="name@email.com", key="follow_email")
    submitted = st.form_submit_button("Follow", use_container_width=True)
    if submitted:
        if email and "@" in email:
            st.success("Saved. (Connect to Zapier/Make/SMTP to send notifications.)")
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

# Sheet fallback map for prices
sheet_fallback = {}
for _, r in df_sheet.iterrows():
    tk = _clean_str(r.get("Ticker", ""))
    sheet_fallback[tk] = {"live": r.get("LivePrice_GS", None), "prev": r.get("PrevClose_GS", None)}

all_symbols = _discover_all_symbols(df_sheet)

with st.spinner("Syncing portfolio data..."):
    prices = build_prices_with_sheet_fallback(
        all_symbols,
        sheet_fallback,
        timeout_per_chunk=timeout_per_chunk,
        skip_timeouts=skip_timeouts,
    )

rows, failures = [], []

for _, r in df_sheet.iterrows():
    tk = _clean_str(r["Ticker"])
    bench = _clean_str(r["Benchmark"])
    region = r["Region"]

    p_tk = prices.get(tk, None)
    p_b = prices.get(bench, None) if bench else None

    # Issue #1: no silent failure
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

    # FX used ONLY for INR exposure/weights (NOT returns)
    fx_locked = float(r.get("FX_Rate", 1.0) or 1.0)

    # Issue #3: zombie holdings / zero checks
    if avg == 0 or live == 0:
        failures.append({"Ticker": tk, "Reason": "Zero price detected (delisted, halted, or bad data)"})
        continue

    total_ret = (live - avg) / avg if avg != 0 else None
    day_ret = (live - prev) / prev if prev != 0 else None

    # absurd intraday moves are usually data issues
    if day_ret is not None and abs(day_ret) > 0.5:
        failures.append({"Ticker": tk, "Reason": f"Suspicious intraday move: {_safe_pct(day_ret):.1f}%"})
        continue

    alpha_day = None
    b_day = None
    if bench:
        b_live = float(p_b["live"])
        b_prev = float(p_b["prev"])
        if b_prev != 0:
            b_day = (b_live - b_prev) / b_prev
        if day_ret is not None and b_day is not None:
            alpha_day = day_ret - b_day

    value_inr = qty * live * fx_locked  # exposure only
    rows.append({
        "Ticker": tk,
        "Region": region,
        "Benchmark": bench,
        "Compared To": _bench_context(bench),
        "QTY": qty,
        "AvgCost": avg,
        "LivePrice": live,
        "PrevClose": prev,
        "Total Ret%": _safe_pct(total_ret),
        "Day Ret%": _safe_pct(day_ret),
        "Bench Day%": _safe_pct(b_day),
        "Alpha Day%": _safe_pct(alpha_day),
        "Beat_Index_Tag": _status_tag(alpha_day, bench),
        "Value_INR": value_inr,
        "FX_Rate_Locked": fx_locked,
        "PriceSource": p_tk.get("source", "none"),
    })

calc_df = pd.DataFrame(rows)

# Degraded Mode banner (Issue #1)
if failures:
    st.warning(f"⚠️ Degraded Mode: {len(failures)} ticker(s) missing live data. Showing partial portfolio.")
    with st.expander("🔧 View Pricing Failures", expanded=False):
        st.dataframe(pd.DataFrame(failures), use_container_width=True, key="failures_table")
        st.caption("Tip: Add LivePrice_GS and PrevClose_GS columns in your Google Sheet for backup pricing.")

if calc_df.empty:
    st.error("No rows could be priced. Check tickers, Yahoo availability, or provide LivePrice_GS/PrevClose_GS fallbacks.")
    st.stop()

# Weights from INR exposure
total_value = float(calc_df["Value_INR"].sum()) if calc_df["Value_INR"].notna().any() else 0.0
if total_value > 0:
    calc_df["Weight%"] = (calc_df["Value_INR"] / total_value) * 100.0
else:
    calc_df["Weight%"] = None

# Header + market status
now_utc = datetime.now(timezone.utc)
status_msg, _ = _market_status_badge(now_utc, calc_df)
st.title(APP_TITLE)
st.caption(status_msg)

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Holdings Priced", f"{len(calc_df)}")
with col2:
    st.metric("Total Exposure (INR)", f"₹{total_value:,.0f}")
with col3:
    avg_day = calc_df["Day Ret%"].dropna().mean()
    st.metric("Avg Day Return", f"{avg_day:.2f}%" if pd.notna(avg_day) else "—")
with col4:
    avg_alpha = calc_df["Alpha Day%"].dropna().mean()
    st.metric("Avg Alpha (Day)", f"{avg_alpha:.2f}%" if pd.notna(avg_alpha) else "—")

st.divider()

# -----------------------------
# Table (Issue #4 mobile UX)
# -----------------------------
if show_details:
    cols_to_show = [
        "Ticker", "Region", "Benchmark", "Weight%", "QTY", "AvgCost",
        "LivePrice", "PrevClose", "Total Ret%", "Day Ret%",
        "Alpha Day%", "Beat_Index_Tag", "PriceSource"
    ]
else:
    cols_to_show = ["Ticker", "Weight%", "Total Ret%", "Day Ret%", "Beat_Index_Tag"]

show = calc_df.copy()
# clean formatting for display
for c in ["Weight%", "Total Ret%", "Day Ret%", "Alpha Day%"]:
    if c in show.columns:
        show[c] = pd.to_numeric(show[c], errors="coerce").round(2)

st.subheader("📌 Current Holdings")
st.dataframe(show[cols_to_show], use_container_width=True, key="holdings_table")

# Download CSV (FIXES your DuplicateElementId crash)
csv_bytes = show.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Picks as CSV",
    data=csv_bytes,
    file_name="atharva_portfolio_picks.csv",
    mime="text/csv",
    help="Download my current picks to track them in your own sheet.",
    key="download_picks_csv_main"   # <- critical
)

st.divider()

# -----------------------------
# Simple allocation pie (optional, fast)
# -----------------------------
st.subheader("🧭 Allocation by Region")
alloc = calc_df.groupby("Region", as_index=False)["Value_INR"].sum()
if not alloc.empty and alloc["Value_INR"].sum() > 0:
    fig = go.Figure(data=[go.Pie(labels=alloc["Region"], values=alloc["Value_INR"], hole=0.35)])
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# -----------------------------
# Feature: Regret Matrix (Opportunity Cost) vs S&P 500
# -----------------------------
st.subheader("😢 The Road Not Taken (Opportunity Cost)")
strategy_start = df_sheet["FirstBuyDate"].dropna().min() if "FirstBuyDate" in df_sheet.columns else pd.NaT

if pd.notna(strategy_start):
    spx_idx = build_index_monthly("^GSPC", strategy_start)
    if spx_idx is not None and len(spx_idx) >= 2:
        spx_return = (float(spx_idx.iloc[-1]) / float(spx_idx.iloc[0])) - 1.0

        invested_capital = float(df_sheet["TotalCost"].sum())  # cost basis (native currency mix)
        # Opportunity cost is conceptual; we show INR exposure comparison using today's INR exposure:
        alt_value = total_value * (1 + spx_return)

        st.metric(
            "If Current Exposure Tracked S&P 500 Since First Buy",
            f"₹{alt_value:,.0f}",
            delta=f"₹{(total_value - alt_value):,.0f}"
        )
        st.caption("This is a directional opportunity-cost view, not a backtest of exact cashflows.")
    else:
        st.info("Regret Matrix unavailable (insufficient S&P 500 history).")
else:
    st.info("Add FirstBuyDate to enable the Regret Matrix.")

# -----------------------------
# Sold positions (optional)
# -----------------------------
if HISTORY_SHEET_URL.strip():
    sold = load_sold_history(HISTORY_SHEET_URL)
    if not sold.empty:
        st.divider()
        st.subheader("✅ Sold Positions")
        st.dataframe(sold, use_container_width=True, key="sold_positions_table")
