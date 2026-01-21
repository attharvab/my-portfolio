import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, time
from zoneinfo import ZoneInfo
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# Global Alpha Strategy Terminal (Human-First Production)
# - Bulletproof UX for first-time viewers:
#   1) Market Status badge (weekend/holiday/closed sessions)
#   2) Clear benchmark context in table (vs Nifty / vs S&P)
#   3) Macro chart re-indexed to your FirstBuyDate baseline
# - Your philosophy enforced:
#   - FX used ONLY for exposure weights, NOT for returns
# ============================================================

st.set_page_config(page_title="Global Alpha Strategy Terminal", layout="wide", page_icon="🏛️")

SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUHmE__8dpl_nKGv5F5mTXO7e3EyVRqz-PJF_4yyIrfJAa7z8XgzkIw6IdLnaotkACka2Q-PvP8P-z/pub?output=csv"

# -----------------------------
# Config
# -----------------------------
REQUIRED_COLS = ["Ticker", "Region", "QTY", "AvgCost"]
OPTIONAL_COLS = ["Benchmark", "Type", "Thesis", "FirstBuyDate"]

SUMMARY_NOISE_REGEX = r"TOTAL|PORTFOLIO|SUMMARY|CASH"

DEFAULT_BENCH = {"us": "^GSPC", "india": "^NSEI"}  # if Benchmark blank

MACRO_ASSETS = ["^GSPC", "^NSEI", "GC=F", "SI=F"]  # includes Silver
ASSET_LABELS = {
    "^GSPC": "^GSPC (S&P 500)",
    "^NSEI": "^NSEI (Nifty 50)",
    "GC=F": "GC=F (Gold)",
    "SI=F": "SI=F (Silver)",
}

BENCH_LABEL = {
    "^GSPC": "S&P 500",
    "^NSEI": "Nifty 50",
    "GC=F": "Gold",
    "SI=F": "Silver"
}

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
    # Expect YYYY-MM-DD; defensively parse anything reasonable
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
# Market session logic (India + US)
# -----------------------------
def _is_market_open(now_utc: datetime, market: str) -> bool:
    """
    market: 'US' or 'India'
    Uses simple weekday + time windows.
    Not perfect for holidays, but great UX and avoids "dead app" perception.
    """
    if market == "US":
        tz = ZoneInfo("America/New_York")
        now = now_utc.astimezone(tz)
        if now.weekday() >= 5:
            return False
        # 9:30 to 16:00 ET
        return time(9, 30) <= now.time() <= time(16, 0)

    if market == "India":
        tz = ZoneInfo("Asia/Kolkata")
        now = now_utc.astimezone(tz)
        if now.weekday() >= 5:
            return False
        # 9:15 to 15:30 IST
        return time(9, 15) <= now.time() <= time(15, 30)

    return False

def _market_status_badge(now_utc: datetime, nifty_day, spx_day, calc_df: pd.DataFrame):
    """
    Returns (status_text, status_type)
    status_type in {'open','closed','mixed'} for display styling.
    Adds a fallback heuristic: if day moves are ~0 AND many live==prev, likely closed/delayed.
    """
    us_open = _is_market_open(now_utc, "US")
    in_open = _is_market_open(now_utc, "India")

    # Heuristic: if most day returns are 0 (or near), it "looks dead"
    near_zero = calc_df["Day_Ret"].dropna().abs() < 1e-6
    pct_zero = float(near_zero.mean()) if len(near_zero) else 0.0

    # If both markets "closed" by time OR data looks stale -> show closed note
    if (not us_open and not in_open) or pct_zero > 0.8:
        return "🟡 Markets are currently closed or prices are not updating. Showing data from the last trading session.", "closed"

    if in_open and not us_open:
        return "🟦 India market is open. US market is closed (US picks will reflect last US session).", "mixed"

    if us_open and not in_open:
        return "🟦 US market is open. India market is closed (India picks will reflect last India session).", "mixed"

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
        # Most recent date for point-in-time start
        FirstBuyDate=("FirstBuyDate", "max"),
    )
    agg["AvgCost"] = agg["TotalCost"] / agg["QTY"]

    agg = agg.dropna(subset=["Ticker", "QTY", "AvgCost"])
    agg = agg[agg["QTY"] != 0]
    return agg.reset_index(drop=True)

# -----------------------------
# Pricing Engine (defensive)
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
def fetch_history_closes(tickers):
    tickers = sorted(list(set([_clean_str(t) for t in tickers if _clean_str(t)])))
    if not tickers:
        return pd.DataFrame()
    return yf.download(
        tickers=tickers,
        period="10d",
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=True
    )

@st.cache_data(ttl=300)
def build_prices(tickers):
    hist = fetch_history_closes(tickers)
    price_map = {}

    for tk in tickers:
        tk = _clean_str(tk)
        last_close, prev_close_fallback = None, None

        try:
            if isinstance(hist.columns, pd.MultiIndex):
                s = hist[("Close", tk)].dropna()
            else:
                s = hist["Close"].dropna()

            if len(s) >= 1:
                last_close = float(s.iloc[-1])
            if len(s) >= 2:
                prev_close_fallback = float(s.iloc[-2])
        except Exception:
            last_close, prev_close_fallback = None, None

        live_fast, prev_fast = _fast_live_prev(tk)
        live = live_fast if live_fast is not None else last_close
        prev = prev_fast if prev_fast is not None else prev_close_fallback

        if live is None:
            live = last_close
        if prev is None:
            prev = live

        price_map[tk] = {"live": live, "prev": prev}

    return price_map

@st.cache_data(ttl=900)
def fetch_fx_usdinr():
    live, _ = _fast_live_prev("USDINR=X")
    if live is not None and live > 0:
        return live
    try:
        fx = yf.download("USDINR=X", period="10d", interval="1d", progress=False, auto_adjust=False)
        s = fx["Close"].dropna()
        if not s.empty:
            v = float(s.iloc[-1])
            return v if v > 0 else 83.0
    except Exception:
        pass
    return 83.0

@st.cache_data(ttl=900)
def fetch_5y_macro():
    t = yf.download(MACRO_ASSETS, period="5y", interval="1mo", progress=False, auto_adjust=False)["Close"]
    if isinstance(t, pd.Series):
        t = t.to_frame()
    return t.dropna(how="all").ffill()

# -----------------------------
# Portfolio Growth (Point-in-Time backtest using FirstBuyDate)
# FX does NOT drive returns: US holdings converted using constant FX at start.
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
        start=(start_date - pd.Timedelta(days=14)).strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=True
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

# -----------------------------
# MAIN
# -----------------------------
try:
    df_sheet = load_and_clean_data(SHEET_URL)
except Exception as e:
    st.error(f"Spreadsheet Error: {e}")
    st.stop()

if df_sheet.empty:
    st.warning("No valid holdings found. Check Ticker/Region/QTY/AvgCost.")
    st.stop()

holdings = df_sheet["Ticker"].unique().tolist()
benchmarks = df_sheet["Benchmark"].unique().tolist()

all_symbols = list(set(holdings + benchmarks + ["USDINR=X"] + MACRO_ASSETS))

with st.spinner("Syncing Alpha Terminal..."):
    fx_usdinr = fetch_fx_usdinr()
    prices = build_prices(all_symbols)
    macro = fetch_5y_macro()
    portfolio_idx = build_portfolio_growth_index(df_sheet)

# -----------------------------
# Compute holding rows
# -----------------------------
rows, failures = [], []

for _, r in df_sheet.iterrows():
    tk = r["Ticker"]
    bench = r["Benchmark"]
    region = r["Region"]

    p_tk = prices.get(tk, None)
    p_b = prices.get(bench, None) if bench else None

    if not p_tk or p_tk["live"] is None or p_tk["prev"] is None:
        failures.append({"Ticker": tk, "Reason": "Missing holding price"})
        continue

    if bench and (not p_b or p_b["live"] is None or p_b["prev"] is None):
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
    })

calc_df = pd.DataFrame(rows)

if calc_df.empty:
    st.error("No holdings could be priced. Check tickers and benchmarks.")
    if failures:
        st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)
    st.stop()

calc_df["Weight"] = calc_df["Value_INR"] / calc_df["Value_INR"].sum()
port_day = (calc_df["Day_Ret"] * calc_df["Weight"]).sum()
port_total = (calc_df["Total_Ret"] * calc_df["Weight"]).sum()

in_w = calc_df.loc[calc_df["Region"].str.upper() == "INDIA", "Weight"].sum()
us_w = calc_df.loc[calc_df["Region"].str.upper() == "US", "Weight"].sum()

def _safe_day(sym):
    p = prices.get(sym, None)
    if not p or p["live"] is None or p["prev"] is None or p["prev"] == 0:
        return None
    return (float(p["live"]) - float(p["prev"])) / float(p["prev"])

nifty_day = _safe_day("^NSEI")
spx_day = _safe_day("^GSPC")

custom_bench_day = None
if nifty_day is not None and spx_day is not None:
    custom_bench_day = (in_w * nifty_day) + (us_w * spx_day)

market_alpha_text = None
if custom_bench_day is not None:
    market_alpha = port_day - custom_bench_day
    if market_alpha >= 0:
        market_alpha_text = f"✅ {abs(market_alpha)*100:.2f}% Ahead of Market"
    else:
        market_alpha_text = f"⚠️ {abs(market_alpha)*100:.2f}% Behind Market"

# -----------------------------
# UI (Human-First)
# -----------------------------
st.title("🏛️ Global Alpha Strategy Terminal")
st.markdown(
    "**This terminal compares your picks to their home benchmarks, Nifty 50 for India and S&P 500 for the US, so you can see if you are actually outperforming the market.**"
)
st.caption(
    "Returns are shown in native currency. FX (USD/INR) is used only to compute portfolio exposure weights (INR value), not to compute returns."
)

# Market Status badge (Fix #1)
now_utc = datetime.now(timezone.utc)
status_text, status_type = _market_status_badge(now_utc, nifty_day, spx_day, calc_df)
if status_type == "closed":
    st.info(status_text)
elif status_type == "mixed":
    st.warning(status_text)
else:
    st.success(status_text)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Total Return (Strategy)", f"{port_total*100:.2f}%")
    st.caption("Total profit or loss vs your AvgCost, in the stock’s own currency.")

with m2:
    st.metric("Today’s Performance", f"{port_day*100:.2f}%", market_alpha_text if market_alpha_text else None)
    if custom_bench_day is not None and nifty_day is not None and spx_day is not None:
        st.caption(
            f"Market baseline today = (India {in_w*100:.1f}% × Nifty {nifty_day*100:.2f}%) + (US {us_w*100:.1f}% × S&P {spx_day*100:.2f}%)."
        )
    else:
        st.caption("Market baseline unavailable right now (missing Nifty/S&P pricing).")

with m3:
    st.metric("Currency Ref (USD/INR)", f"{fx_usdinr:.2f}")
    st.caption("Fetched as USDINR=X. Used only for exposure weights, never for return calculation.")

with m4:
    st.metric("Last Sync (UTC)", now_utc.strftime("%Y-%m-%d %H:%M"))
    st.caption("Prices may be delayed (Yahoo Finance).")

if failures:
    st.warning("Some rows were skipped due to missing pricing data.")
    st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)

st.divider()

# -----------------------------
# Macro + Allocation (Fix #3)
# -----------------------------
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📈 5-Year Macro Trends (Monthly, Indexed to 100)")

    if macro is None or macro.empty:
        st.info("Macro trend data unavailable right now.")
    else:
        # Determine baseline date: earliest FirstBuyDate month-end (if available)
        baseline_date = None
        if "FirstBuyDate" in df_sheet.columns and df_sheet["FirstBuyDate"].notna().any():
            earliest = pd.to_datetime(df_sheet["FirstBuyDate"].dropna().min()).tz_localize(None)
            baseline_date = earliest.to_period("M").to_timestamp("M")

        # Rename macro columns
        macro_named = macro.copy().rename(columns={k: v for k, v in ASSET_LABELS.items() if k in macro.columns})

        # Merge portfolio line
        plot_df = macro_named.copy()
        if portfolio_idx is not None and not portfolio_idx.empty:
            p = portfolio_idx.copy()
            p.index = pd.to_datetime(p.index).to_period("M").to_timestamp("M")
            plot_df = plot_df.merge(p.to_frame(), left_index=True, right_index=True, how="left")

        plot_df = plot_df.dropna(how="all").ffill()

        # Re-base all series to 100 at baseline_date (Fix #3)
        if baseline_date is not None:
            # keep only dates >= baseline_date
            plot_df = plot_df[plot_df.index >= baseline_date].copy()

            # if baseline_date not present due to missing month-end, use first available row
            if plot_df.empty:
                plot_df = macro_named.dropna(how="all").ffill().copy()
                if portfolio_idx is not None and not portfolio_idx.empty:
                    plot_df = plot_df.merge(p.to_frame(), left_index=True, right_index=True, how="left")
                plot_df = plot_df.dropna(how="all").ffill()
            base_row = plot_df.iloc[0]
        else:
            base_row = plot_df.iloc[0]

        plot_idx = (plot_df / base_row) * 100.0

        # Plot with thicker "My Portfolio" line (polish)
        fig = go.Figure()
        for col in plot_idx.columns:
            name = col
            width = 4 if "My Portfolio" in col else 2
            fig.add_trace(go.Scatter(
                x=plot_idx.index,
                y=plot_idx[col],
                mode="lines",
                name=name,
                line=dict(width=width),
            ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Index (Base=100)",
            legend_title="Asset",
            margin=dict(l=10, r=10, t=20, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)

        if portfolio_idx is None:
            st.caption("To add 'My Portfolio' line, add FirstBuyDate (YYYY-MM-DD) in your Google Sheet.")
        else:
            st.caption("All lines are re-indexed to 100 on your strategy start date (FirstBuyDate baseline).")

with c2:
    st.subheader("🌎 Currency Risk (Live Exposure)")
    alloc = calc_df.groupby("Region", as_index=False)["Weight"].sum()
    fig_pie = px.pie(alloc, values="Weight", names="Region", hole=0.5)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.write(f"**India (INR exposure):** {in_w*100:.2f}%")
    st.write(f"**US (USD exposure):** {us_w*100:.2f}%")

    st.divider()
    st.subheader("Today’s Market Moves")
    if nifty_day is not None:
        st.write(f"**Nifty 50:** {nifty_day*100:.2f}%")
    if spx_day is not None:
        st.write(f"**S&P 500:** {spx_day*100:.2f}%")
    if custom_bench_day is not None:
        st.write(f"**Your Weighted Market:** {custom_bench_day*100:.2f}%")

st.divider()

# -----------------------------
# Picks table (Fix #2)
# -----------------------------
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

st.dataframe(
    show[[
        "Ticker", "Region", "Benchmark", "Compared To",
        "Weight%", "AvgCost", "LivePrice",
        "Total Ret%", "Day Ret%", "Beat_Index_Tag", "Score vs Index%"
    ]],
    column_config={
        "Weight%": st.column_config.NumberColumn("Weight (Live)", format="%.2f%%"),
        "Total Ret%": st.column_config.NumberColumn("Total Return", format="%.2f%%"),
        "Day Ret%": st.column_config.NumberColumn("Today Return", format="%.2f%%"),
        "Beat_Index_Tag": st.column_config.TextColumn("Beat Index?"),
        "Score vs Index%": st.column_config.NumberColumn("Score vs Index", format="%.2f%%"),
        "AvgCost": st.column_config.NumberColumn("Avg Cost", format="%.2f"),
        "LivePrice": st.column_config.NumberColumn("Live Price", format="%.2f"),
        "Compared To": st.column_config.TextColumn("Compared To"),
    },
    use_container_width=True,
    hide_index=True
)

with st.expander("🧠 Thesis / Notes (from Google Sheet)"):
    tickers_sorted = sorted(show["Ticker"].unique().tolist())
    selected = st.selectbox("Select a ticker", tickers_sorted)
    pick = show[show["Ticker"] == selected]
    if not pick.empty:
        r = pick.iloc[0]
        st.write(f"**Type:** {r.get('Type','')}")
        st.write(f"**Region:** {r.get('Region','')}")
        st.write(f"**Benchmark:** {r.get('Benchmark','')}")
        st.text_area("Thesis (edit in Google Sheet)", value=str(r.get("Thesis","")), height=180)

st.divider()
st.caption("Data source: Yahoo Finance (prices may be delayed). Educational project, not investment advice.")
