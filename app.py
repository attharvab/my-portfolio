import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
import plotly.express as px

# ============================================================
# Global Alpha Strategy Terminal (Human-First Production)
# What this app does (in plain English):
# 1) Shows your total return per stock in its native currency (USD stocks in USD, India stocks in INR)
# 2) Shows "Did you beat the market TODAY?" using a weighted benchmark:
#    (India weight * Nifty day move) + (US weight * S&P day move)
# 3) Shows per-pick "Beat Index?" tags so first-time users understand instantly
# 4) Shows 5Y macro trends with readable labels: (S&P 500), (Nifty 50), (Gold), (Silver)
# 5) Adds a "My Portfolio (Indexed)" line if FirstBuyDate is present (point-in-time backtest)
#
# Your philosophy enforced:
# - FX is used ONLY for weights (exposure), NOT for returns
# - Returns are native currency
# - Defensive + lean: explicit failures, safe fallbacks, caching
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

# Macro assets + user-friendly names (includes Silver SI=F)
MACRO_ASSETS = ["^GSPC", "^NSEI", "GC=F", "SI=F"]
ASSET_LABELS = {
    "^GSPC": "^GSPC (S&P 500)",
    "^NSEI": "^NSEI (Nifty 50)",
    "GC=F": "GC=F (Gold)",
    "SI=F": "SI=F (Silver)",
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
        return pd.to_datetime(str(x).strip(), errors="coerce").date()
    except Exception:
        return pd.NaT

def _fmt_pct(x):
    if x is None or pd.isna(x):
        return None
    return x * 100

def _status_tag(alpha_day):
    # alpha_day is stock day return minus benchmark day return
    if alpha_day is None or pd.isna(alpha_day):
        return "—"
    return "🔥 Beating Market" if alpha_day >= 0 else "❄️ Lagging Market"

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

    # Ensure optional columns exist
    for c in OPTIONAL_COLS:
        if c not in df.columns:
            df[c] = ""

    # Clean strings
    df["Ticker"] = df["Ticker"].apply(_clean_str)
    df["Region"] = df["Region"].apply(_normalize_region)
    df["Benchmark"] = df["Benchmark"].apply(_clean_str)
    df["Type"] = df["Type"].apply(_clean_str)
    df["Thesis"] = df["Thesis"].apply(_clean_str)

    # Filter out empty / summary rows
    df = df[df["Ticker"].str.len() > 0]
    df = df[~df["Ticker"].str.contains(SUMMARY_NOISE_REGEX, case=False, na=False)]

    # Parse numbers
    df["QTY"] = df["QTY"].apply(_as_float)
    df["AvgCost"] = df["AvgCost"].apply(_as_float)

    # FirstBuyDate (optional)
    df["FirstBuyDate"] = df["FirstBuyDate"].apply(_parse_date)

    # Drop invalid
    df = df.dropna(subset=["Ticker", "Region", "QTY", "AvgCost"])
    df = df[df["QTY"] != 0]

    # Default benchmark if blank
    df.loc[df["Benchmark"].str.len() == 0, "Benchmark"] = df["Region"].apply(_default_benchmark_for_region)

    # Aggregate duplicates (weighted avg cost); for FirstBuyDate keep the MOST RECENT date
    df["TotalCost"] = df["QTY"] * df["AvgCost"]
    agg = df.groupby(["Ticker", "Region", "Benchmark"], as_index=False).agg(
        QTY=("QTY", "sum"),
        TotalCost=("TotalCost", "sum"),
        Type=("Type", "first"),
        Thesis=("Thesis", "first"),
        FirstBuyDate=("FirstBuyDate", "max"),
    )
    agg["AvgCost"] = agg["TotalCost"] / agg["QTY"]

    # Keep only valid
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
    """
    Returns dict: {ticker: {"live": x, "prev": y}}
    live: fast_info last_price if possible else last close
    prev: fast_info previous_close if possible else previous daily close
    """
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
            prev = live  # day ret becomes 0, avoids crash

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
# FX does NOT drive returns: US holdings are converted using FX at START DATE (constant FX)
# -----------------------------
@st.cache_data(ttl=900)
def build_portfolio_growth_index(df_holdings: pd.DataFrame):
    """
    Returns monthly indexed series (base=100) for "My Portfolio"
    Requires FirstBuyDate present for at least 1 row.
    """
    if "FirstBuyDate" not in df_holdings.columns:
        return None

    # Keep rows with valid FirstBuyDate
    dff = df_holdings.dropna(subset=["FirstBuyDate"]).copy()
    if dff.empty:
        return None

    # Determine start date (earliest buy)
    start_date = pd.to_datetime(min(dff["FirstBuyDate"])).tz_localize(None)
    # Pull monthly prices from start to today for all holdings and USDINR (for constant FX)
    tickers = sorted(dff["Ticker"].unique().tolist())
    symbols = list(set(tickers + ["USDINR=X"]))

    px = yf.download(
        tickers=symbols,
        start=(start_date - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=True
    )["Close"]

    if px is None or px.empty:
        return None

    # Monthly end-of-month prices
    px_m = px.resample("M").last().dropna(how="all")

    # Constant FX at portfolio start month end (or first available)
    fx_series = px_m["USDINR=X"].dropna() if "USDINR=X" in px_m.columns else pd.Series(dtype=float)
    fx0 = float(fx_series.iloc[0]) if not fx_series.empty else 83.0

    # Build value series month by month
    portfolio_value = pd.Series(0.0, index=px_m.index)

    for _, r in dff.iterrows():
        tk = r["Ticker"]
        region = r["Region"]
        qty = float(r["QTY"])
        buy_date = pd.to_datetime(r["FirstBuyDate"]).tz_localize(None)

        if tk not in px_m.columns:
            continue

        s = px_m[tk].copy()
        # Only include from buy month onward (point-in-time)
        s.loc[s.index < pd.to_datetime(buy_date).to_period("M").to_timestamp("M")] = pd.NA
        s = s.ffill()

        # Convert US holdings to INR using constant FX0 (so FX doesn't create "return")
        conv = fx0 if _region_key(region) == "us" else 1.0
        portfolio_value = portfolio_value.add(qty * s * conv, fill_value=0.0)

    # Drop months where portfolio was 0 (before any buys)
    portfolio_value = portfolio_value[portfolio_value > 0]

    if portfolio_value.empty:
        return None

    # Index to 100
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

# Symbols to price
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

    # Native returns (FX NOT used here)
    total_ret = (live - avg) / avg if avg != 0 else None
    day_ret = (live - prev) / prev if prev != 0 else None

    # Alpha vs benchmark (native)
    alpha_day = None
    b_day = None
    if bench:
        b_live = float(p_b["live"])
        b_prev = float(p_b["prev"])
        b_day = (b_live - b_prev) / b_prev if b_prev != 0 else None
        if day_ret is not None and b_day is not None:
            alpha_day = day_ret - b_day

    # INR value ONLY for exposure/weights (FX used ONLY here)
    value_inr = qty * live * (fx_usdinr if _region_key(region) == "us" else 1.0)

    rows.append({
        "Ticker": tk,
        "Region": region,
        "Benchmark": bench,
        "QTY": qty,
        "AvgCost": avg,
        "LivePrice": live,
        "PrevClose": prev,
        "Value_INR": value_inr,
        "Total_Ret": total_ret,
        "Day_Ret": day_ret,
        "Alpha_Day": alpha_day,
        "Beat_Index_Tag": _status_tag(alpha_day),
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

# Weights (live value weights)
calc_df["Weight"] = calc_df["Value_INR"] / calc_df["Value_INR"].sum()

# Portfolio day move (weighted)
port_day = (calc_df["Day_Ret"] * calc_df["Weight"]).sum()
port_total = (calc_df["Total_Ret"] * calc_df["Weight"]).sum()

# Region weights (live exposure)
in_w = calc_df.loc[calc_df["Region"].str.upper() == "INDIA", "Weight"].sum()
us_w = calc_df.loc[calc_df["Region"].str.upper() == "US", "Weight"].sum()

# Weighted market benchmark day move: (India*Nifty + US*S&P)
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

# Market beat / behind string
market_alpha = None
market_alpha_text = None
if custom_bench_day is not None:
    market_alpha = port_day - custom_bench_day
    if market_alpha >= 0:
        market_alpha_text = f"✅ {abs(market_alpha)*100:.2f}% Ahead of Market"
    else:
        market_alpha_text = f"⚠️ {abs(market_alpha)*100:.2f}% Behind Market"

# Cost-based weights (to explain your 6.68% vs 4.23% discrepancy)
# This approximates your sheet logic: weight on purchase cost, not live value
df_cost = df_sheet.copy()
df_cost["Cost_INR"] = df_cost["TotalCost"] * (fx_usdinr if df_cost["Region"].str.upper().eq("US").any() else 1.0)
# More exact: convert row-wise
df_cost["Cost_INR"] = df_cost.apply(
    lambda r: float(r["TotalCost"]) * (fx_usdinr if _region_key(r["Region"]) == "us" else 1.0),
    axis=1
)
cost_total = df_cost["Cost_INR"].sum() if not df_cost.empty else 0.0
cost_us_w = (df_cost.loc[df_cost["Region"].str.upper() == "US", "Cost_INR"].sum() / cost_total) if cost_total else 0.0
cost_in_w = (df_cost.loc[df_cost["Region"].str.upper() == "INDIA", "Cost_INR"].sum() / cost_total) if cost_total else 0.0

# -----------------------------
# UI (Human-First)
# -----------------------------
st.title("🏛️ Global Alpha Strategy Terminal")

# Instructional hook (1 sentence)
st.markdown(
    "**This terminal compares your picks to their home benchmarks, Nifty 50 for India and S&P 500 for the US, so you can see if you are actually outperforming the market.**"
)

st.caption(
    "Returns are shown in native currency. FX (USD/INR) is used only to compute portfolio exposure weights (INR value), not to compute returns."
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Total Return (Strategy)", f"{port_total*100:.2f}%")
    st.caption("Total profit or loss vs your AvgCost, in the stock’s own currency.")

with m2:
    st.metric("Today’s Performance", f"{port_day*100:.2f}%", market_alpha_text if market_alpha_text else None)
    if custom_bench_day is not None:
        st.caption(
            f"Market baseline today = (India {in_w*100:.1f}% × Nifty {nifty_day*100:.2f}%) + (US {us_w*100:.1f}% × S&P {spx_day*100:.2f}%)."
        )
    else:
        st.caption("Market baseline unavailable right now (missing Nifty/S&P pricing).")

with m3:
    st.metric("Currency Ref (USD/INR)", f"{fx_usdinr:.2f}")
    st.caption("Fetched as USDINR=X. Used only for exposure weights, never for return calculation.")

with m4:
    st.metric("Last Sync (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
    st.caption("Prices may be delayed (Yahoo Finance).")

with st.expander("🔍 Quick clarity (why your US weight looks different)"):
    st.write(
        f"- **Live exposure (what you *own today*):** US {us_w*100:.2f}% | India {in_w*100:.2f}% (based on current market value)\n"
        f"- **Cost exposure (what you *spent*):** US {cost_us_w*100:.2f}% | India {cost_in_w*100:.2f}% (based on purchase cost)\n\n"
        "If India positions have doubled while US has not, live exposure shifts heavily toward India even if you initially allocated more to US."
    )

if failures:
    st.warning("Some rows were skipped due to missing pricing data.")
    st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)

st.divider()

# -----------------------------
# Macro + Allocation
# -----------------------------
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📈 5-Year Macro Trends (Monthly, Indexed to 100)")

    if macro is None or macro.empty:
        st.info("Macro trend data unavailable right now.")
    else:
        # Rename columns for readability
        macro_named = macro.copy()
        macro_named = macro_named.rename(columns={k: v for k, v in ASSET_LABELS.items() if k in macro_named.columns})

        # Add portfolio growth line if available
        plot_df = macro_named.copy()

        if portfolio_idx is not None and not portfolio_idx.empty:
            # Align portfolio index to macro timeframe
            p = portfolio_idx.copy()
            # Convert to month-end index for merging
            p.index = pd.to_datetime(p.index).to_period("M").to_timestamp("M")
            plot_df = plot_df.merge(p.to_frame(), left_index=True, right_index=True, how="left")

        # Index all series to 100 at their first available point in the plotted window
        plot_df = plot_df.dropna(how="all").ffill()
        base = plot_df.iloc[0]
        plot_idx = (plot_df / base) * 100.0

        long = plot_idx.reset_index().melt(id_vars="Date", var_name="Asset", value_name="Index (Base=100)")
        fig = px.line(long, x="Date", y="Index (Base=100)", color="Asset")
        st.plotly_chart(fig, use_container_width=True)

        if portfolio_idx is None:
            st.caption("To add 'My Portfolio' line, add FirstBuyDate (YYYY-MM-DD) in your Google Sheet.")
        else:
            st.caption("‘My Portfolio (Indexed)’ is a point-in-time backtest using FirstBuyDate and constant FX at start, so FX does not drive returns.")

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
# Picks table (Human labels)
# -----------------------------
st.subheader("📌 Picks (Did each stock beat its index today?)")

show = calc_df.copy()
show["Weight%"] = show["Weight"] * 100
show["Total Ret%"] = show["Total_Ret"].apply(_fmt_pct)
show["Day Ret%"] = show["Day_Ret"].apply(_fmt_pct)
show["Score vs Index%"] = show["Alpha_Day"].apply(_fmt_pct)

# Keep a clean benchmark label
show["Benchmark"] = show["Benchmark"].replace({
    "^GSPC": "^GSPC (S&P 500)",
    "^NSEI": "^NSEI (Nifty 50)",
    "GC=F": "GC=F (Gold)",
    "SI=F": "SI=F (Silver)"
})

st.dataframe(
    show[[
        "Ticker", "Region", "Benchmark", "Weight%", "AvgCost", "LivePrice",
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
