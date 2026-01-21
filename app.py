import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
import plotly.express as px

# ============================================================
# Global Alpha Strategy Terminal (Combined Production Version)
# - Defensive sheet cleaning + duplicate aggregation (Gemini)
# - Strong price engine + explicit failures + defaults (ChatGPT)
# - Native returns + FX only for weights (Your philosophy)
# - Per-pick Alpha vs native benchmark (Day)
# - Weighted custom benchmark for Day: (wIndia*Nifty + wUS*S&P)
# - 5Y monthly macro trends: S&P, Nifty, Gold
# ============================================================

st.set_page_config(page_title="Global Alpha Terminal", layout="wide", page_icon="🏛️")

SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUHmE__8dpl_nKGv5F5mTXO7e3EyVRqz-PJF_4yyIrfJAa7z8XgzkIw6IdLnaotkACka2Q-PvP8P-z/pub?output=csv"

# -----------------------------
# Config
# -----------------------------
REQUIRED_COLS = ["Ticker", "Region", "QTY", "AvgCost"]
OPTIONAL_COLS = ["Benchmark", "Type", "Thesis"]

SUMMARY_NOISE_REGEX = r"TOTAL|PORTFOLIO|SUMMARY|CASH"

DEFAULT_BENCH = {"us": "^GSPC", "india": "^NSEI"}  # fallback if Benchmark blank

MACRO_ASSETS = ["^GSPC", "^NSEI", "GC=F"]  # S&P, Nifty, Gold


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

def _safe_pct(x):
    return None if x is None else x * 100

# -----------------------------
# Load + clean + aggregate (Gemini + ChatGPT)
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

    # Drop invalid
    df = df.dropna(subset=["Ticker", "Region", "QTY", "AvgCost"])
    df = df[df["QTY"] != 0]

    # Default benchmark if blank
    df.loc[df["Benchmark"].str.len() == 0, "Benchmark"] = df["Region"].apply(_default_benchmark_for_region)

    # Aggregate duplicates (weighted avg cost)
    df["TotalCost"] = df["QTY"] * df["AvgCost"]
    agg = df.groupby(["Ticker", "Region", "Benchmark"], as_index=False).agg(
        QTY=("QTY", "sum"),
        TotalCost=("TotalCost", "sum"),
        Type=("Type", "first"),
        Thesis=("Thesis", "first"),
    )
    agg["AvgCost"] = agg["TotalCost"] / agg["QTY"]

    # Keep only valid
    agg = agg.dropna(subset=["Ticker", "QTY", "AvgCost"])
    agg = agg[agg["QTY"] != 0]

    return agg.reset_index(drop=True)


# -----------------------------
# Pricing Engine (ChatGPT stronger version)
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
        last_close = None
        prev_close_fallback = None

        # history fallback
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
            prev = live  # prevents crash; day ret becomes 0

        price_map[tk] = {"live": live, "prev": prev}

    return price_map

@st.cache_data(ttl=900)
def fetch_fx_usdinr():
    live, prev = _fast_live_prev("USDINR=X")
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

# Always include these for custom benchmark and macro
all_symbols = list(set(holdings + benchmarks + MACRO_ASSETS + ["USDINR=X"]))

with st.spinner("Syncing Alpha Terminal..."):
    fx_usdinr = fetch_fx_usdinr()
    prices = build_prices(all_symbols)
    macro = fetch_5y_macro()

# Build calc rows with explicit failures
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

    # Native returns (no FX)
    total_ret = (live - avg) / avg if avg != 0 else None
    day_ret = (live - prev) / prev if prev != 0 else None

    # Alpha Day vs benchmark (native)
    alpha_day = None
    if bench:
        b_live = float(p_b["live"])
        b_prev = float(p_b["prev"])
        b_day = (b_live - b_prev) / b_prev if b_prev != 0 else None
        if day_ret is not None and b_day is not None:
            alpha_day = day_ret - b_day

    # Weighting value in INR (FX only here)
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
        "Type": r.get("Type", ""),
        "Thesis": r.get("Thesis", ""),
    })

calc_df = pd.DataFrame(rows)

if calc_df.empty:
    st.error("No holdings could be priced. Check tickers and benchmarks.")
    if failures:
        st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)
    st.stop()

# Weights + portfolio
calc_df["Weight"] = calc_df["Value_INR"] / calc_df["Value_INR"].sum()
port_day = (calc_df["Day_Ret"] * calc_df["Weight"]).sum()
port_total = (calc_df["Total_Ret"] * calc_df["Weight"]).sum()

# Region weights
in_w = calc_df.loc[calc_df["Region"].str.upper() == "INDIA", "Weight"].sum()
us_w = calc_df.loc[calc_df["Region"].str.upper() == "US", "Weight"].sum()

# Custom market day benchmark (country blend)
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

# -----------------------------
# UI
# -----------------------------
st.title("🏛️ Global Alpha Strategy Terminal")
st.caption("Native alpha focus: returns are in local currency. FX is used only for INR portfolio weighting.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Strategy Total Return", f"{port_total*100:.2f}%")

if custom_bench_day is not None:
    m2.metric(
        "Portfolio Day Move",
        f"{port_day*100:.2f}%",
        f"{(port_day - custom_bench_day)*100:.2f}% vs Weighted Market"
    )
else:
    m2.metric("Portfolio Day Move", f"{port_day*100:.2f}%")

m3.metric("USD/INR (Live/Close)", f"{fx_usdinr:.2f}")
m4.metric("Last Sync (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))

with st.expander("🔧 Debug: skipped tickers / pricing snapshot"):
    if failures:
        st.write("Skipped rows:")
        st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)
    else:
        st.write("All holdings priced successfully.")
    st.write("Snapshot:")
    st.dataframe(calc_df[["Ticker","Benchmark","LivePrice","PrevClose","AvgCost","QTY"]], use_container_width=True, hide_index=True)

st.divider()

c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📈 5-Year Macro Trends (Monthly, Indexed to 100)")
    if macro is None or macro.empty:
        st.info("Macro trend data unavailable right now.")
    else:
        macro_norm = (macro / macro.iloc[0] * 100).reset_index().melt(
            id_vars="Date", var_name="Asset", value_name="Index (Base=100)"
        )
        fig = px.line(macro_norm, x="Date", y="Index (Base=100)", color="Asset")
        st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("🌎 Regional Allocation")
    alloc = calc_df.groupby("Region", as_index=False)["Weight"].sum()
    fig_pie = px.pie(alloc, values="Weight", names="Region", hole=0.5)
    st.plotly_chart(fig_pie, use_container_width=True)
    st.write(f"**India Allocation:** {in_w*100:.2f}%")
    st.write(f"**US Allocation:** {us_w*100:.2f}%")
    if custom_bench_day is not None:
        st.caption(f"Custom Market Day: {custom_bench_day*100:.2f}%")
        if nifty_day is not None:
            st.caption(f"Nifty Day: {nifty_day*100:.2f}%")
        if spx_day is not None:
            st.caption(f"S&P Day: {spx_day*100:.2f}%")

st.subheader("📌 Individual Pick Performance Matrix (Alpha vs Bench)")
show = calc_df.copy()
show["Weight%"] = show["Weight"] * 100
show["Total_Ret%"] = show["Total_Ret"] * 100
show["Day_Ret%"] = show["Day_Ret"] * 100
show["Alpha_Day%"] = show["Alpha_Day"] * 100

st.dataframe(
    show[["Ticker","Region","Benchmark","Weight%","AvgCost","LivePrice","Total_Ret%","Day_Ret%","Alpha_Day%"]],
    column_config={
        "Weight%": st.column_config.NumberColumn("Weight", format="%.2f%%"),
        "Total_Ret%": st.column_config.NumberColumn("Total Ret", format="%.2f%%"),
        "Day_Ret%": st.column_config.NumberColumn("Day Ret", format="%.2f%%"),
        "Alpha_Day%": st.column_config.NumberColumn("Alpha (Day)", format="%.2f%%"),
        "AvgCost": st.column_config.NumberColumn(format="%.2f"),
        "LivePrice": st.column_config.NumberColumn(format="%.2f"),
    },
    use_container_width=True,
    hide_index=True
)

# Optional: thesis viewer (keeps it lean)
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
