from datetime import datetime, timezone, time as dtime
from zoneinfo import ZoneInfo
import calendar as _cal
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


APP_TITLE = "Atharva Portfolio Returns"

# ========= YOUR SHEET URLS =========
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUHmE__8dpl_nKGv5F5mTXO7e3EyVRqz-PJF_4yyIrfJAa7z8XgzkIw6IdLnaotkACka2Q-PvP8P-z/pub?output=csv"

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
    return (f"Beating Market (vs {short})") if alpha_day >= 0 else (f"Lagging Market (vs {short})")


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


def _market_status_badge(now_utc: datetime, calc_df: Optional[pd.DataFrame]):
    us_open = _is_market_open(now_utc, "US")
    in_open = _is_market_open(now_utc, "India")

    if calc_df is None or calc_df.empty or "Day_Ret" not in calc_df.columns:
        if not us_open and not in_open:
            return "Markets are currently closed. Showing data from the last available trading session.", "closed"
        return "Pricing is temporarily unavailable. Showing partial content while data refreshes.", "mixed"

    near_zero = calc_df["Day_Ret"].dropna().abs() < 1e-6
    pct_zero = float(near_zero.mean()) if len(near_zero) else 0.0

    if (not us_open and not in_open) or pct_zero > 0.8:
        return "Markets are currently closed or prices are not updating. Showing data from the last trading session.", "closed"
    if in_open and not us_open:
        return "India market is open. US market is closed (US picks reflect last US session).", "mixed"
    if us_open and not in_open:
        return "US market is open. India market is closed (India picks reflect last India session).", "mixed"
    return "Markets are open.", "open"


# ============================================================
# Load + clean (Holdings)
# ============================================================
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


def fetch_history_closes_chunked(tickers, period="15d", interval="1d", chunk_size=25, retries=2):
    tickers = sorted(list(set([_clean_str(t) for t in tickers if _clean_str(t)])))
    if not tickers:
        return pd.DataFrame()

    frames = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
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
                import time as _time

                _time.sleep(0.7 * (attempt + 1))
                continue

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


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

        if live is None or prev is None:
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


def fetch_fx_usdinr_snapshot():
    live, _ = _fast_live_prev("USDINR=X")
    if live is not None and live > 0:
        return float(live)
    try:
        fx = yf.download(
            "USDINR=X", period="15d", interval="1d", progress=False, auto_adjust=False, threads=False
        )
        if fx is not None and not fx.empty and "Close" in fx.columns:
            s = fx["Close"].dropna()
            if not s.empty:
                v = float(s.iloc[-1])
                return v if v > 0 else 83.0
    except Exception:
        pass
    return 83.0


# ============================================================
# Daily portfolio returns for Webull calendar (NO alpha, NO benchmark)
# (kept to compute a static calendar for the latest month)
# ============================================================
def build_portfolio_daily_returns_4y(holdings_df: pd.DataFrame):
    if holdings_df is None or holdings_df.empty:
        return None

    tickers = holdings_df["Ticker"].dropna().unique().tolist()
    if not tickers:
        return None

    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    start = (end - pd.DateOffset(years=4)).tz_localize(None)

    need = list(set(tickers + ["USDINR=X"]))
    px = yf.download(
        tickers=need,
        start=(start - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    if px is None or px.empty or "Close" not in px.columns:
        return None

    close = px["Close"].copy()
    if isinstance(close, pd.Series):
        close = close.to_frame()

    close = close.dropna(how="all").ffill()
    if close.empty or "USDINR=X" not in close.columns:
        return None

    close.index = pd.to_datetime(close.index).tz_localize(None)

    fx = close["USDINR=X"].dropna().ffill()

    holding_prices_inr = pd.DataFrame(index=close.index)
    region_map = {row["Ticker"]: row["Region"] for _, row in holdings_df.iterrows()}

    for tk in tickers:
        if tk not in close.columns:
            continue
        s = close[tk].dropna().ffill()
        if s.empty:
            continue
        if region_map.get(tk, "") == "US":
            s = s * fx.reindex(s.index).ffill()
        holding_prices_inr[tk] = s

    holding_prices_inr = holding_prices_inr.dropna(how="all").ffill()
    if holding_prices_inr.empty:
        return None

    last_day = holding_prices_inr.index.max()
    last_prices = holding_prices_inr.loc[last_day].dropna()
    qty_map = {row["Ticker"]: float(row["QTY"]) for _, row in holdings_df.iterrows()}

    values = {}
    for tk, p in last_prices.items():
        q = qty_map.get(tk, 0.0)
        values[tk] = q * float(p)

    denom = sum(values.values())
    if denom <= 0:
        return None
    weights = {tk: v / denom for tk, v in values.items()}

    rets = holding_prices_inr.pct_change().replace([np.inf, -np.inf], np.nan)

    port_ret = pd.Series(0.0, index=rets.index)
    for tk, w in weights.items():
        if tk in rets.columns:
            port_ret = port_ret.add(rets[tk].fillna(0.0) * float(w), fill_value=0.0)

    out = pd.DataFrame({"PortfolioRet": port_ret}).dropna()
    if out.empty:
        return None

    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out[out.index >= start]

    if out.empty:
        return None

    return out


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


def build_latest_calendar_context(daily_returns: Optional[pd.Series]) -> Optional[Dict[str, Any]]:
    if daily_returns is None or daily_returns.empty:
        return None

    idx = pd.to_datetime(daily_returns.index).tz_localize(None)
    min_dt, max_dt = idx.min(), idx.max()
    year = max_dt.year
    month = max_dt.month

    z, text, x_labels, y_labels = build_calendar_grid(daily_returns, year, month)

    finite = np.isfinite(z)
    max_abs = float(np.nanmax(np.abs(z[finite]))) if finite.any() else 0.01
    max_abs = max(max_abs, 0.01)

    grid: List[List[Dict[str, Any]]] = []
    for row_idx, label in enumerate(y_labels):
        row_cells: List[Dict[str, Any]] = []
        for col_idx in range(len(x_labels)):
            val = z[row_idx, col_idx]
            if np.isnan(val):
                row_cells.append({"day": "", "ret": None, "class": "cal-empty"})
            else:
                pct = val * 100.0
                norm = abs(val) / max_abs if max_abs > 0 else 0.0
                if val > 0:
                    if norm > 0.66:
                        css = "cal-strong-pos"
                    elif norm > 0.33:
                        css = "cal-pos"
                    else:
                        css = "cal-weak-pos"
                elif val < 0:
                    if norm > 0.66:
                        css = "cal-strong-neg"
                    elif norm > 0.33:
                        css = "cal-neg"
                    else:
                        css = "cal-weak-neg"
                else:
                    css = "cal-neutral"

                day_str = text[row_idx, col_idx].split("<br>")[0]
                row_cells.append({"day": day_str, "ret": f"{pct:.2f}%", "class": css})
        grid.append({"label": label, "cells": row_cells})

    return {
        "year": year,
        "month_name": _cal.month_name[month],
        "columns": x_labels,
        "rows": grid,
    }


# ============================================================
# Core portfolio computation for one static snapshot
# ============================================================
def compute_portfolio_snapshot() -> Dict[str, Any]:
    df_sheet = load_and_clean_data(SHEET_URL)
    if df_sheet.empty:
        raise RuntimeError("No valid holdings found. Check Ticker/Region/QTY/AvgCost.")

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

    prices = build_prices_with_sheet_fallback(all_symbols, sheet_fallback)
    fx_usdinr = fetch_fx_usdinr_snapshot()

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

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

        value_inr = qty * live * (fx_usdinr if _region_key(region) == "us" else 1.0)

        rows.append(
            {
                "Ticker": tk,
                "Region": region,
                "Benchmark": bench,
                "ComparedTo": _bench_context(bench),
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
            }
        )

    calc_df = pd.DataFrame(rows)
    now_utc = datetime.now(timezone.utc)
    status_text, status_type = _market_status_badge(now_utc, calc_df)

    if calc_df.empty:
        return {
            "now_utc": now_utc,
            "status_text": status_text,
            "status_type": status_type,
            "error": "Pricing is temporarily unavailable. Please refresh shortly.",
            "failures": failures,
        }

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

    daily_ret_df_4y = build_portfolio_daily_returns_4y(df_sheet)
    calendar_ctx = (
        build_latest_calendar_context(daily_ret_df_4y["PortfolioRet"]) if daily_ret_df_4y is not None else None
    )

    # region splits
    india_df = calc_df[calc_df["Region"].str.upper() == "INDIA"].copy()
    us_df = calc_df[calc_df["Region"].str.upper() == "US"].copy()

    # picks table
    show = calc_df.copy()
    show["WeightPct"] = show["Weight"] * 100
    show["TotalRetPct"] = show["Total_Ret"].apply(_fmt_pct)
    show["DayRetPct"] = show["Day_Ret"].apply(_fmt_pct)
    show["ScoreVsIndexPct"] = show["Alpha_Day"].apply(_fmt_pct)

    show["BenchmarkLabel"] = show["Benchmark"].replace(
        {
            "^GSPC": "^GSPC (S&P 500)",
            "^NSEI": "^NSEI (Nifty 50)",
            "GC=F": "GC=F (Gold)",
            "SI=F": "SI=F (Silver)",
        }
    )

    picks = []
    for _, row in show.iterrows():
        picks.append(
            {
                "Ticker": row["Ticker"],
                "Region": row["Region"],
                "Benchmark": row["BenchmarkLabel"],
                "ComparedTo": row["ComparedTo"],
                "WeightPct": f"{row['WeightPct']:.2f}%",
                "TotalRetPct": None if pd.isna(row["TotalRetPct"]) else f"{row['TotalRetPct']:.2f}%",
                "DayRetPct": None if pd.isna(row["DayRetPct"]) else f"{row['DayRetPct']:.2f}%",
                "BeatIndexTag": row["Beat_Index_Tag"],
                "ScoreVsIndexPct": None
                if pd.isna(row["ScoreVsIndexPct"])
                else f"{row['ScoreVsIndexPct']:.2f}%",
                "PriceSource": row["PriceSource"],
            }
        )

    country_alloc = calc_df.groupby("Region", as_index=False)["Weight"].sum()
    country_alloc_list = [
        {"region": r["Region"], "weight_pct": f"{r['Weight'] * 100:.2f}%"} for _, r in country_alloc.iterrows()
    ]

    snapshot = {
        "now_utc": now_utc,
        "status_text": status_text,
        "status_type": status_type,
        "port_total_pct": f"{port_total * 100:.2f}%",
        "port_day_pct": f"{port_day * 100:.2f}%",
        "daily_alpha_vs_spx_pct": "—"
        if daily_alpha_vs_spx is None
        else f"{daily_alpha_vs_spx * 100:.2f}%",
        "calendar": calendar_ctx,
        "india_summary": {
            "count": int(len(india_df)),
            "weight_pct": f"{india_df['Weight'].sum() * 100:.2f}%" if not india_df.empty else "0.00%",
        },
        "us_summary": {
            "count": int(len(us_df)),
            "weight_pct": f"{us_df['Weight'].sum() * 100:.2f}%" if not us_df.empty else "0.00%",
        },
        "picks": picks,
        "country_alloc": country_alloc_list,
        "failures": failures,
        "error": None,
    }

    return snapshot


# ============================================================
# FastAPI app & HTML rendering
# ============================================================
from pathlib import Path

app = FastAPI(title=APP_TITLE)

# Resolve paths relative to this file for Vercel compatibility
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Mount static files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        snapshot = compute_portfolio_snapshot()
    except Exception as e:
        snapshot = {
            "now_utc": datetime.now(timezone.utc),
            "status_text": "Error loading data",
            "status_type": "error",
            "error": str(e),
            "picks": [],
            "calendar": None,
            "india_summary": {"count": 0, "weight_pct": "0.00%"},
            "us_summary": {"count": 0, "weight_pct": "0.00%"},
            "country_alloc": [],
            "failures": [],
            "port_total_pct": "—",
            "port_day_pct": "—",
            "daily_alpha_vs_spx_pct": "—",
        }

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": APP_TITLE,
            "snapshot": snapshot,
        },
    )


@app.get("/health", response_class=HTMLResponse)
async def health():
    return "OK"

