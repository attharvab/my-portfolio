import streamlit as st

import pandas as pd

import yfinance as yf

from datetime import datetime, timezone

import plotly.express as px



# 1) SETUP

st.set_page_config(page_title="Global Alpha Strategy", layout="wide", page_icon="📈")



SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUHmE__8dpl_nKGv5F5mTXO7e3EyVRqz-PJF_4yyIrfJAa7z8XgzkIw6IdLnaotkACka2Q-PvP8P-z/pub?output=csv"



@st.cache_data(ttl=300)

def load_data():

    df = pd.read_csv(SHEET_URL)

    df.columns = df.columns.str.strip()

    return df



@st.cache_data(ttl=1800)

def get_usdinr_rate():

    try:

        fx = yf.download("USDINR=X", period="5d", interval="1d", progress=False)

        rate = float(fx["Close"].dropna().iloc[-1])

        return rate if rate > 0 else 83.0

    except Exception:

        return 83.0



@st.cache_data(ttl=300)

def fetch_prices(tickers_list):

    tickers_list = [t for t in tickers_list if isinstance(t, str) and t.strip() and t.lower() != "nan"]

    tickers_list = sorted(list(set([t.strip() for t in tickers_list])))



    if not tickers_list:

        return pd.DataFrame(columns=["Ticker", "Live Price", "Prev Close"])



    prices = yf.download(

        tickers=tickers_list,

        period="5d",
