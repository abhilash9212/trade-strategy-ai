# strategies/generate_trade_alerts.py

import logging
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

# ─── GLOBAL FILTER THRESHOLDS ─────────────────────────────────────────────────────
RSI_THRESH = 30    # oversold below
MACD_POS   = 0     # macd diff above

# ─── 1) FETCH TICKER UNIVERSES ────────────────────────────────────────────────────
def fetch_sp500():
    """Scrape the list of S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, header=0)[0]
    # replace dots in tickers (e.g. BRK.B → BRK-B)
    return table.Symbol.str.replace(r"\.", "-", regex=True).tolist()

def fetch_nasdaq100():
    """Scrape the list of NASDAQ-100 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    tables = pd.read_html(url, header=0)
    for t in tables:
        if "Ticker" in t.columns:
            return t.Ticker.str.replace(r"\.", "-", regex=True).tolist()
    return []

# ─── 2) SINGLE‐TICKER ANALYSIS ────────────────────────────────────────────────────
def analyze_ticker(sym: str):
    """
    Download 6 months of daily closes and compute RSI, MACD diff, Bollinger lower band.
    Return a dict if all 3 filters pass, else None.
    """
    try:
        df = yf.download(sym, period="6mo", interval="1d", progress=False)
        if df.empty or "Close" not in df:
            return None

        close = df.Close.dropna()
        if len(close) < 50:
            return None

        # flatten if yfinance returned a 2‐D frame
        if getattr(close, "ndim", 1) == 2:
            close = pd.Series(close.values.flatten(), index=close.index)

        # compute indicators
        rsi_val  = RSIIndicator(close).rsi().iat[-1]
        macd_val = MACD(close).macd_diff().iat[-1]
        bb_low   = BollingerBands(close).bollinger_lband().iat[-1]
        price    = close.iat[-1]

        # apply filters
        if not (rsi_val <  RSI_THRESH): return None
        if not (macd_val > MACD_POS):    return None
        if not (price    < bb_low):      return None

        return {
            "Ticker":    sym,
            "RSI":       round(rsi_val,  2),
            "MACD Diff": round(macd_val, 4),
            "Price":     round(price,    2),
            "BB Lower":  round(bb_low,   2),
        }

    except Exception:
        logging.exception(f"Failed analyzing {sym}")
        return None

# ─── 3) SCAN ENTIRE UNIVERSE & COLLECT ALERTS ──────────────────────────────────
def generate_trade_alerts():
    """
    Go fetch S&P 500 + NASDAQ-100, analyze each, return list of passing alerts.
    """
    # build universe
    sp500   = fetch_sp500()
    nas100  = fetch_nasdaq100()
    tickers = sorted(set(sp500 + nas100))
    logging.info(f"Scanning {len(tickers)} tickers")

    alerts = []
    for sym in tickers:
        out = analyze_ticker(sym)
        if out:
            alerts.append(out)

    return alerts

# ─── 4) ENTRY POINT FOR CI / APPS ─────────────────────────────────────────────────
def run_trade_alerts():
    """
    Wrapper for CI or web apps to call.
    Returns list[dict] of alerts, or raises on fatal error.
    """
    try:
        return generate_trade_alerts()
    except Exception:
        logging.exception("run_trade_alerts() failed")
        raise

if __name__ == "__main__":
    # simple local smoke test
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    alerts = run_trade_alerts()
    if not alerts:
        print("⚠️  No alerts generated.")
    else:
        print("✅ Alerts:")
        for a in alerts:
            print(a)
