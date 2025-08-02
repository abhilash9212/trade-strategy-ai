# generate_trade_alerts.py

import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

# ─── 1) Build your expanded universe ───────────────────────────────────────────────
def fetch_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, header=0)[0]
    return table.Symbol.str.replace(r"\.", "-", regex=True).tolist()

def fetch_nasdaq100():
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    tables = pd.read_html(url, header=0)
    for t in tables:
        if "Ticker" in t.columns:
            return t["Ticker"].str.replace(r"\.", "-", regex=True).tolist()
    return []

tickers = sorted(set(fetch_sp500() + fetch_nasdaq100()))

# ─── 2) Your strict filters ───────────────────────────────────────────────────────
RSI_THRESH = 30    # oversold
MACD_POS   = 0     # positive MACD diff
# Price < lower Bollinger Band

# ─── 3) Indicator & filter logic ─────────────────────────────────────────────────
def analyze_ticker(sym):
    try:
        df = yf.download(sym, period="6mo", interval="1d", progress=False)
        if df.empty or "Close" not in df.columns:
            return None

        close = df["Close"].dropna()
        if len(close) < 50:
            return None

        # ensure 1D
        if getattr(close, "ndim", 1) == 2:
            close = pd.Series(close.values.flatten(), index=close.index)

        # compute indicators
        rsi_val   = RSIIndicator(close).rsi().iloc[-1]
        macd_val  = MACD(close).macd_diff().iloc[-1]
        bb_low    = BollingerBands(close).bollinger_lband().iloc[-1]
        price     = close.iloc[-1]

        # strict 3‑signal check
        if not (rsi_val   < RSI_THRESH): return None
        if not (macd_val  > MACD_POS):    return None
        if not (price     < bb_low):      return None

        return {
            "Ticker":     sym,
            "RSI":        round(rsi_val,   2),
            "MACD Diff":  round(macd_val, 4),
            "Price":      round(price,     2),
            "BB Lower":   round(bb_low,    2)
        }

    except Exception:
        return None

# ─── 4) Scan & report ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"🔍 Scanning {len(tickers)} tickers from S&P 500 + NASDAQ‑100…\n")
    candidates = []

    for s in tickers:
        print(f" • {s}", end=" … ")
        out = analyze_ticker(s)
        if out:
            print("✅")
            candidates.append(out)
        else:
            print("—")

    if not candidates:
        print("\n⚠️ Still no tickers passed the 3‑signal filter.")
    else:
        df = pd.DataFrame(candidates).sort_values("MACD Diff", ascending=False)
        best = df.iloc[0]
        print("\n💡 BEST TRADE CANDIDATE:\n")
        print(best.to_string())
        df.to_csv("../output/trade_alerts_best.csv", index=False)
        print("\n✔ Saved results to ../output/trade_alerts_best.csv")
