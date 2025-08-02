# generate_option_payoff.py

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime

def bs_call_price(S, K, T, r, sigma):
    """Black–Scholes call price; returns intrinsic if T or sigma ≤ 0."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

if __name__ == "__main__":
    TICKER     = "CPRT"
    RFR        = 0.05       # 5% risk‑free rate
    OPTION_QTY = 1          # number of contracts

    tk   = yf.Ticker(TICKER)
    hist = tk.history(period="1d")
    if hist.empty:
        raise SystemExit(f"[ERROR] No spot price for {TICKER}")
    spot = hist["Close"].iloc[-1]

    # 1) Pick the nearest non-expired expiration
    today   = datetime.utcnow().date()
    exps    = [e for e in tk.options if (pd.to_datetime(e).date() - today).days > 0]
    if not exps:
        raise SystemExit("[ERROR] No upcoming expirations found")
    expiry = exps[0]
    T_days = (pd.to_datetime(expiry).date() - today).days
    T      = T_days / 365

    # 2) Load calls and choose ~5% OTM strike
    calls = tk.option_chain(expiry).calls
    otm   = calls[calls.strike > spot * 1.05]
    if otm.empty:
        raise SystemExit("[ERROR] No OTM calls above 5% found")
    row = otm.sort_values("strike").iloc[0]
    K   = row.strike

    # 3) Try live lastPrice → bid/ask mid → BS fallback
    last_px = getattr(row, "lastPrice", np.nan) or np.nan
    bid     = getattr(row, "bid",       np.nan) or np.nan
    ask     = getattr(row, "ask",       np.nan) or np.nan
    iv      = getattr(row, "impliedVolatility", np.nan) or np.nan

    if last_px > 0:
        price = float(last_px)
        source = "LastPrice"
    elif bid > 0 and ask >= bid:
        price = float((bid + ask) / 2)
        source = "Bid/Ask Mid"
    else:
        price = bs_call_price(spot, K, T, RFR, iv)
        source = "Black–Scholes"

    # 4) Build payoff profile
    prices = np.linspace(0.5 * spot, 1.5 * spot, 200)
    payoff = (np.maximum(prices - K, 0) - price) * OPTION_QTY * 100

    # 5) Export CSV
    out_df = pd.DataFrame({"Price@Expiry": prices, "P/L": payoff})
    csv_path = "../output/option_payoff_CPRT.csv"
    out_df.to_csv(csv_path, index=False)

    # 6) Plot
    plt.figure(figsize=(8,5))
    plt.plot(prices, payoff, linewidth=2)
    plt.axhline(0, color="black", linewidth=1)
    entry = round(price,2)
    iv_pct = f"{iv*100:.2f}%" if iv>0 else "N/A"
    plt.title(
        f"Payoff: {TICKER} ${int(K)} Call\n"
        f"(Entry: ${entry} via {source}, IV: {iv_pct}, Exp: {expiry})",
        fontsize=12
    )
    plt.xlabel("Underlying Price @ Expiry")
    plt.ylabel("P/L per Contract ($)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    print(f"\n✅ Data exported to {csv_path}")
