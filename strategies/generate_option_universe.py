# generate_option_universe.py
# Scans all upcoming expirations and a range of strikes (90%-120% of spot) for the best long-call opportunity

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime
import matplotlib.pyplot as plt

# Black-Scholes formula for call options
def bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

if __name__ == '__main__':
    TICKER = "CPRT"
    RFR = 0.05                   # Risk-free rate
    OPTION_QTY = 1               # Contracts
    SPOT_LOOKBACK = 1            # in days
    STRIKE_RANGE = (0.90, 1.20)  # 90% to 120% of spot

    tk = yf.Ticker(TICKER)
    hist = tk.history(period=f"{SPOT_LOOKBACK}d")
    if hist.empty:
        raise SystemExit(f"[ERROR] No spot price for {TICKER}")
    spot = hist['Close'].iloc[-1]

    today = datetime.utcnow().date()
    expiries = [e for e in tk.options if (pd.to_datetime(e).date() - today).days > 0]
    if not expiries:
        raise SystemExit("[ERROR] No future expirations found")

    results = []
    for exp in expiries:
        T_days = (pd.to_datetime(exp).date() - today).days
        T = T_days / 365.0
        chain = tk.option_chain(exp).calls
        # Filter strikes within desired range
        valid = chain[(chain.strike >= spot * STRIKE_RANGE[0]) & (chain.strike <= spot * STRIKE_RANGE[1])]
        for _, row in valid.iterrows():
            K = row.strike
            # live lastPrice
            lp = getattr(row, 'lastPrice', np.nan) or np.nan
            bid = getattr(row, 'bid', np.nan) or np.nan
            ask = getattr(row, 'ask', np.nan) or np.nan
            iv = getattr(row, 'impliedVolatility', np.nan) or np.nan
            # determine premium
            if lp > 0:
                premium = float(lp)
            elif bid > 0 and ask >= bid:
                premium = float((bid + ask) / 2)
            else:
                premium = bs_call_price(spot, K, T, RFR, iv)
            # payoff profile
            prices = np.linspace(0.5 * spot, 1.5 * spot, 200)
            payoff = (np.maximum(prices - K, 0) - premium) * OPTION_QTY * 100
            max_payoff = float(payoff.max())
            breakeven = float(K + premium)
            results.append({
                'expiry': exp,
                'strike': K,
                'premium': round(premium, 2),
                'iv': round(iv * 100, 2) if iv>0 else None,
                'breakeven': round(breakeven, 2),
                'max_payoff': round(max_payoff, 2),
                'days_to_expiry': T_days
            })

    df = pd.DataFrame(results)
    if df.empty:
        print("⚠️ No valid option candidates found.")
    else:
        df = df.sort_values('max_payoff', ascending=False)
        out_csv = '../output/best_option_candidates.csv'
        df.to_csv(out_csv, index=False)
        print(f"✅ Saved {len(df)} candidates to {out_csv}")
        # Show top 3
        print("\n🥇 Top 3 candidates:")
        print(df.head(3).to_string(index=False))

        # Plot payoff for the top candidate
        top = df.iloc[0]
        exp = top.expiry
        K = top.strike
        # regenerate payoff for plot
        T = top.days_to_expiry / 365.0
        row = tk.option_chain(exp).calls.loc[tk.option_chain(exp).calls.strike == K].iloc[0]
        lp = getattr(row, 'lastPrice', np.nan) or np.nan
        bid = getattr(row, 'bid', np.nan) or np.nan
        ask = getattr(row, 'ask', np.nan) or np.nan
        iv = getattr(row, 'impliedVolatility', np.nan) or np.nan
        if lp>0:
            premium = float(lp)
        elif bid>0 and ask>=bid:
            premium = float((bid+ask)/2)
        else:
            premium = bs_call_price(spot, K, T, RFR, iv)
        prices = np.linspace(0.5 * spot, 1.5 * spot, 200)
        payoff = (np.maximum(prices - K, 0) - premium) * OPTION_QTY * 100
        plt.figure(figsize=(8,5))
        plt.plot(prices, payoff, label=f"LONG {TICKER} {int(K)}C")
        plt.axhline(0, color='black', lw=1)
        plt.title(f"Top: {TICKER} ${int(K)} Call | Exp {exp}\n"
                  f"Prem: ${premium:.2f}, IV: {iv*100:.2f}%")
        plt.xlabel("Underlying Price @ Expiry")
        plt.ylabel("P/L per Contract ($)")
        plt.grid(True, ls='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()
