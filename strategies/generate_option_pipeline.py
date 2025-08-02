# File: strategies/generate_option_pipeline.py

import os
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) CONFIG & HELPERS
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pure-Pandas RSI
def compute_rsi(series, window=14):
    delta = series.diff().dropna()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=(window - 1), adjust=False).mean()
    ma_down = down.ewm(com=(window - 1), adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

# Pure-Pandas MACD (12,26,9)
def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig

# Bollinger Bands
def compute_bb(series, window=20, num_std=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return sma + num_std*std, sma - num_std*std

# 2) TICKER LIST
tickers = [
    # add as many as you like...
    "AAPL","MSFT","GOOGL","AMZN","TSLA","SPY","QQQ",
    "CPRT","INTC","BMY","DUK","SO","ED","GIS","KMB",
    "D","NEE","AEP","XEL","SPG","VZ","SYY","DOW","CL","EXC"
]

# 3) BUILD OPTION UNIVERSE
univ = []
today = datetime.datetime.utcnow().date()

for t in tickers:
    tk = yf.Ticker(t)
    for exp in tk.options:
        df_chain = tk.option_chain(exp).calls
        df_chain = df_chain[['strike','lastPrice','impliedVolatility']].copy()
        df_chain['ticker'] = t
        df_chain['expiry'] = exp
        df_chain['days_to_expiry'] = (pd.to_datetime(exp).date() - today).days
        univ.append(df_chain)

univ = pd.concat(univ, ignore_index=True)
univ.to_csv(os.path.join(OUTPUT_DIR, 'option_universe.csv'), index=False)
print(f"✅ Saved {len(univ)} contracts to ../output/option_universe.csv")

# 4) BUILD ALL CALL DEBIT SPREADS (adjacent strikes)
spreads = []
for grp, df in univ.groupby(['ticker','expiry']):
    tkr, exp = grp
    df = df.sort_values('strike').reset_index()
    for i in range(len(df)-1):
        low, high = df.loc[i], df.loc[i+1]
        net_debit = low.lastPrice - high.lastPrice
        width = high.strike - low.strike
        max_payoff = max(0, width*100 - net_debit*100)
        breakeven = low.strike + net_debit
        spreads.append({
            'ticker': tkr,
            'expiry': exp,
            'strike_low': low.strike,
            'strike_high': high.strike,
            'net_debit': round(net_debit,2),
            'iv_low': low.impliedVolatility,
            'iv_high': high.impliedVolatility,
            'days_to_expiry': low.days_to_expiry,
            'max_payoff': round(max_payoff,2),
            'breakeven': round(breakeven,2),
        })

spreads = pd.DataFrame(spreads)
spreads.to_csv(os.path.join(OUTPUT_DIR, 'option_strategies.csv'), index=False)
print(f"✅ Saved {len(spreads)} debit spreads to ../output/option_strategies.csv")

# 5) TECHNICAL SCREEN ON UNDERLYINGS
candidates = []
for t in tickers:
    # grab price series
    data = yf.download(t, period="6mo", interval="1d", progress=False)['Close'].dropna()
    if data.empty: 
        continue

    # compute indicators
    rsi = compute_rsi(data)
    macd, sig = compute_macd(data)
    macd_diff = macd - sig
    bb_upper, bb_lower = compute_bb(data)

    last_rsi       = rsi.iloc[-1]
    last_macd_diff = macd_diff.iloc[-1]
    last_price     = data.iloc[-1]
    last_bb_lower  = bb_lower.iloc[-1]

    if last_rsi < 30 and last_macd_diff > 0 and last_price < last_bb_lower:
        candidates.append(t)

print(f"⚡️ {len(candidates)} tickers passed tech (RSI<30, MACD>0, Price<BB lower): {candidates}")

# 6) FILTER SPREAD UNIVERSE & PICK BEST
df_strat = spreads[spreads['ticker'].isin(candidates)]
if df_strat.empty:
    print("⚠️ No candidates matched all filters.")
    exit()

best = df_strat.sort_values('max_payoff', ascending=False).iloc[0:1]
best.to_csv(os.path.join(OUTPUT_DIR, 'trade_alerts_best.csv'), index=False)
print("\n💡 BEST TRADE CANDIDATE:\n")
print(best.to_string(index=False))
print(f"\n✅ Saved results to ../output/trade_alerts_best.csv")

# 7) PAYOFF DIAGRAM
row = best.iloc[0]
K1, K2 = row.strike_low, row.strike_high
debit   = row.net_debit
label   = f"{row.ticker} {K1}->{K2} Debit"

S = np.linspace(K1*0.5, K2*2, 200)
payoff = np.maximum(S - K1, 0) - np.maximum(S - K2, 0) - debit

plt.figure(figsize=(8,5))
plt.plot(S, payoff*100, label=f"{label}", linewidth=2)
plt.axhline(0, color='k', lw=1)
plt.title(f"Payoff: {label}\nEntry: ${debit:.2f}, BE: {row.breakeven:.2f}, Exp: {row.expiry}")
plt.xlabel("Underlying Price @ Expiry")
plt.ylabel("P/L per Contract ($)")
plt.legend()
plt.grid(True)
fn = os.path.join(OUTPUT_DIR, 'best_strategy_payoff.png')
plt.savefig(fn, bbox_inches='tight')
print(f"📈 Payoff diagram saved to {fn}")
