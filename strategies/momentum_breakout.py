import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# Create output directory if it doesn't exist
os.makedirs("../output", exist_ok=True)

tickers = ["AAPL", "MSFT", "TSLA", "AMD", "NVDA", "META", "KO", "XOM", "JPM", "NFLX"]
momentum_picks = []

for ticker in tickers:
    df = yf.download(ticker, period="6mo")

    if df.empty:
        continue

    # Calculate indicators
    df["20EMA"] = df["Close"].ewm(span=20).mean()
    df["50EMA"] = df["Close"].ewm(span=50).mean()
    df["AvgVol20"] = df["Volume"].rolling(20).mean()

    # RSI calculation
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Get the last row
    last_row = df.iloc[-1]

    # Apply filters
    if (
        float(last_row["20EMA"]) > float(last_row["50EMA"])
        and float(last_row["Volume"]) > 1.5 * float(df["AvgVol20"].iloc[-1])
        and 50 < float(last_row["RSI"]) < 70
    ):
        momentum_picks.append({
            "Ticker": ticker,
            "Price": round(last_row["Close"], 2),
            "Volume": int(last_row["Volume"]),
            "20EMA": round(last_row["20EMA"], 2),
            "50EMA": round(last_row["50EMA"], 2),
            "RSI": round(last_row["RSI"], 2),
            "Date": datetime.today().strftime('%Y-%m-%d'),
            "Rationale": "EMA20 > EMA50, Volume Surge, RSI 50–70"
        })

# Export results
df_out = pd.DataFrame(momentum_picks)
df_out.to_csv("../output/momentum_breakouts.csv", index=False)
print(f"✅ {len(df_out)} breakout candidates saved to output/momentum_breakouts.csv")
