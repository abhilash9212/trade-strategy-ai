import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

tickers = ["AAPL", "MSFT", "TSLA", "AMD", "NVDA", "META"]
output = []

for ticker in tickers:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5d")
    
    if hist.empty:
        continue

    current_price = hist["Close"].iloc[-1]
    strike_price = round(current_price * 0.90, 2)  # simulate ~10% OTM
    est_premium = round(current_price * 0.02, 2)   # simulate ~2% premium
    roi = round((est_premium / (strike_price * 100)) * 100, 2)

    output.append({
        "Ticker": ticker,
        "Current Price": current_price,
        "Strike": strike_price,
        "Est Premium": est_premium,
        "Days to Expiry": 14,
        "Est ROI": f"{roi}%",
        "Rationale": "Simulated CSP: ~10% OTM, ~2% premium"
    })

df = pd.DataFrame(output)
df["Date Generated"] = datetime.today().strftime('%Y-%m-%d')
df.to_csv("../output/csp_simulated_candidates.csv", index=False)
print("✅ Simulated CSP picks saved to output/csp_simulated_candidates.csv")
 
