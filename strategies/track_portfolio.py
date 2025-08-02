import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# Simulated holdings
holdings = [
    {"ticker": "KO", "buy_price": 58.00, "quantity": 50, "buy_date": "2024-11-01"},
    {"ticker": "XOM", "buy_price": 105.00, "quantity": 30, "buy_date": "2024-12-15"},
    {"ticker": "PEP", "buy_price": 160.00, "quantity": 40, "buy_date": "2025-01-10"},
]

tickers = [h["ticker"] for h in holdings]
df = yf.download(tickers, period="1d", auto_adjust=True)

# Debug print - confirm structure
print(">>> Columns:", df.columns)

# Handle single ticker vs multiple ticker cases
if isinstance(df.columns, pd.MultiIndex):
    latest_prices = df["Close"].iloc[-1]  # With auto_adjust=True, Close is already adjusted
else:
    latest_prices = pd.Series({tickers[0]: df["Close"].iloc[-1]})

# Build output
results = []
for h in holdings:
    current_price = latest_prices[h["ticker"]]
    pnl = (current_price - h["buy_price"]) * h["quantity"]
    return_pct = ((current_price / h["buy_price"]) - 1) * 100
    results.append({
        "Ticker": h["ticker"],
        "Buy Price": h["buy_price"],
        "Quantity": h["quantity"],
        "Current Price": round(current_price, 2),
        "PnL ($)": round(pnl, 2),
        "Return (%)": round(return_pct, 2),
        "Buy Date": h["buy_date"]
    })

df_out = pd.DataFrame(results)

# Save to output
#os.makedirs("output", exist_ok=True)
filename = f"../output/portfolio_tracker_{datetime.today().strftime('%Y%m%d')}.csv"
df_out.to_csv(filename, index=False)

print(f"✅ Portfolio saved to {filename}")
