import yfinance as yf
import pandas as pd
from datetime import datetime

# Expanded dividend-heavy tickers
tickers = [
    "KO", "PG", "PEP", "T", "VZ", "XOM", "CVX", "JNJ", "MMM", "WBA", "IBM", "DOW", "PFE", "MO",
    "O", "MAIN", "WMT", "HD", "LMT", "PM", "BMY", "ABBV", "RTX", "DUK", "SO", "ED", "NHI", "VICI"
]

results = []

for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        pe = info.get("trailingPE", None)
        dividend_yield = info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0
        eps = info.get("trailingEps", 0)

        # Log for debugging
        print(f"{ticker} -> PE: {pe}, Yield: {dividend_yield:.2f}%, EPS: {eps}")

        if pe and pe < 25 and dividend_yield > 2 and eps > 0:
            results.append({
                "Ticker": ticker,
                "PE Ratio": pe,
                "Dividend Yield (%)": dividend_yield,
                "EPS": eps,
                "Sector": info.get("sector", ""),
                "Industry": info.get("industry", "")
            })

    except Exception as e:
        print(f"Failed for {ticker}: {e}")

df = pd.DataFrame(results)
df.to_csv("../output/dividend_value_candidates.csv", index=False)

print(f"\n✅ {len(results)} value picks saved to output/dividend_value_candidates.csv")
