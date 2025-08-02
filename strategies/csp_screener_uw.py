 import requests
import pandas as pd
from datetime import datetime

# === CONFIG ===
API_KEY = "your_unusual_whales_api_key"
TICKERS = ["AAPL", "MSFT", "TSLA", "AMD", "NVDA", "META"]
MIN_PREMIUM = 1.0
DELTA_RANGE = (0.25, 0.35)
IVR_THRESHOLD = 50

# === API CALL ===
url = "https://api.unusualwhales.com/option-trades/flow_alerts"
headers = {"Authorization": f"Bearer {API_KEY}"}
params = {
    "symbols": ",".join(TICKERS),
    "limit": 100,
    "option_type": "put"
}

response = requests.get(url, headers=headers, params=params)
if response.status_code != 200:
    print(f"Error: {response.status_code} - {response.text}")
    exit()

alerts = pd.DataFrame(response.json().get("results", []))

if alerts.empty:
    print("No flow alerts returned.")
    exit()

# === FILTERING LOGIC ===
filtered = alerts[
    (alerts["option_type"] == "put") &
    (alerts["delta"].between(DELTA_RANGE[0], DELTA_RANGE[1])) &
    (alerts["premium"] >= MIN_PREMIUM) &
    (alerts["iv_rank"] >= IVR_THRESHOLD)
]

if filtered.empty:
    print("No CSP candidates met the filter criteria.")
else:
    filtered = filtered[[
        "symbol", "strike_price", "premium", "expiration", "delta", "iv_rank", "underlying_price", "timestamp"
    ]]

    filtered["strategy"] = "Cash Secured Put"
    filtered["date_generated"] = datetime.now().strftime('%Y-%m-%d')
    filtered.to_csv("./output/csp_flow_candidates.csv", index=False)
    print(f"✅ {len(filtered)} CSP picks saved to output/csp_flow_candidates.csv")
