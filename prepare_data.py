# # save_as_prepare_data.py
# import os
# import json
# import shutil
# import pandas as pd
# import uuid

# # Source and destination directories
# SOURCE_DIR = "/Users/mac/Downloads/scef/market_data"
# DEST_DIR = "/Users/mac/Downloads/scef/web/storage/data"

# # Create destination directory if it doesn't exist
# os.makedirs(DEST_DIR, exist_ok=True)

# # Process each CSV file
# for filename in os.listdir(SOURCE_DIR):
#     if filename.endswith('.csv'):
#         # Create a unique ID
#         data_id = str(uuid.uuid4())
        
#         # Read CSV to get basic info
#         data_path = os.path.join(SOURCE_DIR, filename)
#         data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
#         # Extract name from filename (remove _daily.csv)
#         name = filename.replace('_daily.csv', '')
        
#         # Create metadata
#         metadata = {
#             "id": data_id,
#             "name": f"{name} Market Data",
#             "filename": filename,
#             "date_range": f"{data.index[0]} to {data.index[-1]}",
#             "num_rows": len(data)
#         }
        
#         # Save metadata to JSON
#         with open(os.path.join(DEST_DIR, f"{data_id}.json"), 'w') as f:
#             json.dump(metadata, f, indent=2)
        
#         # Copy CSV file with new ID-based name
#         shutil.copy(data_path, os.path.join(DEST_DIR, f"{data_id}.csv"))
        
#         print(f"Processed {filename} → ID: {data_id}")

# print("Done! All data files have been prepared for SCEF.")



import requests
import json
import numpy as np

# Define a very simple strategy
def simple_strategy():
    return {
        "name": "API Test Strategy",
        "description": "Simple strategy for testing the API",
        "components": [
            {
                "name": "always_buy",
                "type": "signal",
                "function": "lambda data, context: np.ones(len(data))",
                "parameters": {}
            },
            {
                "name": "full_allocation",
                "type": "allocation",
                "function": "lambda signal, data, context: signal",
                "parameters": {}
            },
            {
                "name": "no_risk_control",
                "type": "risk_control",
                "function": "lambda position, data, context: position",
                "parameters": {}
            },
            {
                "name": "simple_execution",
                "type": "execution",
                "function": "lambda trades, data, context: trades",
                "parameters": {}
            }
        ]
    }

# Create the strategy via API
response = requests.post(
    "http://localhost:8011/api/strategies",
    json=simple_strategy()
)

print("======hh77ghh", response.json())

strategy_id = response.json()["strategy_id"]
print(f"Created strategy with ID: {strategy_id}")

# Get the first available data ID
response = requests.get("http://localhost:8011/api/data")
data_id = response.json()[0]["id"]
print(f"Using data with ID: {data_id}")

# Run a backtest
response = requests.post(
    "http://localhost:8011/api/backtest",
    json={
        "strategy_id": strategy_id,
        "data_id": data_id,
        "config": {
            "starting_capital": 100000,
            "commission_rate": 0.0,
            "slippage_model": "none",
            "track_trade_history": True
        }
    }
)

print(f"Backtest response: {json.dumps(response.json(), indent=2)}")