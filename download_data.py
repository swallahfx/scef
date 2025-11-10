import requests
import pandas as pd
import time
import os

# Your Alpha Vantage API Key
API_KEY = "P33DZM40PY8HN3I9"

# Function to download stock data
def download_stock_data(symbol, output_dir="market_data"):
    print(f"Downloading stock data for {symbol}...")
    
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "full",  # get full data (up to 20 years)
        "datatype": "json",
        "apikey": API_KEY
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{symbol}_daily.csv")
    
    # Make API request
    response = requests.get(base_url, params=params)
    data = response.json()
    
    # Check for error messages
    if "Error Message" in data:
        print(f"Error: {data['Error Message']}")
        return None
    
    if "Time Series (Daily)" not in data:
        print(f"Error: Could not get data for {symbol}")
        print(data)
        return None
    
    # Convert to DataFrame
    time_series = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(time_series, orient="index")
    
    # Convert index to datetime and sort
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Rename columns to match SCEF format
    df.columns = [col.split(". ")[1] for col in df.columns]
    df.rename(columns={
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close", 
        "volume": "volume"
    }, inplace=True)
    
    # Convert string values to float/int
    for col in df.columns:
        if col != 'volume':
            df[col] = df[col].astype(float)
        else:
            df[col] = df[col].astype(int)
    
    # Save to CSV
    df.to_csv(output_file)
    print(f"Data for {symbol} saved to {output_file}")
    return output_file

# Function to download forex data
def download_forex_data(from_currency, to_currency, output_dir="market_data"):
    print(f"Downloading forex data for {from_currency}/{to_currency}...")
    
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_DAILY",
        "from_symbol": from_currency,
        "to_symbol": to_currency,
        "outputsize": "full",  # get full data (up to 20 years)
        "datatype": "json",
        "apikey": API_KEY
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{from_currency}{to_currency}_daily.csv")
    
    # Make API request
    response = requests.get(base_url, params=params)
    data = response.json()
    
    # Check for error messages
    if "Error Message" in data:
        print(f"Error: {data['Error Message']}")
        return None
    
    if "Time Series FX (Daily)" not in data:
        print(f"Error: Could not get data for {from_currency}/{to_currency}")
        print(data)
        return None
    
    # Convert to DataFrame
    time_series = data["Time Series FX (Daily)"]
    df = pd.DataFrame.from_dict(time_series, orient="index")
    
    # Convert index to datetime and sort
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Rename columns to match SCEF format
    df.columns = [col.split(". ")[1] for col in df.columns]
    
    # Forex data doesn't have volume, so create a dummy volume column
    df['volume'] = 0
    
    # Create proper OHLCV format by duplicating close price for missing columns
    if 'open' not in df.columns:
        df['open'] = df['close']
    if 'high' not in df.columns:
        df['high'] = df['close']
    if 'low' not in df.columns:
        df['low'] = df['close']
    
    # Reorder columns to match SCEF format
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    # Convert string values to float/int
    for col in df.columns:
        if col != 'volume':
            df[col] = df[col].astype(float)
    
    # Save to CSV
    df.to_csv(output_file)
    print(f"Data for {from_currency}/{to_currency} saved to {output_file}")
    return output_file

# Main function with menu
if __name__ == "__main__":
    print("=======================================")
    print("  SCEF Market Data Downloader")
    print("=======================================")
    print("Select data type to download:")
    print("1. Stock data")
    print("2. Forex data")
    print("3. Both")
    
    choice = input("Enter your choice (1-3): ")
    
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    forex_pairs = [
        ("EUR", "USD"),  # Euro to US Dollar
        ("USD", "JPY"),  # US Dollar to Japanese Yen
        ("GBP", "USD"),  # British Pound to US Dollar
        ("USD", "CHF"),  # US Dollar to Swiss Franc
        ("AUD", "USD")   # Australian Dollar to US Dollar
    ]
    
    if choice == "1":
        # Download stocks only
        print("Starting stock data download...")
        results = {}
        for symbol in stocks:
            file_path = download_stock_data(symbol)
            if file_path:
                results[symbol] = file_path
            time.sleep(15)  # Wait 15 seconds between requests
        
    elif choice == "2":
        # Download forex only
        print("Starting forex data download...")
        results = {}
        for from_currency, to_currency in forex_pairs:
            file_path = download_forex_data(from_currency, to_currency)
            if file_path:
                results[f"{from_currency}/{to_currency}"] = file_path
            time.sleep(15)  # Wait 15 seconds between requests
    
    elif choice == "3":
        # Download both
        print("Starting combined data download...")
        results = {}
        
        print("Downloading stocks...")
        for symbol in stocks:
            file_path = download_stock_data(symbol)
            if file_path:
                results[symbol] = file_path
            time.sleep(15)
        
        print("Downloading forex pairs...")
        for from_currency, to_currency in forex_pairs:
            file_path = download_forex_data(from_currency, to_currency)
            if file_path:
                results[f"{from_currency}/{to_currency}"] = file_path
            time.sleep(15)
    
    else:
        print("Invalid choice. Please run again and select 1, 2, or 3.")
        exit()
    
    # Print summary
    print("\nDownload Summary:")
    for name, file_path in results.items():
        print(f"✅ {name}: {file_path}")
    
    print("\nData download complete. You can now upload these files to SCEF.")