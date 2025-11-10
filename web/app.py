"""
FastAPI web interface for the Strategy Composition and Evaluation Framework.

This module provides a web-based interface for creating, testing, and
visualizing trading strategies using the SCEF framework.
"""
import os
import json
import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
import io
import base64

from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Import SCEF components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web.ml_api import include_ml_router

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.strategy_dsl import Strategy, create_indicator, create_signal, create_allocation, create_risk_control, create_execution, create_post_trade
from src.indicators import sma, ema, macd, rsi, bollinger_bands
from src.backtester import Backtester, BacktestConfig, BacktestResult
from src.online_learning import StrategyBandit, OnlineLearner
from src.production_engine import ProductionEngine, StrategyConfig
import os
import pickle
import json

# In-memory storage for strategies and backtest results
strategies = {}  # IMPORTANT: Define this BEFORE the load functions
backtest_results = {}
market_data = {}  # IMPORTANT: Define this BEFORE the load functions
production_engine = None
deployed_strategies = {}

# Create storage directories if they don't exist
STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage')
STRATEGIES_DIR = os.path.join(STORAGE_DIR, 'strategies')
DATA_DIR = os.path.join(STORAGE_DIR, 'data')
RESULTS_DIR = os.path.join(STORAGE_DIR, 'results')

os.makedirs(STRATEGIES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# Load market data
def load_market_data():
    global market_data  # This line is critical
    if os.path.exists(DATA_DIR):
        for filename in os.listdir(DATA_DIR):
            if filename.endswith('.json'):
                data_id = filename[:-5]  # Remove .json extension
                try:
                    with open(os.path.join(DATA_DIR, filename), 'r') as f:
                        data_info = json.load(f)
                    
                    data_path = os.path.join(DATA_DIR, f"{data_id}.csv")
                    if os.path.exists(data_path):
                        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
                        data_info['data'] = data
                        market_data[data_id] = data_info
                except Exception as e:
                    print(f"Error loading market data {data_id}: {e}")

# Load strategies
def load_strategies():
    global strategies  # This line is critical
    if os.path.exists(STRATEGIES_DIR):
        for filename in os.listdir(STRATEGIES_DIR):
            if filename.endswith('.json'):
                strategy_id = filename[:-5]  # Remove .json extension
                try:
                    with open(os.path.join(STRATEGIES_DIR, filename), 'r') as f:
                        strategy_info = json.load(f)
                    
                    with open(os.path.join(STRATEGIES_DIR, f"{strategy_id}.pkl"), 'rb') as f:
                        strategy_obj = pickle.load(f)
                    
                    strategy_info['strategy_obj'] = strategy_obj
                    strategies[strategy_id] = strategy_info
                except Exception as e:
                    print(f"Error loading strategy {strategy_id}: {e}")

# Call loading functions at startup
load_market_data()
load_strategies()



# Import routes
from web.routes import register_routes

# Create FastAPI app
app = FastAPI(
    title="SCEF Web Interface",
    description="Web interface for the Strategy Composition and Evaluation Framework",
    version="1.0.0")
# Include ML router
include_ml_router(app)

# Register routes
register_routes(app)



# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# In-memory storage for strategies and backtest results
strategies = {}
backtest_results = {}
market_data = {}
production_engine = None
deployed_strategies = {}

# Models for API requests/responses
class StrategyComponent(BaseModel):
    name: str
    type: str  # 'indicator', 'signal', 'allocation', 'risk_control', 'execution', 'post_trade'
    function: str
    parameters: Dict[str, Any] = {}

class StrategyDefinition(BaseModel):
    name: str
    description: str = ""
    components: List[StrategyComponent] = []

class BacktestRequest(BaseModel):
    strategy_id: str
    data_id: str
    config: Dict[str, Any] = {}

class DeployRequest(BaseModel):
    strategy_id: str
    data_id: str
    config: Dict[str, Any] = {}

# Helper functions
def create_sample_data(days=500, volatility=0.01):
    """Create sample market data for demonstration."""
    np.random.seed(42)  # For reproducibility
    
    # Generate dates
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate prices
    returns = np.random.normal(0.0005, volatility, size=len(dates))
    prices = 100 * (1 + returns).cumprod()
    
    # Add some trend and seasonality
    t = np.arange(len(dates))
    trend = 0.1 * np.sin(t / 100) + 0.05 * np.sin(t / 50)
    prices = prices * (1 + trend)
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': prices * (1 - 0.002 * np.random.random(len(dates))),
        'high': prices * (1 + 0.004 * np.random.random(len(dates))),
        'low': prices * (1 - 0.004 * np.random.random(len(dates))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, size=len(dates))
    }, index=dates)
    
    return data

def get_function_by_name(name: str):
    """Get a function by name."""
    function_map = {
        "sma": sma,
        "ema": ema,
        "macd": macd,
        "rsi": rsi,
        "bollinger_bands": bollinger_bands,
        "dual_ma_crossover": dual_ma_crossover_signal,
        "rsi_filter": rsi_filter,
        "macd_signal": macd_signal,
        "position_sizer": position_sizer,
        "volatility_position_sizer": volatility_position_sizer,
        "max_position_limit": max_position_limit,
        "stop_loss": stop_loss,
        "simple_execution": simple_execution,
        "trade_analyzer": trade_analyzer
    }
    
    return function_map.get(name)

def create_strategy_from_definition(definition: StrategyDefinition) -> Strategy:
    """Create a Strategy object from a strategy definition."""
    strategy = Strategy(definition.name)
    
    for component in definition.components:
        func = get_function_by_name(component.function)
        if func is None:
            raise ValueError(f"Unknown function: {component.function}")
            
        if component.type == "indicator":
            strategy.add_indicator(create_indicator(component.name, func, **component.parameters))
        elif component.type == "signal":
            strategy.add_signal(create_signal(component.name, func, **component.parameters))
        elif component.type == "allocation":
            strategy.add_allocation(create_allocation(component.name, func, **component.parameters))
        elif component.type == "risk_control":
            strategy.add_risk_control(create_risk_control(component.name, func, **component.parameters))
        elif component.type == "execution":
            strategy.add_execution(create_execution(component.name, func, **component.parameters))
        elif component.type == "post_trade":
            strategy.add_post_trade(create_post_trade(component.name, func, **component.parameters))
        else:
            raise ValueError(f"Unknown component type: {component.type}")
    
    return strategy

def figure_to_base64(fig):
    """Convert a plotly figure to a base64 string."""
    buffer = io.BytesIO()
    fig.write_image(buffer, format="png")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def plot_equity_curve(result: BacktestResult):
    """Create a plotly figure of the equity curve."""
    fig = go.Figure()
    
    # Add equity curve
    if result.equity_curve is not None:
        fig.add_trace(go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve,
            mode='lines',
            name='Equity'
        ))
    
    # Add drawdown
    if result.drawdown_curve is not None:
        fig.add_trace(go.Scatter(
            x=result.drawdown_curve.index,
            y=result.drawdown_curve * 100,
            mode='lines',
            name='Drawdown %',
            yaxis='y2'
        ))
    
    # Update layout
    fig.update_layout(
        title='Equity Curve',
        xaxis_title='Date',
        yaxis_title='Equity',
        yaxis2=dict(
            title='Drawdown %',
            overlaying='y',
            side='right',
            range=[min(result.drawdown_curve) * 100 * 1.5 if result.drawdown_curve is not None else 0, 0]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_positions_over_time(result: BacktestResult, data: pd.DataFrame):
    """Create a plotly figure of positions over time."""
    fig = go.Figure()
    
    # Add positions
    if result.positions is not None and 'position' in result.positions.columns:
        fig.add_trace(go.Scatter(
            x=result.positions.index,
            y=result.positions['position'],
            mode='lines',
            name='Position'
        ))
    
    # Add price overlay
    if 'close' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['close'],
            mode='lines',
            name='Price',
            yaxis='y2'
        ))
    
    # Update layout
    fig.update_layout(
        title='Positions Over Time',
        xaxis_title='Date',
        yaxis_title='Position Size',
        yaxis2=dict(
            title='Price',
            overlaying='y',
            side='right'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Define strategy component functions
def dual_ma_crossover_signal__(data, context, fast_period=20, slow_period=50, price_col='close'):
    """Dual moving average crossover signal."""
    # Get prices
    prices = data[price_col].values
    
    # Calculate fast and slow moving averages
    if 'indicators' in context and 'fast_ma' in context['indicators'] and 'slow_ma' in context['indicators']:
        fast_ma = context['indicators']['fast_ma']
        slow_ma = context['indicators']['slow_ma']
    else:
        fast_ma = sma(prices, context, window=fast_period)
        slow_ma = sma(prices, context, window=slow_period)
    
    # Calculate signal: 1 when fast MA crosses above slow MA, -1 when fast MA crosses below slow MA
    signal = np.zeros_like(prices)
    
    for i in range(slow_period, len(prices)):
        if fast_ma[i-1] <= slow_ma[i-1] and fast_ma[i] > slow_ma[i]:
            # Bullish crossover
            signal[i] = 1
        elif fast_ma[i-1] >= slow_ma[i-1] and fast_ma[i] < slow_ma[i]:
            # Bearish crossover
            signal[i] = -1
    
    return signal


def dual_ma_crossover_signal(data, context, fast_period=20, slow_period=50, price_col='close'):
    """Dual moving average crossover signal with debugging."""
    # Get prices
    prices = data[price_col].values
    
    # Print debug information about the data
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns}")
    print(f"First 5 prices: {prices[:5]}")
    print(f"Last 5 prices: {prices[-5:]}")
    
    # Calculate fast and slow moving averages
    if 'indicators' in context and 'fast_ma' in context['indicators'] and 'slow_ma' in context['indicators']:
        fast_ma = context['indicators']['fast_ma']
        slow_ma = context['indicators']['slow_ma']
        print("Using pre-calculated indicators from context")
    else:
        print("Calculating indicators directly")
        fast_ma = sma(prices, context, window=fast_period)
        slow_ma = sma(prices, context, window=slow_period)
    
    # Print information about the moving averages
    print(f"Fast MA first 5 values after warmup: {fast_ma[fast_period:fast_period+5]}")
    print(f"Slow MA first 5 values after warmup: {slow_ma[slow_period:slow_period+5]}")
    
    # Check if we have enough data for crossovers
    if len(prices) <= slow_period:
        print(f"Warning: Not enough data points ({len(prices)}) for slow MA period ({slow_period})")
        return np.zeros_like(prices)
    
    # Calculate signal: 1 when fast MA crosses above slow MA, -1 when fast MA crosses below slow MA
    signal = np.zeros_like(prices)
    crossover_count = 0
    
    for i in range(slow_period, len(prices)):
        if fast_ma[i-1] <= slow_ma[i-1] and fast_ma[i] > slow_ma[i]:
            # Bullish crossover
            signal[i] = 1
            crossover_count += 1
        elif fast_ma[i-1] >= slow_ma[i-1] and fast_ma[i] < slow_ma[i]:
            # Bearish crossover
            signal[i] = -1
            crossover_count += 1
    
    print(f"Total crossovers detected: {crossover_count}")
    print(f"Number of buy signals: {np.sum(signal > 0)}")
    print(f"Number of sell signals: {np.sum(signal < 0)}")
    
    return signal


def rsi_filter(data, context, threshold_low=30, threshold_high=70, price_col='close'):
    """RSI filter signal."""
    # Get prices
    prices = data[price_col].values
    
    # Calculate RSI
    if 'indicators' in context and 'rsi' in context['indicators']:
        rsi_values = context['indicators']['rsi']
    else:
        rsi_values = rsi(prices, context)
    
    # Calculate signal: 1 when RSI crosses below threshold_low (oversold), -1 when RSI crosses above threshold_high (overbought)
    signal = np.zeros_like(prices)
    
    for i in range(1, len(prices)):
        if not np.isnan(rsi_values[i-1]) and not np.isnan(rsi_values[i]):
            if rsi_values[i-1] >= threshold_low and rsi_values[i] < threshold_low:
                # Oversold - bullish
                signal[i] = 1
            elif rsi_values[i-1] <= threshold_high and rsi_values[i] > threshold_high:
                # Overbought - bearish
                signal[i] = -1
    
    return signal

def macd_signal(data, context, fast_period=12, slow_period=26, signal_period=9, price_col='close'):
    """MACD signal."""
    # Get prices
    prices = data[price_col].values
    
    # Calculate MACD
    if 'indicators' in context and 'macd' in context['indicators']:
        macd_values = context['indicators']['macd']
        macd_line = macd_values['macd_line']
        signal_line = macd_values['signal_line']
    else:
        macd_result = macd(prices, context, fast_window=fast_period, slow_window=slow_period, signal_window=signal_period)
        macd_line = macd_result['macd_line']
        signal_line = macd_result['signal_line']
    
    # Calculate signal: 1 when MACD crosses above signal line, -1 when MACD crosses below signal line
    signal = np.zeros_like(prices)
    
    for i in range(slow_period + signal_period, len(prices)):
        if not np.isnan(macd_line[i-1]) and not np.isnan(signal_line[i-1]) and \
           not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]):
            if macd_line[i-1] <= signal_line[i-1] and macd_line[i] > signal_line[i]:
                # Bullish crossover
                signal[i] = 1
            elif macd_line[i-1] >= signal_line[i-1] and macd_line[i] < signal_line[i]:
                # Bearish crossover
                signal[i] = -1
    
    return signal

def position_sizer(signal, data, context, max_position=1.0):
    """Position sizer based on signal strength."""
    # Simple position sizing: full position based on signal
    return signal * max_position

def volatility_position_sizer(signal, data, context, window=20, target_volatility=0.01, max_position=1.0):
    """Position sizer adjusting for volatility."""
    # Calculate volatility
    returns = np.log(data['close'] / data['close'].shift(1)).dropna().values
    
    # Calculate rolling volatility
    vol = np.zeros_like(signal)
    for i in range(window, len(vol)):
        vol[i] = np.std(returns[i-window:i])
    
    # Scale position inversely with volatility
    position = np.zeros_like(signal)
    for i in range(window, len(position)):
        if vol[i] > 0:
            volatility_scalar = target_volatility / vol[i]
            position[i] = signal[i] * min(volatility_scalar, 1.0) * max_position
        else:
            position[i] = 0.0
    
    return position

def max_position_limit(position, data, context, max_position=1.0):
    """Limit maximum position size."""
    return np.clip(position, -max_position, max_position)

def stop_loss(position, data, context, stop_percent=0.05):
    """Apply stop loss to positions."""
    # Get current position from context
    current_position = context.get('current_position', 0.0)
    
    # Get current price and entry price
    current_price = data['close'].iloc[-1]
    entry_price = context.get('entry_price', current_price)
    
    # If position changes, update entry price
    if current_position != position[-1] and position[-1] != 0:
        context['entry_price'] = current_price
        return position
    
    # Check for stop loss
    if current_position > 0:  # Long position
        if current_price < entry_price * (1 - stop_percent):
            # Stop loss triggered
            position[-1] = 0
    elif current_position < 0:  # Short position
        if current_price > entry_price * (1 + stop_percent):
            # Stop loss triggered
            position[-1] = 0
    
    return position

def simple_execution(trades, data, context, slippage=0.001):
    """Simple execution model with slippage."""
    # Apply slippage
    trades_with_slippage = trades.copy()
    
    for i in range(len(trades)):
        if trades[i] > 0:  # Buy
            trades_with_slippage[i] *= (1 + slippage)
        elif trades[i] < 0:  # Sell
            trades_with_slippage[i] *= (1 - slippage)
    
    return trades_with_slippage

def trade_analyzer(trades, data, context, price_col='close'):
    """Analyze trades to calculate returns and metrics."""
    # Initialize returns array
    returns = np.zeros(len(trades))
    
    # Get prices
    prices = data[price_col].values
    
    # Get current position from context
    current_position = context.get('current_position', 0.0)
    
    # Calculate returns
    for i in range(len(trades)):
        # If there's a trade, calculate returns
        if trades[i] != 0:
            # Returns = position change * price change
            returns[i] = current_position * (prices[i] - prices[i-1]) / prices[i-1] if i > 0 else 0
            
            # Update position
            current_position += trades[i]
    
    # Update context
    context['current_position'] = current_position
    context['returns'] = returns
    
    return returns

# FastAPI route for home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# FastAPI route for creating a new strategy
@app.post("/api/strategies--", response_class=JSONResponse)
async def create_strategy__(strategy_def: StrategyDefinition):
    try:
        # Generate ID for strategy
        strategy_id = str(uuid.uuid4())
        
        # Create strategy
        strategy = create_strategy_from_definition(strategy_def)
        
        # Store strategy
        strategies[strategy_id] = {
            "id": strategy_id,
            "name": strategy_def.name,
            "description": strategy_def.description,
            "components": strategy_def.components,
            "strategy_obj": strategy
        }
        
        return {
            "status": "success",
            "message": f"Strategy '{strategy_def.name}' created successfully",
            "strategy_id": strategy_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/strategies", response_class=JSONResponse)
async def create_strategy(strategy_def: StrategyDefinition):
    try:
        # Generate ID for strategy
        strategy_id = str(uuid.uuid4())
        
        # Create strategy
        strategy = create_strategy_from_definition(strategy_def)
        
        # Store strategy info and object
        strategy_info = {
            "id": strategy_id,
            "name": strategy_def.name,
            "description": strategy_def.description,
            "components": [comp.dict() for comp in strategy_def.components],
            "strategy_obj": strategy
        }
        
        # Store in memory
        strategies[strategy_id] = strategy_info
        
        # Save to file (JSON for metadata, pickle for object)
        info_to_save = {k: v for k, v in strategy_info.items() if k != 'strategy_obj'}
        
        with open(os.path.join(STRATEGIES_DIR, f"{strategy_id}.json"), 'w') as f:
            json.dump(info_to_save, f)
        
        with open(os.path.join(STRATEGIES_DIR, f"{strategy_id}.pkl"), 'wb') as f:
            pickle.dump(strategy, f)
        
        return {
            "status": "success",
            "message": f"Strategy '{strategy_def.name}' created successfully",
            "strategy_id": strategy_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# FastAPI route for getting strategy details
@app.get("/api/strategies/{strategy_id}", response_class=JSONResponse)
async def get_strategy(strategy_id: str):
    if strategy_id not in strategies:
        raise HTTPException(status_code=404, detail=f"Strategy with ID {strategy_id} not found")
    
    strategy_info = strategies[strategy_id].copy()
    # Remove strategy object from response
    strategy_info.pop("strategy_obj", None)
    
    return strategy_info

# FastAPI route for getting all strategies
@app.get("/api/strategies", response_class=JSONResponse)
async def get_all_strategies():
    # Return all strategies without the strategy objects
    return [
        {k: v for k, v in strategy.items() if k != "strategy_obj"}
        for strategy in strategies.values()
    ]

# FastAPI route for uploading market data
@app.post("/api/data/upload--", response_class=JSONResponse)
async def upload_data__(
    data_name: str = Form(...),
    data_file: UploadFile = File(...)
):
    try:
        # Read data file
        contents = await data_file.read()
        
        # Parse data based on file type
        if data_file.filename.endswith('.csv'):
            data = pd.read_csv(io.BytesIO(contents), index_col=0, parse_dates=True)
        elif data_file.filename.endswith('.parquet'):
            data = pd.read_parquet(io.BytesIO(contents))
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Parquet files.")
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Generate ID for data
        data_id = str(uuid.uuid4())
        
        # Store data
        market_data[data_id] = {
            "id": data_id,
            "name": data_name,
            "filename": data_file.filename,
            "date_range": f"{data.index[0]} to {data.index[-1]}",
            "num_rows": len(data),
            "data": data
        }
        
        return {
            "status": "success",
            "message": f"Market data '{data_name}' uploaded successfully",
            "data_id": data_id,
            "summary": {
                "rows": len(data),
                "start_date": data.index[0].strftime('%Y-%m-%d'),
                "end_date": data.index[-1].strftime('%Y-%m-%d')
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/data/upload", response_class=JSONResponse)
async def upload_data(
    data_name: str = Form(...),
    data_file: UploadFile = File(...)
):
    try:
        # Read data file
        contents = await data_file.read()
        
        # Parse data based on file type
        if data_file.filename.endswith('.csv'):
            data = pd.read_csv(io.BytesIO(contents), index_col=0, parse_dates=True)
        elif data_file.filename.endswith('.parquet'):
            data = pd.read_parquet(io.BytesIO(contents))
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Parquet files.")
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Generate ID for data
        data_id = str(uuid.uuid4())
        
        # Store data
        data_info = {
            "id": data_id,
            "name": data_name,
            "filename": data_file.filename,
            "date_range": f"{data.index[0]} to {data.index[-1]}",
            "num_rows": len(data),
            "data": data
        }
        
        # Store in memory
        market_data[data_id] = data_info
        
        # Save to file
        info_to_save = {k: v for k, v in data_info.items() if k != 'data'}
        
        with open(os.path.join(DATA_DIR, f"{data_id}.json"), 'w') as f:
            json.dump(info_to_save, f)
        
        # Save data CSV
        data.to_csv(os.path.join(DATA_DIR, f"{data_id}.csv"))
        
        return {
            "status": "success",
            "message": f"Market data '{data_name}' uploaded successfully",
            "data_id": data_id,
            "summary": {
                "rows": len(data),
                "start_date": data.index[0].strftime('%Y-%m-%d'),
                "end_date": data.index[-1].strftime('%Y-%m-%d')
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# FastAPI route for creating sample data
@app.post("/api/data/sample", response_class=JSONResponse)
async def create_sample_market_data(days: int = 500, volatility: float = 0.01):
    try:
        # Generate sample data
        data = create_sample_data(days, volatility)
        
        # Generate ID for data
        data_id = str(uuid.uuid4())
        
        # Store data
        market_data[data_id] = {
            "id": data_id,
            "name": f"Sample Data ({days} days)",
            "filename": "sample_data.csv",
            "date_range": f"{data.index[0]} to {data.index[-1]}",
            "num_rows": len(data),
            "data": data
        }
        
        return {
            "status": "success",
            "message": f"Sample market data created successfully",
            "data_id": data_id,
            "summary": {
                "rows": len(data),
                "start_date": data.index[0].strftime('%Y-%m-%d'),
                "end_date": data.index[-1].strftime('%Y-%m-%d')
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# FastAPI route for getting all market data
@app.get("/api/data", response_class=JSONResponse)
async def get_all_data():
    # Return all market data without the data objects
    return [
        {k: v for k, v in data.items() if k != "data"}
        for data in market_data.values()
    ]

# FastAPI route for running a backtest
@app.post("/api/backtest", response_class=JSONResponse)
async def run_backtest(backtest_req: BacktestRequest):
    try:
        # Check if strategy exists
        if backtest_req.strategy_id not in strategies:
            print("=========555=======", strategies)
            print("==================", backtest_req.strategy_id)
            raise HTTPException(status_code=404, detail=f"Strategy with ID {backtest_req.strategy_id} not found")
        
        # Check if data exists
        if backtest_req.data_id not in market_data:
            raise HTTPException(status_code=404, detail=f"Market data with ID {backtest_req.data_id} not found")
        
        # Get strategy and data
        strategy = strategies[backtest_req.strategy_id]["strategy_obj"]
        data = market_data[backtest_req.data_id]["data"]
        
        # Create backtester config
        config = BacktestConfig(**backtest_req.config) if backtest_req.config else BacktestConfig()
        
        # Create backtester
        backtester = Backtester(config)
        
        # Run backtest
        result = backtester.backtest(strategy, data)
        
        # Generate ID for result
        result_id = str(uuid.uuid4())
        
        # Store result
        backtest_results[result_id] = {
            "id": result_id,
            "strategy_id": backtest_req.strategy_id,
            "data_id": backtest_req.data_id,
            "config": backtest_req.config,
            "result": result
        }
        
        # Generate plots
        equity_curve_fig = plot_equity_curve(result)
        positions_fig = plot_positions_over_time(result, data)
        
        # Convert figures to base64 for embedding in HTML
        equity_curve_img = figure_to_base64(equity_curve_fig)
        positions_img = figure_to_base64(positions_fig)
        
        # Extract metrics
        metrics = {
            "total_return": result.total_return,
            "annual_return": result.annual_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "volatility": result.volatility,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "num_trades": result.trades.shape[0] if result.trades is not None else 0
        }
        
        return {
            "status": "success",
            "message": "Backtest completed successfully",
            "result_id": result_id,
            "metrics": metrics,
            "plots": {
                "equity_curve": equity_curve_img,
                "positions": positions_img
            }
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

# FastAPI route for getting backtest result
@app.get("/api/backtest/{result_id}", response_class=JSONResponse)
async def get_backtest_result(result_id: str):
    if result_id not in backtest_results:
        raise HTTPException(status_code=404, detail=f"Backtest result with ID {result_id} not found")
    
    result_info = backtest_results[result_id].copy()
    
    # Get result object
    result = result_info["result"]
    
    # Extract metrics
    metrics = {
        "total_return": result.total_return,
        "annual_return": result.annual_return,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "volatility": result.volatility,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "num_trades": result.trades.shape[0] if result.trades is not None else 0
    }
    
    # Generate plots
    data = market_data[result_info["data_id"]]["data"]
    equity_curve_fig = plot_equity_curve(result)
    positions_fig = plot_positions_over_time(result, data)
    
    # Convert figures to base64 for embedding in HTML
    equity_curve_img = figure_to_base64(equity_curve_fig)
    positions_img = figure_to_base64(positions_fig)
    
    return {
        "strategy_id": result_info["strategy_id"],
        "strategy_name": strategies[result_info["strategy_id"]]["name"],
        "data_id": result_info["data_id"],
        "data_name": market_data[result_info["data_id"]]["name"],
        "config": result_info["config"],
        "metrics": metrics,
        "plots": {
            "equity_curve": equity_curve_img,
            "positions": positions_img
        }
    }

# FastAPI route for deploying a strategy to production
@app.post("/api/deploy", response_class=JSONResponse)
async def deploy_strategy(deploy_req: DeployRequest, background_tasks: BackgroundTasks):
    try:
        # Check if strategy exists
        if deploy_req.strategy_id not in strategies:
            raise HTTPException(status_code=404, detail=f"Strategy with ID {deploy_req.strategy_id} not found")
        
        # Check if data exists
        if deploy_req.data_id not in market_data:
            raise HTTPException(status_code=404, detail=f"Market data with ID {deploy_req.data_id} not found")
        
        # Get strategy and data
        strategy = strategies[deploy_req.strategy_id]["strategy_obj"]
        data = market_data[deploy_req.data_id]["data"]
        
        # Create or get production engine
        global production_engine
        if production_engine is None:
            production_engine = ProductionEngine()
            background_tasks.add_task(production_engine.start)
        
        # Create strategy config
        config = StrategyConfig(**deploy_req.config) if deploy_req.config else StrategyConfig(
            name=strategies[deploy_req.strategy_id]["name"],
            market_data_config={"symbols": [market_data[deploy_req.data_id]["name"]]}
        )
        
        # Add market data to production engine
        production_engine.add_market_data(market_data[deploy_req.data_id]["name"], data)
        
        # Deploy strategy
        deployed_id = production_engine.deploy_strategy(strategy, config)
        
        # Store deployed strategy info
        deployed_strategies[deployed_id] = {
            "id": deployed_id,
            "strategy_id": deploy_req.strategy_id,
            "data_id": deploy_req.data_id,
            "config": deploy_req.config
        }
        
        return {
            "status": "success",
            "message": f"Strategy '{strategies[deploy_req.strategy_id]['name']}' deployed successfully",
            "deployed_id": deployed_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# FastAPI route for getting deployed strategy status
@app.get("/api/deploy/{deployed_id}", response_class=JSONResponse)
async def get_deployed_status(deployed_id: str):
    global production_engine
    
    if production_engine is None:
        raise HTTPException(status_code=400, detail="Production engine not started")
    
    if deployed_id not in deployed_strategies:
        raise HTTPException(status_code=404, detail=f"Deployed strategy with ID {deployed_id} not found")
    
    # Get strategy status
    status = production_engine.get_strategy_status(deployed_id)
    
    if status is None:
        raise HTTPException(status_code=404, detail=f"Strategy with ID {deployed_id} not found in production engine")
    
    return {
        "deployed_id": deployed_id,
        "strategy_id": deployed_strategies[deployed_id]["strategy_id"],
        "strategy_name": strategies[deployed_strategies[deployed_id]["strategy_id"]]["name"],
        "data_id": deployed_strategies[deployed_id]["data_id"],
        "data_name": market_data[deployed_strategies[deployed_id]["data_id"]]["name"],
        "status": status
    }

# FastAPI route for pausing a deployed strategy
@app.post("/api/deploy/{deployed_id}/pause", response_class=JSONResponse)
async def pause_deployed_strategy(deployed_id: str):
    global production_engine
    
    if production_engine is None:
        raise HTTPException(status_code=400, detail="Production engine not started")
    
    if deployed_id not in deployed_strategies:
        raise HTTPException(status_code=404, detail=f"Deployed strategy with ID {deployed_id} not found")
    
    # Pause strategy
    success = production_engine.pause_strategy(deployed_id)
    
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to pause strategy with ID {deployed_id}")
    
    return {
        "status": "success",
        "message": f"Strategy paused successfully"
    }

# FastAPI route for resuming a deployed strategy
@app.post("/api/deploy/{deployed_id}/resume", response_class=JSONResponse)
async def resume_deployed_strategy(deployed_id: str):
    global production_engine
    
    if production_engine is None:
        raise HTTPException(status_code=400, detail="Production engine not started")
    
    if deployed_id not in deployed_strategies:
        raise HTTPException(status_code=404, detail=f"Deployed strategy with ID {deployed_id} not found")
    
    # Resume strategy
    success = production_engine.resume_strategy(deployed_id)
    
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to resume strategy with ID {deployed_id}")
    
    return {
        "status": "success",
        "message": f"Strategy resumed successfully"
    }

# FastAPI route for stopping a deployed strategy
@app.post("/api/deploy/{deployed_id}/stop", response_class=JSONResponse)
async def stop_deployed_strategy(deployed_id: str):
    global production_engine
    
    if production_engine is None:
        raise HTTPException(status_code=400, detail="Production engine not started")
    
    if deployed_id not in deployed_strategies:
        raise HTTPException(status_code=404, detail=f"Deployed strategy with ID {deployed_id} not found")
    
    # Stop strategy
    success = production_engine.stop_strategy(deployed_id)
    
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to stop strategy with ID {deployed_id}")
    
    return {
        "status": "success",
        "message": f"Strategy stopped successfully"
    }

# FastAPI route for getting all deployed strategies
@app.get("/api/deploy", response_class=JSONResponse)
async def get_all_deployed_strategies():
    global production_engine
    
    if production_engine is None:
        return []
    
    # Get all statuses
    statuses = production_engine.get_all_strategy_statuses()
    
    return [
        {
            "deployed_id": deployed_id,
            "strategy_id": deployed_info["strategy_id"],
            "strategy_name": strategies[deployed_info["strategy_id"]]["name"],
            "data_id": deployed_info["data_id"],
            "data_name": market_data[deployed_info["data_id"]]["name"],
            "status": statuses.get(deployed_id)
        }
        for deployed_id, deployed_info in deployed_strategies.items()
        if deployed_id in statuses
    ]

# FastAPI route for the strategies page
@app.get("/strategies", response_class=HTMLResponse)
async def strategies_page(request: Request):
    return templates.TemplateResponse("strategies.html", {"request": request})

# FastAPI route for the data page
@app.get("/data", response_class=HTMLResponse)
async def data_page(request: Request):
    return templates.TemplateResponse("data.html", {"request": request})

# FastAPI route for the backtest page
@app.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request):
    return templates.TemplateResponse("backtest.html", {"request": request})

# FastAPI route for the deployment page
@app.get("/deploy", response_class=HTMLResponse)
async def deploy_page(request: Request):
    return templates.TemplateResponse("deploy.html", {"request": request})

# Shutdown event handler
@app.on_event("shutdown")
def shutdown_event():
    global production_engine
    if production_engine is not None:
        production_engine.stop()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
