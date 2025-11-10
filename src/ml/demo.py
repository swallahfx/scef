"""
Demonstration of ML-based trading strategies in SCEF.
This script showcases how to use machine learning components with the SCEF framework.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from typing import Dict, Any, List, Optional

# Add parent directory to path to import SCEF modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import SCEF components
from src.backtester import Backtester, BacktestConfig, BacktestResult
from src.strategy_dsl import Strategy
from src.indicators import sma, ema, rsi

# Import ML components
from src.ml.features import prepare_ml_features, normalize_features
from src.ml.models import MLModel, ml_signal_generator
from src.ml.reinforcement import rl_signal_generator
from src.ml.ensemble import ensemble_signal_generator, StrategyEnsemble
from src.ml.integration import (
    create_ml_strategy, create_rl_strategy, 
    create_ensemble_strategy, register_ml_components
)


def load_market_data(filepath: str) -> pd.DataFrame:
    """
    Load market data from CSV file.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        DataFrame with market data
    """
    try:
        # Read CSV file
        df = pd.read_csv(filepath)
        
        # Check if date column exists
        if 'date' in df.columns:
            date_col = 'date'
        elif 'Date' in df.columns:
            date_col = 'Date'
        else:
            # Try to use the first column as date
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            return df
        
        # Convert date to datetime and set as index
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        
        # Ensure required columns exist (convert if necessary)
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        
        # Rename columns if needed
        df.rename(columns=column_mapping, inplace=True)
        
        # Check for missing required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing required columns: {missing_cols}")
            
            # If 'close' is missing but 'adj_close' exists, use it
            if 'close' in missing_cols and 'adj_close' in df.columns:
                df['close'] = df['adj_close']
                missing_cols.remove('close')
            
            # If volume is missing, add dummy volume
            if 'volume' in missing_cols:
                df['volume'] = 0
                missing_cols.remove('volume')
            
            # For other missing columns, use close price
            for col in missing_cols:
                if 'close' in df.columns:
                    df[col] = df['close']
        
        return df
    
    except Exception as e:
        print(f"Error loading market data: {str(e)}")
        return None


def create_traditional_strategy(name: str) -> Strategy:
    """
    Create a traditional (non-ML) trading strategy for comparison.
    
    Args:
        name: Strategy name
    
    Returns:
        Strategy object
    """
    from src.strategy_dsl import (
        Strategy, create_indicator, create_signal, 
        create_allocation, create_risk_control, 
        create_execution, create_post_trade
    )
    
    # Create strategy
    strategy = Strategy(name)
    
    # Add indicators
    strategy.add_indicator(create_indicator("sma20", sma, window=20))
    strategy.add_indicator(create_indicator("sma50", sma, window=50))
    strategy.add_indicator(create_indicator("rsi14", rsi, window=14))
    
    # Add signal generator
    def ma_crossover_signal(data, context):
        # Get indicators from context
        if 'indicators' not in context:
            print("Warning: Indicators not found in context")
            return np.zeros(len(data))
        
        indicators = context['indicators']
        if 'sma20' not in indicators or 'sma50' not in indicators:
            print("Warning: Required indicators (sma20, sma50) not found in context")
            return np.zeros(len(data))
        
        # Get indicator values
        sma20 = indicators['sma20']
        sma50 = indicators['sma50']
        rsi14 = indicators['rsi14'] if 'rsi14' in indicators else None
        
        # Generate signals
        signals = np.zeros(len(data))
        
        for i in range(1, len(data)):
            # MA crossover: Buy when SMA20 crosses above SMA50, sell when crosses below
            if sma20[i-1] <= sma50[i-1] and sma20[i] > sma50[i]:
                signals[i] = 1.0  # Buy
            elif sma20[i-1] >= sma50[i-1] and sma20[i] < sma50[i]:
                signals[i] = -1.0  # Sell
            
            # Filter with RSI if available
            if rsi14 is not None:
                # Avoid buying when RSI is overbought
                if signals[i] > 0 and rsi14[i] > 70:
                    signals[i] = 0.0
                
                # Avoid selling when RSI is oversold
                if signals[i] < 0 and rsi14[i] < 30:
                    signals[i] = 0.0
        
        return signals
    
    strategy.add_signal(create_signal("ma_crossover", ma_crossover_signal))
    
    # Add allocation
    def position_sizer(signal, data, context, max_position=1.0):
        return signal * max_position
    
    strategy.add_allocation(create_allocation("position_sizer", position_sizer))
    
    # Add risk control
    def max_position_limit(position, data, context, max_position=1.0):
        return np.clip(position, -max_position, max_position)
    
    strategy.add_risk_control(create_risk_control("max_position", max_position_limit))
    
    # Add execution
    def slippage_model(trades, data, context, slippage=0.001):
        slipped_trades = trades.copy()
        for i in range(len(trades)):
            if trades[i] > 0:
                slipped_trades[i] = trades[i] * (1 + slippage)
            elif trades[i] < 0:
                slipped_trades[i] = trades[i] * (1 - slippage)
        
        return slipped_trades
    
    strategy.add_execution(create_execution("slippage_model", slippage_model))
    
    # Add post-trade analysis
    def performance_tracker(trades, data, context, price_col='close'):
        # Calculate returns
        prices = data[price_col].values
        returns = np.zeros(len(trades))
        
        position = 0.0
        for i in range(1, len(trades)):
            # Calculate returns based on position
            if i > 0:
                price_return = prices[i] / prices[i-1] - 1.0
                returns[i] = position * price_return
            
            # Update position
            position += trades[i]
        
        return returns
    
    strategy.add_post_trade(create_post_trade("performance_tracker", performance_tracker))
    
    return strategy


def plot_backtest_results(results: List[BacktestResult], strategy_names: List[str]):
    """
    Plot results for multiple backtest results.
    
    Args:
        results: List of BacktestResult objects
        strategy_names: List of strategy names
    """
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Plot equity curves
    plt.subplot(2, 1, 1)
    for i, result in enumerate(results):
        plt.plot(result.equity_curve, label=strategy_names[i])
    plt.title('Equity Curves')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    
    # Plot drawdowns
    plt.subplot(2, 1, 2)
    for i, result in enumerate(results):
        drawdowns = 1.0 - result.equity_curve / np.maximum.accumulate(result.equity_curve)
        plt.plot(drawdowns, label=strategy_names[i])
    plt.title('Drawdowns')
    plt.xlabel('Time')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('backtest_results.png')
    plt.close()


def print_performance_metrics(results: List[BacktestResult], strategy_names: List[str]):
    """
    Print performance metrics for multiple strategies.
    
    Args:
        results: List of BacktestResult objects
        strategy_names: List of strategy names
    """
    print("\n" + "="*50)
    print("Performance Metrics")
    print("="*50)
    
    # Define metrics to display
    metrics = [
        ('Total Return', lambda r: r.total_return, '{:.2%}'),
        ('Annual Return', lambda r: r.annual_return, '{:.2%}'),
        ('Sharpe Ratio', lambda r: r.sharpe_ratio, '{:.2f}'),
        ('Max Drawdown', lambda r: r.max_drawdown, '{:.2%}'),
        ('Win Rate', lambda r: r.win_rate, '{:.2%}'),
        ('Profit Factor', lambda r: r.profit_factor, '{:.2f}'),
        ('Number of Trades', lambda r: len(r.trades) if r.trades is not None else 0, '{:.0f}')
    ]
    
    # Print header
    header = "{:<15}".format("Metric")
    for name in strategy_names:
        header += "{:<15}".format(name)
    print(header)
    print("-" * (15 + 15 * len(strategy_names)))
    
    # Print metrics
    for metric_name, metric_func, format_str in metrics:
        row = "{:<15}".format(metric_name)
        for result in results:
            try:
                value = metric_func(result)
                row += format_str.format(value).ljust(15)
            except Exception as e:
                row += "N/A".ljust(15)
        print(row)


def main():
    """Main function to demonstrate ML-based trading strategies."""
    print("="*50)
    print("SCEF Machine Learning Strategy Demo")
    print("="*50)
    
    # Register ML components
    register_ml_components()
    
    # Load market data
    data_dir = "market_data"
    data_file = "EURUSD_daily.csv"
    
    print(f"\nLoading market data from {os.path.join(data_dir, data_file)}...")
    
    data = load_market_data(os.path.join(data_dir, data_file))
    if data is None:
        print("Error: Failed to load market data")
        return
    
    print(f"Loaded {len(data)} rows of data from {data.index[0]} to {data.index[-1]}")
    
    # Create strategies
    print("\nCreating strategies...")
    
    # Traditional strategy
    traditional_strategy = create_traditional_strategy("Traditional MA Crossover")
    
    # ML-based strategy
    ml_strategy = create_ml_strategy(
        name="ML Random Forest",
        model_type="classifier",
        model_name="random_forest",
        model_params={"n_estimators": 100, "random_state": 42},
        prediction_horizon=1,
        position_sizing=1.0
    )
    
    # Reinforcement learning strategy
    rl_strategy = create_rl_strategy(
        name="RL Strategy",
        window_size=20,
        action_space=3,
        position_sizing=1.0
    )
    
    # Ensemble strategy
    ensemble_strategy = create_ensemble_strategy(
        name="Ensemble Strategy",
        strategies=[traditional_strategy, ml_strategy],
        weights=[0.5, 0.5],
        ensemble_method="weighted_average"
    )
    
    # Define backtest config
    config = BacktestConfig(
        starting_capital=100000,
        commission_rate=0.0001,
        slippage_model="percentage",
        slippage_value=0.0001,
        track_trade_history=True,
        track_positions_history=True,
        track_capital_history=True
    )
    
    # Create backtester
    backtester = Backtester(config)
    
    # Split data into training and testing
    split_date = data.index[int(len(data) * 0.7)]
    print(f"\nSplitting data at {split_date}")
    
    training_data = data[:split_date]
    testing_data = data[split_date:]
    
    # Train ML strategy
    print("\nTraining ML strategy...")
    
    # Extract features and prepare data
    features = prepare_ml_features(training_data, {})
    normalized_features = normalize_features(features, {})
    
    # Initialize ML model
    model = MLModel(
        model_type="classifier",
        model_name="random_forest",
        model_params={"n_estimators": 100, "random_state": 42},
        prediction_horizon=1
    )
    
    # Train model
    model.fit(normalized_features, training_data)
    
    # Store model in ML strategy context
    ml_context = {"ml_model": model}
    
    # Train RL strategy (simplified - would need more epochs in practice)
    print("\nTraining RL strategy...")
    # This is a placeholder - in a real system, we would train an RL agent here
    rl_context = {}
    
    # Run backtests
    print("\nRunning backtests...")
    
    results = []
    strategy_names = []
    
    # Test traditional strategy
    print("Testing Traditional MA Crossover strategy...")
    traditional_result = backtester.backtest(traditional_strategy, testing_data)
    results.append(traditional_result)
    strategy_names.append("Traditional")
    
    # Test ML strategy
    print("Testing ML Random Forest strategy...")
    ml_result = backtester.backtest(ml_strategy, testing_data, ml_context)
    results.append(ml_result)
    strategy_names.append("ML")
    
    # Test RL strategy
    print("Testing RL strategy...")
    rl_result = backtester.backtest(rl_strategy, testing_data, rl_context)
    results.append(rl_result)
    strategy_names.append("RL")
    
    # Test ensemble strategy
    print("Testing Ensemble strategy...")
    ensemble_result = backtester.backtest(ensemble_strategy, testing_data)
    results.append(ensemble_result)
    strategy_names.append("Ensemble")
    
    # Print results
    print_performance_metrics(results, strategy_names)
    
    # Plot results
    print("\nPlotting results...")
    plot_backtest_results(results, strategy_names)
    
    print("\nDemo completed successfully!")
    print("Results have been saved to backtest_results.png")


if __name__ == "__main__":
    main()
