"""
Example strategy demonstration.

This script demonstrates how to use the Strategy Composition and Evaluation Framework
to create, backtest, and deploy trading strategies.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from strategy_dsl import (Strategy, create_indicator, create_signal,
                        create_allocation, create_risk_control,
                        create_execution, create_post_trade)
from indicators import sma, ema, macd, rsi, bollinger_bands
from backtester import Backtester, BacktestConfig, BacktestResult
from online_learning import StrategyBandit, OnlineLearner
from production_engine import ProductionEngine, StrategyConfig

# Create sample market data
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

# Define strategy components
def dual_ma_crossover_signal(data, context, fast_period=20, slow_period=50, price_col='close'):
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

# Create and backtest a strategy
def create_and_backtest_strategy(data):
    """Create and backtest a sample strategy."""
    # Create strategy
    strategy = Strategy("Dual MA Crossover with RSI Filter")
    
    # Add indicators
    strategy.add_indicator(create_indicator("fast_ma", sma, window=20))
    strategy.add_indicator(create_indicator("slow_ma", sma, window=50))
    strategy.add_indicator(create_indicator("rsi", rsi, window=14))
    strategy.add_indicator(create_indicator("macd", macd, fast_window=12, slow_window=26, signal_window=9))
    
    # Add signals
    strategy.add_signal(create_signal("ma_crossover", dual_ma_crossover_signal, fast_period=20, slow_period=50))
    strategy.add_signal(create_signal("rsi_filter", rsi_filter, threshold_low=30, threshold_high=70))
    strategy.add_signal(create_signal("macd_signal", macd_signal))
    
    # Add allocations
    strategy.add_allocation(create_allocation("position_sizer", volatility_position_sizer, window=20, target_volatility=0.01, max_position=1.0))
    
    # Add risk controls
    strategy.add_risk_control(create_risk_control("max_position", max_position_limit, max_position=1.0))
    strategy.add_risk_control(create_risk_control("stop_loss", stop_loss, stop_percent=0.05))
    
    # Add execution components
    strategy.add_execution(create_execution("simple_execution", simple_execution, slippage=0.001))
    
    # Add post-trade analysis
    strategy.add_post_trade(create_post_trade("trade_analyzer", trade_analyzer))
    
    # Create backtester with config
    config = BacktestConfig(
        starting_capital=100000.0,
        commission_rate=0.001,
        slippage_model="percentage",
        slippage_value=0.001,
        market_impact_model="none",
        position_size_limit=1.0,
        track_trade_history=True,
        track_positions_history=True
    )
    backtester = Backtester(config)
    
    # Run backtest
    result = backtester.backtest(strategy, data)
    
    return strategy, result

# Test online learning
def test_online_learning(strategy, data):
    """Test online learning with a strategy."""
    # Create online learner
    learner = OnlineLearner(strategy, learning_rate=0.01, window_size=20)
    
    # Split data into training periods
    period_length = len(data) // 10
    
    for i in range(10):
        # Get data for this period
        start_idx = i * period_length
        end_idx = min((i + 1) * period_length, len(data))
        period_data = data.iloc[start_idx:end_idx]
        
        # Run strategy on this period
        result = strategy.run(period_data)
        
        # Extract returns
        if 'returns' in result:
            returns = result['returns']
            
            # Update learner
            updates = learner.update(period_data, returns)
            
            print(f"Period {i+1}: Updated parameters: {updates}")
    
    # Get optimized strategy
    optimized_strategy = learner.get_optimized_strategy()
    
    return optimized_strategy

# Test strategy bandit
def test_strategy_bandit(data):
    """Test strategy bandit with multiple strategies."""
    # Create strategies
    strategies = {}
    
    # Strategy 1: Fast MA Crossover
    strategy1 = Strategy("Fast MA Crossover")
    strategy1.add_indicator(create_indicator("fast_ma", sma, window=10))
    strategy1.add_indicator(create_indicator("slow_ma", sma, window=30))
    strategy1.add_signal(create_signal("ma_crossover", dual_ma_crossover_signal, fast_period=10, slow_period=30))
    strategy1.add_allocation(create_allocation("position_sizer", position_sizer, max_position=1.0))
    strategy1.add_risk_control(create_risk_control("max_position", max_position_limit, max_position=1.0))
    strategy1.add_execution(create_execution("simple_execution", simple_execution, slippage=0.001))
    strategy1.add_post_trade(create_post_trade("trade_analyzer", trade_analyzer))
    
    strategies["fast_ma"] = strategy1
    
    # Strategy 2: Slow MA Crossover
    strategy2 = Strategy("Slow MA Crossover")
    strategy2.add_indicator(create_indicator("fast_ma", sma, window=50))
    strategy2.add_indicator(create_indicator("slow_ma", sma, window=200))
    strategy2.add_signal(create_signal("ma_crossover", dual_ma_crossover_signal, fast_period=50, slow_period=200))
    strategy2.add_allocation(create_allocation("position_sizer", position_sizer, max_position=1.0))
    strategy2.add_risk_control(create_risk_control("max_position", max_position_limit, max_position=1.0))
    strategy2.add_execution(create_execution("simple_execution", simple_execution, slippage=0.001))
    strategy2.add_post_trade(create_post_trade("trade_analyzer", trade_analyzer))
    
    strategies["slow_ma"] = strategy2
    
    # Strategy 3: RSI Mean Reversion
    strategy3 = Strategy("RSI Mean Reversion")
    strategy3.add_indicator(create_indicator("rsi", rsi, window=14))
    strategy3.add_signal(create_signal("rsi_filter", rsi_filter, threshold_low=30, threshold_high=70))
    strategy3.add_allocation(create_allocation("position_sizer", position_sizer, max_position=1.0))
    strategy3.add_risk_control(create_risk_control("max_position", max_position_limit, max_position=1.0))
    strategy3.add_execution(create_execution("simple_execution", simple_execution, slippage=0.001))
    strategy3.add_post_trade(create_post_trade("trade_analyzer", trade_analyzer))
    
    strategies["rsi"] = strategy3
    
    # Create strategy bandit
    bandit = StrategyBandit(strategies, bandit_algo='ucb', c=0.5)
    
    # Split data into periods
    period_length = 30
    num_periods = len(data) // period_length
    
    # Track selected strategies and returns
    selected_strategies = []
    period_returns = []
    
    for i in range(num_periods):
        # Get data for this period
        start_idx = i * period_length
        end_idx = min((i + 1) * period_length, len(data))
        period_data = data.iloc[start_idx:end_idx]
        
        # Select strategy
        strategy_name, strategy = bandit.select_strategy()
        selected_strategies.append(strategy_name)
        
        # Run strategy
        result = strategy.run(period_data)
        
        # Extract returns
        if 'returns' in result:
            returns = result['returns']
            avg_return = np.mean(returns)
            period_returns.append(avg_return)
            
            # Update bandit
            bandit.update(strategy_name, returns)
            
            print(f"Period {i+1}: Selected {strategy_name}, Return: {avg_return:.4f}")
        
    # Get bandit statistics
    stats = bandit.get_stats()
    
    print("\nBandit Statistics:")
    for arm, stat in stats.items():
        print(f"{arm}: {stat}")
    
    return bandit, selected_strategies, period_returns

# Test production engine
def test_production_engine(strategy, data):
    """Test production engine with a strategy."""
    # Create production engine
    engine = ProductionEngine()
    
    # Create strategy config
    config = StrategyConfig(
        name="Dual MA Crossover with RSI Filter",
        max_position=1.0,
        schedule={"interval_seconds": 60},
        market_data_config={"symbols": ["SAMPLE"]}
    )
    
    # Add market data
    engine.add_market_data("SAMPLE", data)
    
    # Start engine
    engine.start()
    
    # Deploy strategy
    strategy_id = engine.deploy_strategy(strategy, config)
    
    # Sleep for a bit to let strategy run
    print(f"Deployed strategy with ID: {strategy_id}")
    print("Press Enter to continue...")
    input()
    
    # Get strategy status
    status = engine.get_strategy_status(strategy_id)
    print(f"Strategy status: {status}")
    
    # Stop engine
    engine.stop()
    
    return engine, strategy_id

# Main function
def main():
    """Main function to demonstrate framework capabilities."""
    # Create sample data
    print("Creating sample data...")
    data = create_sample_data(days=500, volatility=0.01)
    
    # Create and backtest strategy
    print("\nCreating and backtesting strategy...")
    strategy, result = create_and_backtest_strategy(data)
    
    # Print backtest results
    print("\nBacktest Results:")
    print(result.summary())
    
    # Plot equity curve
    equity_fig = result.plot_equity_curve()
    plt.savefig('equity_curve.png')
    
    # Plot positions over time
    positions_fig = result.plot_positions_over_time()
    plt.savefig('positions.png')
    
    # Plot signals heat map
    signals_fig = result.plot_signals_heat_map()
    plt.savefig('signals.png')
    
    # Test online learning
    print("\nTesting online learning...")
    optimized_strategy = test_online_learning(strategy, data)
    
    # Backtest optimized strategy
    print("\nBacktesting optimized strategy...")
    backtester = Backtester()
    optimized_result = backtester.backtest(optimized_strategy, data)
    
    print("\nOptimized Strategy Results:")
    print(optimized_result.summary())
    
    # Test strategy bandit
    print("\nTesting strategy bandit...")
    bandit, selected_strategies, period_returns = test_strategy_bandit(data)
    
    # Test production engine
    print("\nTesting production engine...")
    engine, strategy_id = test_production_engine(strategy, data)
    
    print("\nDemonstration complete!")

if __name__ == "__main__":
    main()
