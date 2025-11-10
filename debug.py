"""
Debug Backtest Script

This script tests and debugs trading strategies by running them on sample data
and providing detailed output at each stage of the process.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import your existing modules - update these imports based on your project structure
try:
    from src.strategy_dsl import Strategy, create_indicator, create_signal, create_allocation, create_execution, create_risk_control, create_post_trade
except ImportError:
    print("Error: Could not import strategy_dsl module. Make sure the path is correct.")
    print(f"Current path: {sys.path}")
    sys.exit(1)

def generate_sample_data(days=252, volatility=0.015):
    """Generate sample market data for testing."""
    np.random.seed(42)  # For reproducibility
    
    # Generate dates
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=days)
    
    # Generate prices with random walk
    returns = np.random.normal(0.0002, volatility, days)
    price = 100.0
    prices = [price]
    
    for ret in returns:
        price *= (1 + ret)
        prices.append(price)
    
    prices = np.array(prices[1:])  # Remove initial price and convert to numpy array
    
    # Create dataframe
    df = pd.DataFrame({
        'date': dates,
        'open': prices * (1 - volatility/2),
        'high': prices * (1 + volatility),
        'low': prices * (1 - volatility),
        'close': prices,
        'volume': np.random.randint(1000, 100000, days)
    })
    
    print(f"Generated sample data with {len(df)} rows")
    print(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
    return df

def create_debug_rsi_strategy():
    """Create a debug version of the RSI strategy with extensive logging."""
    strategy = Strategy("Debug RSI")
    
    # Add RSI indicator with debug
    def calculate_rsi(data, context, window=14, price_col='close'):
        """Calculate RSI with debug information."""
        print(f"Calculating RSI with window={window}, price_col={price_col}")
        
        # Check data validity
        if price_col not in data.columns:
            print(f"Error: {price_col} column not found in data!")
            print(f"Available columns: {data.columns.tolist()}")
            return np.zeros(len(data))
        
        prices = data[price_col].values
        print(f"Price data shape: {prices.shape}")
        print(f"Price range: ${prices.min():.2f} to ${prices.max():.2f}")
        
        # Calculate RSI
        deltas = np.diff(prices)
        deltas = np.append(deltas, 0)  # Add a zero for the last position
        
        print(f"Deltas shape: {deltas.shape}")
        print(f"Delta range: {deltas.min():.4f} to {deltas.max():.4f}")
        
        # Seed up and down
        seed = deltas[:window+1]
        up = seed[seed >= 0].sum()/window if len(seed[seed >= 0]) > 0 else 0
        down = -seed[seed < 0].sum()/window if len(seed[seed < 0]) > 0 else 0.0001
        
        print(f"Initial up/down: {up:.4f}/{down:.4f}")
        
        # Calculate RS and RSI
        rs = up/down if down > 0 else 999
        rsi = np.zeros_like(prices)
        rsi[:window] = 100. - 100./(1. + rs)
        
        for i in range(window, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
                
            up = (up * (window - 1) + upval) / window
            down = (down * (window - 1) + downval) / window
            
            rs = up/down if down > 0 else 999
            rsi[i] = 100. - 100./(1. + rs)
        
        # Debug output
        print(f"RSI calculation complete. Values range: min={np.min(rsi):.2f}, max={np.max(rsi):.2f}")
        print(f"RSI distribution: <30: {np.sum(rsi < 30)}, 30-70: {np.sum((rsi >= 30) & (rsi <= 70))}, >70: {np.sum(rsi > 70)}")
        
        # Store in context for easier debugging
        if 'debug' not in context:
            context['debug'] = {}
        context['debug']['rsi_values'] = rsi
        
        return rsi
    
    strategy.add_indicator(create_indicator("rsi", calculate_rsi, window=14))
    
    # Add signal generator
    def rsi_signal_generator(data, context):
        """Generate trading signals based on RSI."""
        print("Generating RSI signals...")
        
        # Get RSI values from indicators
        indicators = context.get('indicators', {})
        rsi_values = indicators.get('rsi', None)
        
        if rsi_values is None:
            print("Error: RSI indicator not found in context!")
            return np.zeros(len(data))
        
        # Generate signals with more aggressive thresholds
        signals = np.zeros(len(data))
        
        print("Using thresholds: Oversold < 30, Overbought > 70")
        for i in range(len(data)):
            if rsi_values[i] < 30:  # Oversold condition
                signals[i] = 1.0
            elif rsi_values[i] > 70:  # Overbought condition
                signals[i] = -1.0
        
        # Print signal stats
        non_zero = np.count_nonzero(signals)
        print(f"Generated {non_zero} signals out of {len(signals)} data points")
        print(f"Buy signals: {np.sum(signals > 0)}, Sell signals: {np.sum(signals < 0)}")
        
        # Store for debugging
        if 'debug' not in context:
            context['debug'] = {}
        context['debug']['signals'] = signals
        
        return signals
    # RSI Signal generation with proper thresholds
    def rsi_signal_generator__(data, context):
        indicators = context.get('indicators', {})
        rsi_values = indicators.get('rsi', None)
        
        if rsi_values is None:
            return np.zeros(len(data))
        
        signals = np.zeros(len(data))
        
        # Use reasonable thresholds
        for i in range(len(data)):
            if rsi_values[i] < 30:  # Oversold - Buy signal
                signals[i] = 1.0
            elif rsi_values[i] > 70:  # Overbought - Sell signal
                signals[i] = -1.0
        
        return signals


    strategy.add_signal(create_signal("rsi_signal", rsi_signal_generator))
    
    # Add position sizing
    def position_sizer(signal, data, context, max_position=1.0):
        """Size positions based on signals."""
        print(f"Sizing positions with max_position={max_position}...")
        
        positions = signal * max_position
        
        # Debug
        non_zero = np.count_nonzero(positions)
        print(f"Allocated {non_zero} positions out of {len(positions)} data points")
        print(f"Position range: min={positions.min():.2f}, max={positions.max():.2f}")
        
        # Store for debugging
        if 'debug' not in context:
            context['debug'] = {}
        context['debug']['positions'] = positions
        
        return positions
    
    strategy.add_allocation(create_allocation("position_sizer", position_sizer, max_position=1.0))
    
    # Add risk control
    def simple_risk_control(positions, data, context, max_risk=0.1):
        """Apply simple risk controls to positions."""
        print(f"Applying risk control with max_risk={max_risk}...")
        
        controlled_positions = positions.copy()
        
        # Apply maximum position size limit
        controlled_positions = np.clip(controlled_positions, -max_risk, max_risk)
        
        # Debug
        print(f"After risk control, positions range: min={controlled_positions.min():.2f}, max={controlled_positions.max():.2f}")
        
        # Store for debugging
        if 'debug' not in context:
            context['debug'] = {}
        context['debug']['controlled_positions'] = controlled_positions
        
        return controlled_positions
    
    strategy.add_risk_control(create_risk_control("simple_risk_control", simple_risk_control, max_risk=1.0))
    
    # Add execution with debug
    def debug_execution(trades, data, context):
        """Debug execution of trades."""
        print("Executing trades...")
        
        if 'debug' not in context:
            context['debug'] = {}
        
        # If trades is difference of current and previous positions
        if 'controlled_positions' in context['debug']:
            positions = context['debug']['controlled_positions']
            # Simulate trades as position changes
            actual_trades = np.zeros_like(trades)
            actual_trades[1:] = positions[1:] - positions[:-1]
            
            print(f"Execution - Number of trades: {np.count_nonzero(actual_trades)}")
            print(f"Execution - Total volume: {np.sum(np.abs(actual_trades)):.4f}")
            
            # Store for debugging
            context['debug']['trades'] = actual_trades
            
            return actual_trades
        else:
            print("No controlled positions found, using original trades")
            print(f"Execution - Number of trades: {np.count_nonzero(trades)}")
            print(f"Execution - Total volume: {np.sum(np.abs(trades)):.4f}")
            
            # Store for debugging
            context['debug']['trades'] = trades
            
            return trades
    
    strategy.add_execution(create_execution("debug_execution", debug_execution))
    
    # Add performance tracker
    def performance_tracker(trades, data, context, price_col='close'):
        """Track performance metrics for a strategy."""
        print("Tracking performance...")
        
        if 'debug' not in context:
            context['debug'] = {}
            
        prices = data[price_col].values
        
        # Initialize arrays
        returns = np.zeros(len(trades))
        equity = np.ones(len(trades)) * 100000.0  # Starting with $100,000
        positions = np.zeros(len(trades))
        
        # Calculate position and returns
        for i in range(1, len(trades)):
            # Update position with trades
            positions[i] = positions[i-1] + trades[i]
            
            # Calculate returns based on position
            price_return = prices[i] / prices[i-1] - 1.0
            trade_return = positions[i-1] * price_return
            
            # Store results
            returns[i] = trade_return
            equity[i] = equity[i-1] * (1 + trade_return)
        
        # Calculate performance metrics
        total_return = equity[-1] / equity[0] - 1.0
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        max_drawdown = 0.0
        peak = equity[0]
        
        for val in equity:
            if val > peak:
                peak = val
            drawdown = (peak - val) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Count trades
        trade_count = 0
        position = 0
        for t in trades:
            if t != 0:
                if (position == 0 and t != 0) or (position != 0 and position + t == 0):
                    trade_count += 1
                position += t
        
        # Store in context
        context['debug'].update({
            'returns': returns,
            'equity': equity,
            'positions': positions,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'trade_count': trade_count
        })
        
        print(f"Performance: Total Return = {total_return:.4f}, Ann. Return = {annualized_return:.4f}")
        print(f"Max Drawdown: {max_drawdown:.4f}, Total Trades: {trade_count}")
        
        return returns
    
    strategy.add_post_trade(create_post_trade("performance_tracker", performance_tracker))
    
    return strategy

def create_simple_ma_strategy():
    """Create a simple moving average crossover strategy."""
    strategy = Strategy("Simple MA Crossover")
    
    # Add indicators
    def calculate_sma(data, context, window=20, price_col='close'):
        """Calculate Simple Moving Average with debug info."""
        print(f"Calculating SMA with window={window}, price_col={price_col}")
        
        if price_col not in data.columns:
            print(f"Error: {price_col} column not found in data!")
            return np.zeros(len(data))
        
        values = data[price_col].rolling(window=window).mean().values
        
        # Replace NaN with zeros
        values = np.nan_to_num(values)
        
        print(f"SMA calculation complete. Values range: min={np.nanmin(values):.2f}, max={np.nanmax(values):.2f}")
        
        return values
    
    strategy.add_indicator(create_indicator("fast_ma", calculate_sma, window=10, price_col='close'))
    strategy.add_indicator(create_indicator("slow_ma", calculate_sma, window=30, price_col='close'))
    
    # Add signal generator
    def ma_crossover_signal(data, context):
        """Generate trading signals based on MA crossovers."""
        print("Generating MA crossover signals...")
        
        # Get indicators
        indicators = context.get('indicators', {})
        fast_ma = indicators.get('fast_ma')
        slow_ma = indicators.get('slow_ma')
        
        if fast_ma is None or slow_ma is None:
            print("Error: Moving averages not found in context!")
            return np.zeros(len(data))
        
        # Generate signals
        signals = np.zeros(len(data))
        
        for i in range(1, len(data)):
            # Buy when fast MA crosses above slow MA
            if fast_ma[i-1] <= slow_ma[i-1] and fast_ma[i] > slow_ma[i]:
                signals[i] = 1.0
            # Sell when fast MA crosses below slow MA
            elif fast_ma[i-1] >= slow_ma[i-1] and fast_ma[i] < slow_ma[i]:
                signals[i] = -1.0
        
        # Print signal stats
        non_zero = np.count_nonzero(signals)
        print(f"Generated {non_zero} MA crossover signals out of {len(signals)} data points")
        print(f"Buy signals: {np.sum(signals > 0)}, Sell signals: {np.sum(signals < 0)}")
        
        # Store for debugging
        if 'debug' not in context:
            context['debug'] = {}
        context['debug']['signals'] = signals
        context['debug']['fast_ma'] = fast_ma
        context['debug']['slow_ma'] = slow_ma
        
        return signals
    
    strategy.add_signal(create_signal("ma_signal", ma_crossover_signal))
    
    # Add position sizing
    def position_sizer(signal, data, context, max_position=1.0):
        """Size positions based on signals."""
        print(f"Sizing positions with max_position={max_position}...")
        
        positions = signal * max_position
        
        # Debug
        non_zero = np.count_nonzero(positions)
        print(f"Allocated {non_zero} positions out of {len(positions)} data points")
        
        # Store for debugging
        if 'debug' not in context:
            context['debug'] = {}
        context['debug']['positions'] = positions
        
        return positions
    
    strategy.add_allocation(create_allocation("position_sizer", position_sizer, max_position=1.0))
    
    # Add execution
    def simple_execution(trades, data, context):
        """Execute trades with simple execution model."""
        print("Executing trades...")
        
        # Store for debugging
        if 'debug' not in context:
            context['debug'] = {}
        
        # Simulate trades
        actual_trades = trades.copy()
        
        print(f"Execution - Number of trades: {np.count_nonzero(actual_trades)}")
        print(f"Execution - Total volume: {np.sum(np.abs(actual_trades)):.4f}")
        
        context['debug']['trades'] = actual_trades
        return actual_trades
    
    strategy.add_execution(create_execution("simple_execution", simple_execution))
    
    # Add performance tracker (same as RSI strategy)
    def performance_tracker(trades, data, context, price_col='close'):
        """Track performance metrics for a strategy."""
        print("Tracking performance...")
        
        if 'debug' not in context:
            context['debug'] = {}
            
        prices = data[price_col].values
        
        # Initialize arrays
        returns = np.zeros(len(trades))
        equity = np.ones(len(trades)) * 100000.0  # Starting with $100,000
        positions = np.zeros(len(trades))
        
        # Calculate position and returns
        for i in range(1, len(trades)):
            # Update position with trades
            positions[i] = positions[i-1] + trades[i]
            
            # Calculate returns based on position
            price_return = prices[i] / prices[i-1] - 1.0
            trade_return = positions[i-1] * price_return
            
            # Store results
            returns[i] = trade_return
            equity[i] = equity[i-1] * (1 + trade_return)
        
        # Calculate performance metrics
        total_return = equity[-1] / equity[0] - 1.0
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        max_drawdown = 0.0
        peak = equity[0]
        
        for val in equity:
            if val > peak:
                peak = val
            drawdown = (peak - val) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Count trades
        trade_count = 0
        position = 0
        for t in trades:
            if t != 0:
                if (position == 0 and t != 0) or (position != 0 and position + t == 0):
                    trade_count += 1
                position += t
        
        # Store in context
        context['debug'].update({
            'returns': returns,
            'equity': equity,
            'positions': positions,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'trade_count': trade_count
        })
        
        print(f"Performance: Total Return = {total_return:.4f}, Ann. Return = {annualized_return:.4f}")
        print(f"Max Drawdown: {max_drawdown:.4f}, Total Trades: {trade_count}")
        
        return returns
    
    strategy.add_post_trade(create_post_trade("performance_tracker", performance_tracker))
    
    return strategy

def plot_strategy_results(context, strategy_name, save_path=None):
    """Plot strategy results from context."""
    print(f"Plotting results for {strategy_name}...")
    
    if 'debug' not in context:
        print("No debug data found in context!")
        return
    
    debug = context['debug']
    
    if 'equity' not in debug:
        print("No equity curve found in debug data!")
        return
    
    equity = debug['equity']
    positions = debug.get('positions', np.zeros_like(equity))
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot equity curve
    axes[0].plot(equity, 'b-', linewidth=2)
    axes[0].set_title(f"{strategy_name} - Equity Curve")
    axes[0].set_ylabel('Equity ($)')
    axes[0].grid(True)
    
    # Plot positions
    axes[1].plot(positions, 'g-', linewidth=1)
    axes[1].set_title('Positions Over Time')
    axes[1].set_ylabel('Position Size')
    axes[1].set_xlabel('Time')
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()

def save_backtest_results(context, strategy_name, save_path=None):
    """Save backtest results to a JSON file."""
    if 'debug' not in context:
        print("No debug data found in context!")
        return
    
    debug = context['debug']
    
    results = {
        'strategy_name': strategy_name,
        'total_return': float(debug.get('total_return', 0.0)),
        'annualized_return': float(debug.get('annualized_return', 0.0)),
        'max_drawdown': float(debug.get('max_drawdown', 0.0)),
        'trade_count': int(debug.get('trade_count', 0)),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Convert numpy arrays to lists for JSON serialization
    if 'equity' in debug:
        results['equity'] = debug['equity'].tolist()
    
    if 'positions' in debug:
        results['positions'] = debug['positions'].tolist()
    
    if 'trades' in debug:
        results['trades'] = debug['trades'].tolist()
    
    # Save to file
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {save_path}")
    
    return results

def run_backtest(strategy, data, save_dir=None):
    """Run a backtest for a given strategy on provided data."""
    print(f"Running backtest for strategy: {strategy.name}")
    print(f"Data shape: {data.shape}")
    
    # Create context
    context = {}
    
    # Run strategy
    try:
        result_context = strategy.run(data, context)
        print("Strategy run completed successfully")
    except Exception as e:
        print(f"Error running strategy: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Plot results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, f"{strategy.name}_results.png")
        plot_strategy_results(result_context, strategy.name, save_path=plot_path)
        
        # Save results
        results_path = os.path.join(save_dir, f"{strategy.name}_results.json")
        save_backtest_results(result_context, strategy.name, save_path=results_path)
    else:
        plot_strategy_results(result_context, strategy.name)
    
    return result_context

def main():
    """Main function to run backtest debugging."""
    print("Starting backtest debugging script...")
    
    # Generate sample data
    data = generate_sample_data(days=252, volatility=0.015)
    
    # Create strategies
    rsi_strategy = create_debug_rsi_strategy()
    ma_strategy = create_simple_ma_strategy()
    
    # Set save directory
    save_dir = "backtest_results"
    
    # Run backtests
    print("\n" + "="*50)
    print("Running RSI Strategy Backtest")
    print("="*50)
    rsi_context = run_backtest(rsi_strategy, data, save_dir)
    
    print("\n" + "="*50)
    print("Running Moving Average Strategy Backtest")
    print("="*50)
    ma_context = run_backtest(ma_strategy, data, save_dir)
    
    print("\n" + "="*50)
    print("Backtest Results Summary")
    print("="*50)
    
    if rsi_context and 'debug' in rsi_context:
        rsi_debug = rsi_context['debug']
        print(f"RSI Strategy:")
        print(f"  Total Return: {rsi_debug.get('total_return', 0.0):.4f}")
        print(f"  Annualized Return: {rsi_debug.get('annualized_return', 0.0):.4f}")
        print(f"  Max Drawdown: {rsi_debug.get('max_drawdown', 0.0):.4f}")
        print(f"  Trade Count: {rsi_debug.get('trade_count', 0)}")
    else:
        print("RSI Strategy: No results available")
    
    if ma_context and 'debug' in ma_context:
        ma_debug = ma_context['debug']
        print(f"MA Strategy:")
        print(f"  Total Return: {ma_debug.get('total_return', 0.0):.4f}")
        print(f"  Annualized Return: {ma_debug.get('annualized_return', 0.0):.4f}")
        print(f"  Max Drawdown: {ma_debug.get('max_drawdown', 0.0):.4f}")
        print(f"  Trade Count: {ma_debug.get('trade_count', 0)}")
    else:
        print("MA Strategy: No results available")
    
    print("\nBacktest debugging complete!")
    print(f"Results saved in '{save_dir}' directory")

if __name__ == "__main__":
    main()