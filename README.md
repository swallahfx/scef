# Strategy Composition and Evaluation Framework (SCEF)

SCEF is a comprehensive framework for developing, testing, and deploying algorithmic trading strategies. Built with functional programming principles, it enables rapid composition and evaluation of strategies across the entire lifecycle from research to production.

## Key Features

- **Functional DSL for Strategy Composition**: Create strategies by composing modular, reusable components
- **Optimized Technical Indicators**: High-performance implementations of common technical indicators using Numba
- **Comprehensive Backtesting Engine**: Simulate strategies on historical data with realistic trading conditions
- **Online Learning System**: Adapt strategies in real-time using bandit algorithms and online optimization
- **Production Deployment System**: Deploy strategies to production with monitoring, state management, and fault tolerance

## System Architecture

![Architecture Diagram](computer:///mnt/user-data/outputs/architecture.mermaid)

## Core Components

### Strategy DSL

The Strategy Domain-Specific Language allows traders and researchers to compose trading strategies from modular components:

- **Indicators**: Technical calculations like moving averages, RSI, MACD
- **Signals**: Logic that generates trading signals based on indicators
- **Allocations**: Position sizing based on signals and risk parameters
- **Executions**: Trade execution models with slippage and market impact
- **Risk Controls**: Risk management rules like stop-losses and position limits
- **Post-Trade Analysis**: Performance measurement and trade analysis

### Backtesting Engine

The backtesting engine provides:

- Realistic simulation of trading with commissions, slippage, and market impact
- Comprehensive performance metrics (Sharpe ratio, drawdown, win rate)
- Visualization tools for equity curves, positions, and signals
- Trade-level analysis for strategy improvement

### Online Learning

The online learning system enables adaptive strategies through:

- Multi-armed bandit algorithms for strategy selection (UCB, Thompson Sampling)
- Parameter optimization using gradient-based methods
- Automatic adaptation to changing market conditions

### Production System

The production system features:

- Thread-safe management of multiple strategies
- State persistence and recovery
- Real-time monitoring of performance metrics
- Version control for strategy updates
- Configurable execution schedules

## Example Usage

Here's a simple example of creating a dual moving average crossover strategy:

```python
from strategy_dsl import Strategy, create_indicator, create_signal, create_allocation
from indicators import sma

# Create strategy
strategy = Strategy("Dual MA Crossover")

# Add indicators
strategy.add_indicator(create_indicator("fast_ma", sma, window=20))
strategy.add_indicator(create_indicator("slow_ma", sma, window=50))

# Add signal
strategy.add_signal(create_signal("ma_crossover", dual_ma_crossover_signal))

# Add allocation
strategy.add_allocation(create_allocation("position_sizer", position_sizer))

# Run backtest
backtester = Backtester()
result = backtester.backtest(strategy, market_data)

# Analyze results
print(result.summary())
result.plot_equity_curve()
```

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the demo: `python src/demo.py`

## Requirements

- Python 3.8+
- NumPy
- pandas
- matplotlib
- seaborn
- numba

## Development Notes

This framework was designed to meet the requirements of the Two Lambda Engineer role at Two Lions, focusing on:

- Functional programming principles with composable components
- High-performance numerical implementations
- Comprehensive testing and simulation environments
- Production-ready deployment capabilities
- Machine learning integration

For real-world usage, consider adding:

- More comprehensive market data handling
- Additional technical indicators and signal generators
- More sophisticated execution models
- Enhanced risk management and portfolio allocation
- Integration with exchange APIs for live trading

## License

MIT
