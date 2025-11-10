"""
Backtesting engine for simulating trading strategies on historical data.

This module provides a framework for running strategies against historical
market data and calculating performance metrics.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from datetime import datetime

from src.strategy_dsl import Strategy


@dataclass
class BacktestConfig:
    """Configuration for a backtest."""
    starting_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_model: str = "fixed"  # "fixed", "percentage", or "none"
    slippage_value: float = 0.0005  # 0.05% for percentage or $0.0005 for fixed
    market_impact_model: str = "none"  # "linear", "square_root", or "none"
    market_impact_factor: float = 0.1
    position_size_limit: float = 1.0  # Maximum position size as fraction of capital
    lot_size: int = 1  # Minimum trade size
    track_trade_history: bool = True
    track_positions_history: bool = True
    track_capital_history: bool = True
    track_drawdowns: bool = True


class BacktestResult:
    """Results from a backtest run."""
    
    def __init__(self, config: BacktestConfig, data: pd.DataFrame):
        self.config = config
        self.data = data
        self.start_time = data.index[0] if isinstance(data.index, pd.DatetimeIndex) else None
        self.end_time = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else None
        
        # Performance metrics
        self.total_return: float = 0.0
        self.annual_return: float = 0.0
        self.sharpe_ratio: float = 0.0
        self.max_drawdown: float = 0.0
        self.volatility: float = 0.0
        self.win_rate: float = 0.0
        self.profit_factor: float = 0.0
        
        # History
        self.positions: Optional[pd.DataFrame] = None
        self.trades: Optional[pd.DataFrame] = None
        self.equity_curve: Optional[pd.Series] = None
        self.drawdown_curve: Optional[pd.Series] = None
        
        # Strategy results
        self.signals: Optional[np.ndarray] = None
        self.context: Dict[str, Any] = {}
    
    def calculate_metrics(self):
        """Calculate performance metrics from equity curve."""
        if self.equity_curve is None or len(self.equity_curve) == 0:
            return
            
        # Calculate returns
        daily_returns = self.equity_curve.pct_change().dropna()
        
        # Total return
        self.total_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1
        
        # Annual return
        if isinstance(self.equity_curve.index, pd.DatetimeIndex) and len(self.equity_curve) > 1:
            days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
            years = days / 365.25
            self.annual_return = (1 + self.total_return) ** (1 / max(years, 0.01)) - 1
        
        # Volatility (annualized)
        self.volatility = daily_returns.std() * (252 ** 0.5)
        
        # Sharpe ratio (annualized, assuming 0 risk-free rate)
        if self.volatility > 0:
            self.sharpe_ratio = self.annual_return / self.volatility
        
        # Maximum drawdown
        if self.drawdown_curve is not None:
            self.max_drawdown = self.drawdown_curve.min()
        
        # Trade statistics
        if self.trades is not None and len(self.trades) > 0:
            winning_trades = self.trades[self.trades['pnl'] > 0]
            losing_trades = self.trades[self.trades['pnl'] < 0]
            
            # Win rate
            if len(self.trades) > 0:
                self.win_rate = len(winning_trades) / len(self.trades)
            
            # Profit factor
            total_gains = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
            total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
            
            if total_losses > 0:
                self.profit_factor = total_gains / total_losses
            elif total_gains > 0:
                self.profit_factor = float('inf')
    
    def plot_equity_curve(self, figsize=(12, 6)):
        """Plot equity curve."""
        if self.equity_curve is None:
            return None
            
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.equity_curve, label='Equity Curve')
        
        # Add drawdown overlay if available
        if self.drawdown_curve is not None:
            ax_twin = ax.twinx()
            ax_twin.fill_between(self.drawdown_curve.index, 0, self.drawdown_curve, 
                                 alpha=0.3, color='red', label='Drawdown')
            ax_twin.set_ylabel('Drawdown')
            ax_twin.set_ylim(self.drawdown_curve.min() * 1.5, 0)
            
        ax.set_title('Equity Curve')
        ax.set_xlabel('Time')
        ax.set_ylabel('Equity')
        ax.grid(True)
        fig.tight_layout()
        
        return fig
    
    def plot_positions_over_time(self, figsize=(12, 6)):
        """Plot positions over time."""
        if self.positions is None or 'position' not in self.positions.columns:
            return None
            
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.positions.index, self.positions['position'], label='Position')
        
        # Add price overlay
        if 'close' in self.data.columns:
            ax_twin = ax.twinx()
            ax_twin.plot(self.data.index, self.data['close'], 'r-', alpha=0.3, label='Price')
            ax_twin.set_ylabel('Price')
            
        ax.set_title('Positions Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Position Size')
        ax.grid(True)
        fig.tight_layout()
        
        return fig
    
    def plot_signals_heat_map(self, figsize=(12, 6)):
        """Plot signal strength as a heat map over time."""
        if self.signals is None:
            return None
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert signals to DataFrame if needed
        if isinstance(self.signals, np.ndarray):
            signals_df = pd.Series(self.signals, index=self.data.index)
        else:
            signals_df = self.signals
            
        # Plot as a colormap
        cmap = plt.cm.RdYlGn  # Red for negative, yellow for neutral, green for positive
        ax.pcolormesh(np.array([signals_df.values]), cmap=cmap, vmin=-1, vmax=1)
        
        # Add price overlay
        ax_twin = ax.twinx()
        if 'close' in self.data.columns:
            ax_twin.plot(range(len(self.data.index)), self.data['close'], 'b-', alpha=0.7, label='Price')
        
        # Set x-axis ticks to show dates
        if isinstance(signals_df.index, pd.DatetimeIndex):
            xtick_indices = np.linspace(0, len(signals_df) - 1, min(10, len(signals_df))).astype(int)
            xtick_labels = [signals_df.index[i].strftime('%Y-%m-%d') for i in xtick_indices]
            ax.set_xticks(xtick_indices)
            ax.set_xticklabels(xtick_labels, rotation=45)
        
        ax.set_title('Signal Strength Over Time')
        ax.set_xlabel('Time')
        ax_twin.set_ylabel('Price')
        
        # Add colorbar
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1)), 
                           ax=ax, orientation='vertical', pad=0.01)
        cbar.set_label('Signal Strength')
        
        fig.tight_layout()
        return fig
    
    def plot_trade_analysis(self, figsize=(15, 10)):
        """Plot trade analysis charts."""
        if self.trades is None or len(self.trades) == 0:
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Trade PnL distribution
        sns.histplot(self.trades['pnl'], kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('PnL Distribution')
        axes[0, 0].set_xlabel('PnL')
        
        # Trade PnL over time
        axes[0, 1].plot(self.trades.index, self.trades['pnl'].cumsum())
        axes[0, 1].set_title('Cumulative PnL')
        axes[0, 1].set_xlabel('Trade Number')
        
        # Win/Loss by trade size
        if 'trade_size' in self.trades.columns:
            trade_result = np.where(self.trades['pnl'] > 0, 'Win', 'Loss')
            axes[1, 0].scatter(self.trades['trade_size'], self.trades['pnl'], 
                              c=np.where(self.trades['pnl'] > 0, 'g', 'r'), alpha=0.6)
            axes[1, 0].set_title('PnL vs Trade Size')
            axes[1, 0].set_xlabel('Trade Size')
            axes[1, 0].set_ylabel('PnL')
        
        # Trade duration analysis
        if 'duration' in self.trades.columns:
            sns.boxplot(x=np.where(self.trades['pnl'] > 0, 'Win', 'Loss'), 
                       y=self.trades['duration'], ax=axes[1, 1])
            axes[1, 1].set_title('Trade Duration by Outcome')
            axes[1, 1].set_xlabel('Outcome')
            axes[1, 1].set_ylabel('Duration (bars)')
        
        fig.tight_layout()
        return fig
    
    def summary(self) -> str:
        """Return a summary of backtest results."""
        return f"""
        Backtest Summary:
        -----------------
        Period: {self.start_time} to {self.end_time}
        Starting Capital: ${self.config.starting_capital:,.2f}
        Final Equity: ${self.equity_curve.iloc[-1]:,.2f} if self.equity_curve is not None else 'N/A'
        
        Performance Metrics:
        -------------------
        Total Return: {self.total_return:.2%}
        Annualized Return: {self.annual_return:.2%}
        Sharpe Ratio: {self.sharpe_ratio:.2f}
        Volatility (ann.): {self.volatility:.2%}
        Max Drawdown: {self.max_drawdown:.2%}
        
        Trading Statistics:
        ------------------
        Number of Trades: {len(self.trades) if self.trades is not None else 0}
        Win Rate: {self.win_rate:.2%}
        Profit Factor: {self.profit_factor:.2f}
        """


class Backtester:
    """Backtesting engine for simulating trading strategies."""
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config if config is not None else BacktestConfig()
    
    def _calculate_slippage(self, trade_size: float, price: float) -> float:
        """Calculate slippage for a given trade."""
        if self.config.slippage_model == "none":
            return 0.0
        elif self.config.slippage_model == "fixed":
            return self.config.slippage_value * abs(trade_size)
        elif self.config.slippage_model == "percentage":
            return price * self.config.slippage_value * abs(trade_size)
        else:
            return 0.0
    
    def _calculate_commission(self, trade_size: float, price: float) -> float:
        """Calculate commission for a given trade."""
        return abs(trade_size) * price * self.config.commission_rate
    
    def _calculate_market_impact(self, trade_size: float, price: float, volume: Optional[float]) -> float:
        """Calculate market impact for a given trade."""
        if self.config.market_impact_model == "none" or volume is None or volume == 0:
            return 0.0
            
        # Normalize trade size as percentage of volume
        normalized_size = abs(trade_size) / volume
            
        if self.config.market_impact_model == "linear":
            return price * normalized_size * self.config.market_impact_factor
        elif self.config.market_impact_model == "square_root":
            return price * (normalized_size ** 0.5) * self.config.market_impact_factor
        else:
            return 0.0
    
    def _handle_trade(self, trade_size: float, price: float, volume: Optional[float], 
                     current_capital: float, current_position: float) -> Dict[str, float]:
        """Process a trade with slippage, commission, and market impact."""
        # Round trade size to lot size
        trade_size = round(trade_size / self.config.lot_size) * self.config.lot_size
        
        # Skip if trade size is zero
        if trade_size == 0:
            return {
                'executed_size': 0.0,
                'executed_price': price,
                'commission': 0.0,
                'slippage': 0.0,
                'market_impact': 0.0,
                'total_cost': 0.0,
                'new_position': current_position,
                'new_capital': current_capital
            }
        
        # Calculate trading costs
        slippage = self._calculate_slippage(trade_size, price)
        commission = self._calculate_commission(trade_size, price)
        market_impact = self._calculate_market_impact(trade_size, price, volume)
        
        # Calculate effective price
        sign = 1 if trade_size > 0 else -1
        effective_price = price + sign * (slippage / abs(trade_size) + market_impact / abs(trade_size))
        
        # Calculate total cost
        total_cost = abs(trade_size) * effective_price + commission
        
        # Update position and capital
        new_position = current_position + trade_size
        new_capital = current_capital - trade_size * effective_price - commission
        
        return {
            'executed_size': trade_size,
            'executed_price': effective_price,
            'commission': commission,
            'slippage': slippage,
            'market_impact': market_impact,
            'total_cost': total_cost,
            'new_position': new_position,
            'new_capital': new_capital
        }
    
    def _calculate_equity(self, capital: float, position: float, price: float) -> float:
        """Calculate equity from capital and position."""
        return capital + position * price
    
    def _calculate_drawdown(self, equity: pd.Series) -> pd.Series:
        """Calculate drawdown from equity curve."""
        # Calculate running maximum
        running_max = equity.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity / running_max) - 1
        
        return drawdown
    
    def backtest(self, strategy: Strategy, data: pd.DataFrame) -> BacktestResult:
        """Run backtest of a strategy on historical data."""
        # Prepare result object
        result = BacktestResult(self.config, data)
        
        # Initialize tracking variables
        current_position = 0.0
        current_capital = self.config.starting_capital
        
        # Initialize tracking dataframes
        if self.config.track_positions_history:
            positions_history = pd.DataFrame(index=data.index)
            positions_history['position'] = 0.0
        
        if self.config.track_capital_history:
            capital_history = pd.DataFrame(index=data.index)
            capital_history['capital'] = self.config.starting_capital
            capital_history['equity'] = self.config.starting_capital
        
        if self.config.track_trade_history:
            trades = []
            trade_index = 0
            current_trade = None
        
        # Set up context for strategy execution
        context = {
            'current_position': current_position,
            'current_capital': current_capital,
            'backtest_config': self.config
        }
        
        # Run strategy in a single pass
        result.context = strategy.run(data, context)
        
        # Extract signals and positions from strategy result
        result.signals = result.context.get('signal')
        target_positions = result.context.get('positions')
        
        # Execute trades
        for i in range(len(data)):
            row = data.iloc[i]
            price = row['close']
            volume = row.get('volume', None)
            
            # Skip if no target positions
            if target_positions is None:
                continue
                
            # Calculate trade size
            target_position = target_positions[i]
            target_position_value = target_position * price
            
            # Apply position size limit
            max_position_value = self.config.position_size_limit * current_capital
            if abs(target_position_value) > max_position_value:
                target_position = np.sign(target_position) * (max_position_value / price)
            
            # Calculate trade size
            trade_size = target_position - current_position
            
            # Execute trade
            trade_result = self._handle_trade(
                trade_size, 
                price, 
                volume, 
                current_capital, 
                current_position
            )
            
            # Update current position and capital
            current_position = trade_result['new_position']
            current_capital = trade_result['new_capital']
            
            # Track positions history
            if self.config.track_positions_history:
                positions_history.loc[data.index[i], 'position'] = current_position
            
            # Track trade history
            if self.config.track_trade_history and trade_size != 0:
                # Start new trade
                if current_trade is None or np.sign(trade_size) != np.sign(current_trade.get('position', 0)):
                    # Close existing trade if it exists
                    if current_trade is not None:
                        trades.append(current_trade)
                    
                    # Start new trade
                    current_trade = {
                        'entry_time': data.index[i],
                        'entry_price': trade_result['executed_price'],
                        'position': trade_size,
                        'trade_size': abs(trade_size),
                        'commission': trade_result['commission'],
                        'slippage': trade_result['slippage'],
                        'market_impact': trade_result['market_impact']
                    }
                else:
                    # Add to existing trade
                    current_trade['position'] += trade_size
                    current_trade['trade_size'] += abs(trade_size)
                    current_trade['commission'] += trade_result['commission']
                    current_trade['slippage'] += trade_result['slippage']
                    current_trade['market_impact'] += trade_result['market_impact']
                    
                # Close trade if position is zero
                if current_position == 0:
                    current_trade['exit_time'] = data.index[i]
                    current_trade['exit_price'] = trade_result['executed_price']
                    current_trade['pnl'] = (current_trade['exit_price'] - current_trade['entry_price']) * \
                                          current_trade['position'] - \
                                          current_trade['commission'] - \
                                          current_trade['slippage'] - \
                                          current_trade['market_impact']
                    current_trade['duration'] = i - data.index.get_loc(current_trade['entry_time'])
                    
                    trades.append(current_trade)
                    current_trade = None
            
            # Calculate equity and update history
            if self.config.track_capital_history:
                equity = self._calculate_equity(current_capital, current_position, price)
                capital_history.loc[data.index[i], 'capital'] = current_capital
                capital_history.loc[data.index[i], 'equity'] = equity
        
        # Close any open trades at the end of the backtest
        if self.config.track_trade_history and current_trade is not None:
            last_price = data['close'].iloc[-1]
            current_trade['exit_time'] = data.index[-1]
            current_trade['exit_price'] = last_price
            current_trade['pnl'] = (current_trade['exit_price'] - current_trade['entry_price']) * \
                                  current_trade['position'] - \
                                  current_trade['commission'] - \
                                  current_trade['slippage'] - \
                                  current_trade['market_impact']
            current_trade['duration'] = len(data) - data.index.get_loc(current_trade['entry_time'])
            
            trades.append(current_trade)
        
        # Store results
        if self.config.track_positions_history:
            result.positions = positions_history
        
        if self.config.track_capital_history:
            result.equity_curve = capital_history['equity']
            
            # Calculate drawdown
            if self.config.track_drawdowns:
                result.drawdown_curve = self._calculate_drawdown(result.equity_curve)
        
        if self.config.track_trade_history and trades:
            result.trades = pd.DataFrame(trades)
            if not result.trades.empty:
                result.trades.set_index('entry_time', inplace=True)
        
        # Calculate metrics
        result.calculate_metrics()
        
        return result
