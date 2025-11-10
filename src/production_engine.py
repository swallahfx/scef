"""
Production deployment system for running trading strategies in real-time.

This module provides a framework for deploying strategies to production,
including state management, version control, and monitoring.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import queue
import time
import logging
import uuid
import json
import os
import pickle
from enum import Enum

from src.strategy_dsl import Strategy


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('production_system')


class StrategyState(Enum):
    """Possible states of a deployed strategy."""
    INITIALIZING = 'initializing'
    RUNNING = 'running'
    PAUSED = 'paused'
    STOPPED = 'stopped'
    ERROR = 'error'


@dataclass
class StrategyMetrics:
    """Performance metrics for a strategy."""
    cumulative_pnl: float = 0.0
    daily_pnl: Dict[str, float] = field(default_factory=dict)
    sharpe_ratio: float = 0.0
    drawdown: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0
    average_trade_pnl: float = 0.0
    average_trade_duration: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    daily_returns: List[float] = field(default_factory=list)
    
    def update(self, new_data: Dict[str, Any]):
        """Update metrics with new data."""
        for key, value in new_data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, dict) and all(isinstance(k, datetime) for k in value.keys()):
                result[key] = {k.isoformat(): v for k, v in value.items()}
            else:
                result[key] = value
        return result


@dataclass
class StrategyConfig:
    """Configuration for a deployed strategy."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "unnamed_strategy"
    version: str = "0.0.1"
    max_position: float = 1.0
    position_limits: Dict[str, float] = field(default_factory=dict)
    risk_limits: Dict[str, float] = field(default_factory=dict)
    execution_params: Dict[str, Any] = field(default_factory=dict)
    schedule: Dict[str, Any] = field(default_factory=dict)
    market_data_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'StrategyConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


class DeployedStrategy:
    """A strategy deployed to production."""
    
    def __init__(self, strategy: Strategy, config: StrategyConfig):
        """
        Initialize deployed strategy.
        
        Args:
            strategy: Strategy to deploy
            config: Strategy configuration
        """
        self.strategy = strategy
        self.config = config
        self.state = StrategyState.INITIALIZING
        self.metrics = StrategyMetrics()
        self.context = {}
        self.trade_history = []
        self.position_history = []
        self.state_history = []
        self.error_log = []
        self.last_run_time = None
        self.next_run_time = None
        self.lock = threading.RLock()  # Reentrant lock for thread safety
    
    def initialize(self) -> bool:
        """Initialize strategy for production deployment."""
        with self.lock:
            try:
                logger.info(f"Initializing strategy {self.config.name} (v{self.config.version})")
                
                # Set initial context
                self.context = {
                    'current_position': 0.0,
                    'current_capital': 0.0,
                    'config': self.config.to_dict()
                }
                
                # Set state to running
                self.state = StrategyState.RUNNING
                self._log_state_change(StrategyState.INITIALIZING, StrategyState.RUNNING, "Initialization complete")
                
                return True
                
            except Exception as e:
                logger.error(f"Error initializing strategy: {str(e)}")
                self.state = StrategyState.ERROR
                self._log_state_change(StrategyState.INITIALIZING, StrategyState.ERROR, str(e))
                self._log_error(str(e), "initialization")
                return False
    
    def run(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run strategy on current market data."""
        with self.lock:
            if self.state != StrategyState.RUNNING:
                logger.warning(f"Strategy {self.config.name} not in RUNNING state, current state: {self.state}")
                return {}
                
            try:
                logger.info(f"Running strategy {self.config.name} (v{self.config.version})")
                
                # Record run time
                self.last_run_time = datetime.now()
                
                # Run strategy
                result = self.strategy.run(market_data, self.context.copy())
                
                # Update context with results
                self.context.update(result)
                
                # Extract and record positions
                if 'positions' in result:
                    self._record_position(result['positions'][-1], market_data.index[-1])
                
                # Extract and record trades
                if 'trades' in result:
                    self._record_trades(result['trades'], market_data)
                
                # Update metrics
                if 'returns' in result:
                    self._update_metrics(result)
                
                # Schedule next run
                self._schedule_next_run()
                
                return result
                
            except Exception as e:
                logger.error(f"Error running strategy: {str(e)}")
                self.state = StrategyState.ERROR
                self._log_state_change(StrategyState.RUNNING, StrategyState.ERROR, str(e))
                self._log_error(str(e), "execution")
                return {}
    
    def pause(self) -> bool:
        """Pause strategy execution."""
        with self.lock:
            if self.state != StrategyState.RUNNING:
                logger.warning(f"Cannot pause strategy {self.config.name}, current state: {self.state}")
                return False
                
            previous_state = self.state
            self.state = StrategyState.PAUSED
            self._log_state_change(previous_state, StrategyState.PAUSED, "Manual pause")
            logger.info(f"Paused strategy {self.config.name}")
            return True
    
    def resume(self) -> bool:
        """Resume strategy execution."""
        with self.lock:
            if self.state != StrategyState.PAUSED:
                logger.warning(f"Cannot resume strategy {self.config.name}, current state: {self.state}")
                return False
                
            previous_state = self.state
            self.state = StrategyState.RUNNING
            self._log_state_change(previous_state, StrategyState.RUNNING, "Manual resume")
            logger.info(f"Resumed strategy {self.config.name}")
            
            # Schedule next run
            self._schedule_next_run()
            
            return True
    
    def stop(self) -> bool:
        """Stop strategy execution."""
        with self.lock:
            if self.state in [StrategyState.STOPPED, StrategyState.ERROR]:
                logger.warning(f"Strategy {self.config.name} already in {self.state} state")
                return False
                
            previous_state = self.state
            self.state = StrategyState.STOPPED
            self._log_state_change(previous_state, StrategyState.STOPPED, "Manual stop")
            logger.info(f"Stopped strategy {self.config.name}")
            return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of strategy."""
        with self.lock:
            return {
                'id': self.config.id,
                'name': self.config.name,
                'version': self.config.version,
                'state': self.state.value,
                'last_run_time': self.last_run_time.isoformat() if self.last_run_time else None,
                'next_run_time': self.next_run_time.isoformat() if self.next_run_time else None,
                'metrics': self.metrics.to_dict(),
                'current_position': self.context.get('current_position', 0.0),
                'error_count': len(self.error_log)
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self.lock:
            return self.metrics.to_dict()
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update strategy configuration."""
        with self.lock:
            try:
                # Create new config by updating existing config
                updated_config = self.config.to_dict()
                updated_config.update(new_config)
                
                # Create new config object
                self.config = StrategyConfig.from_dict(updated_config)
                
                # Update context
                self.context['config'] = self.config.to_dict()
                
                logger.info(f"Updated configuration for strategy {self.config.name}")
                return True
                
            except Exception as e:
                logger.error(f"Error updating config: {str(e)}")
                self._log_error(str(e), "config_update")
                return False
    
    def _record_position(self, position: float, timestamp: Any):
        """Record position change."""
        self.position_history.append({
            'timestamp': timestamp,
            'position': position
        })
        
        # Update context
        self.context['current_position'] = position
    
    def _record_trades(self, trades: np.ndarray, market_data: pd.DataFrame):
        """Record trades."""
        # Skip if no trades
        if np.all(trades == 0):
            return
            
        # Record each non-zero trade
        for i, trade_size in enumerate(trades):
            if trade_size == 0:
                continue
                
            timestamp = market_data.index[i]
            price = market_data['close'].iloc[i]
            
            trade_record = {
                'timestamp': timestamp,
                'size': float(trade_size),
                'price': float(price),
                'value': float(trade_size * price),
                'strategy_id': self.config.id,
                'strategy_version': self.config.version
            }
            
            self.trade_history.append(trade_record)
            
            # Update metrics
            self.metrics.num_trades += 1
    
    def _update_metrics(self, result: Dict[str, Any]):
        """Update performance metrics."""
        # Extract returns
        returns = result.get('returns', [])
        if not isinstance(returns, list) and not isinstance(returns, np.ndarray):
            returns = [returns]
            
        # Skip if no returns
        if len(returns) == 0:
            return
            
        # Calculate daily PnL
        if isinstance(result.get('timestamp', None), pd.Timestamp):
            date_str = result['timestamp'].date().isoformat()
            self.metrics.daily_pnl[date_str] = float(returns[-1])
        
        # Update cumulative PnL
        self.metrics.cumulative_pnl += float(returns[-1])
        
        # Update daily returns
        self.metrics.daily_returns.append(float(returns[-1]))
        
        # Calculate Sharpe ratio (if enough data)
        if len(self.metrics.daily_returns) > 1:
            mean_return = np.mean(self.metrics.daily_returns)
            std_return = np.std(self.metrics.daily_returns)
            if std_return > 0:
                self.metrics.sharpe_ratio = mean_return / std_return * np.sqrt(252)
        
        # Calculate drawdown
        if self.metrics.cumulative_pnl > 0:
            peak = np.maximum.accumulate(
                [0] + [self.metrics.cumulative_pnl]
            )[-1]
            if peak > 0:
                current_drawdown = (self.metrics.cumulative_pnl - peak) / peak
                self.metrics.drawdown = min(self.metrics.drawdown, current_drawdown)
        
        # Calculate average trade PnL
        if self.metrics.num_trades > 0:
            self.metrics.average_trade_pnl = self.metrics.cumulative_pnl / self.metrics.num_trades
        
        # Calculate win rate
        if len(self.trade_history) > 0:
            profitable_trades = sum(1 for trade in self.trade_history if trade.get('pnl', 0) > 0)
            self.metrics.win_rate = profitable_trades / len(self.trade_history)
    
    def _log_state_change(self, from_state: StrategyState, to_state: StrategyState, reason: str):
        """Log state change."""
        self.state_history.append({
            'timestamp': datetime.now(),
            'from_state': from_state.value,
            'to_state': to_state.value,
            'reason': reason
        })
    
    def _log_error(self, error_message: str, error_type: str):
        """Log error."""
        self.error_log.append({
            'timestamp': datetime.now(),
            'type': error_type,
            'message': error_message
        })
    
    def _schedule_next_run(self):
        """Schedule next strategy run."""
        # Default to running every minute
        seconds_to_next_run = 60
        
        # Check if custom schedule is configured
        if 'interval_seconds' in self.config.schedule:
            seconds_to_next_run = self.config.schedule['interval_seconds']
        
        # Set next run time
        self.next_run_time = datetime.now() + timedelta(seconds=seconds_to_next_run)
    
    def save_state(self, directory: str) -> str:
        """Save strategy state to disk."""
        with self.lock:
            try:
                # Create directory if it doesn't exist
                os.makedirs(directory, exist_ok=True)
                
                # Create state dictionary
                state = {
                    'config': self.config.to_dict(),
                    'metrics': self.metrics.to_dict(),
                    'context': self.context,
                    'trade_history': self.trade_history,
                    'position_history': self.position_history,
                    'state_history': self.state_history,
                    'error_log': self.error_log,
                    'last_run_time': self.last_run_time.isoformat() if self.last_run_time else None,
                    'next_run_time': self.next_run_time.isoformat() if self.next_run_time else None,
                    'state': self.state.value
                }
                
                # Save state to file
                filename = f"{self.config.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                filepath = os.path.join(directory, filename)
                
                with open(filepath, 'w') as f:
                    json.dump(state, f, indent=2)
                
                # Save strategy object separately
                strategy_filename = f"{self.config.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_strategy.pkl"
                strategy_filepath = os.path.join(directory, strategy_filename)
                
                with open(strategy_filepath, 'wb') as f:
                    pickle.dump(self.strategy, f)
                
                logger.info(f"Saved strategy state to {filepath}")
                return filepath
                
            except Exception as e:
                logger.error(f"Error saving state: {str(e)}")
                self._log_error(str(e), "state_save")
                return ""
    
    @classmethod
    def load_state(cls, filepath: str, strategy_filepath: str) -> Optional['DeployedStrategy']:
        """Load strategy state from disk."""
        try:
            # Load state from file
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Load strategy object
            with open(strategy_filepath, 'rb') as f:
                strategy = pickle.load(f)
            
            # Create config object
            config = StrategyConfig.from_dict(state['config'])
            
            # Create deployed strategy
            deployed_strategy = cls(strategy, config)
            
            # Restore metrics
            deployed_strategy.metrics.update(state['metrics'])
            
            # Restore history
            deployed_strategy.trade_history = state['trade_history']
            deployed_strategy.position_history = state['position_history']
            deployed_strategy.state_history = state['state_history']
            deployed_strategy.error_log = state['error_log']
            
            # Restore context
            deployed_strategy.context = state['context']
            
            # Restore timing
            if state['last_run_time']:
                deployed_strategy.last_run_time = datetime.fromisoformat(state['last_run_time'])
            
            if state['next_run_time']:
                deployed_strategy.next_run_time = datetime.fromisoformat(state['next_run_time'])
            
            # Restore state
            deployed_strategy.state = StrategyState(state['state'])
            
            logger.info(f"Loaded strategy state from {filepath}")
            return deployed_strategy
            
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            return None


class ProductionEngine:
    """Engine for running multiple trading strategies in production."""
    
    def __init__(self):
        """Initialize production engine."""
        self.strategies = {}  # Map of strategy ID to DeployedStrategy
        self.market_data_cache = {}  # Cache of market data
        self.running = False
        self.thread = None
        self.queue = queue.Queue()  # Command queue
        self.lock = threading.RLock()  # Lock for thread safety
    
    def start(self):
        """Start production engine."""
        with self.lock:
            if self.running:
                logger.warning("Production engine already running")
                return
                
            self.running = True
            self.thread = threading.Thread(target=self._run_loop)
            self.thread.daemon = True  # Allow process to exit even if thread is running
            self.thread.start()
            
            logger.info("Started production engine")
    
    def stop(self):
        """Stop production engine."""
        with self.lock:
            if not self.running:
                logger.warning("Production engine not running")
                return
                
            self.running = False
            self.queue.put(('stop', None))
            
            if self.thread:
                self.thread.join(timeout=5.0)
                if self.thread.is_alive():
                    logger.warning("Production engine thread did not stop gracefully")
                
            logger.info("Stopped production engine")
    
    def deploy_strategy(self, strategy: Strategy, config: Optional[StrategyConfig] = None) -> str:
        """Deploy a new strategy to production."""
        with self.lock:
            # Create config if not provided
            if config is None:
                config = StrategyConfig(name=strategy.name)
            
            # Create deployed strategy
            deployed_strategy = DeployedStrategy(strategy, config)
            
            # Initialize strategy
            if not deployed_strategy.initialize():
                logger.error(f"Failed to initialize strategy {config.name}")
                return ""
            
            # Add to strategies
            self.strategies[deployed_strategy.config.id] = deployed_strategy
            
            # Schedule first run
            self._schedule_strategy_run(deployed_strategy.config.id)
            
            logger.info(f"Deployed strategy {config.name} (ID: {deployed_strategy.config.id})")
            return deployed_strategy.config.id
    
    def update_strategy(self, strategy_id: str, strategy: Strategy) -> bool:
        """Update an existing strategy."""
        with self.lock:
            if strategy_id not in self.strategies:
                logger.error(f"Strategy {strategy_id} not found")
                return False
                
            deployed_strategy = self.strategies[strategy_id]
            
            # Update strategy
            deployed_strategy.strategy = strategy
            
            logger.info(f"Updated strategy {deployed_strategy.config.name} (ID: {strategy_id})")
            return True
    
    def update_strategy_config(self, strategy_id: str, config_updates: Dict[str, Any]) -> bool:
        """Update configuration for an existing strategy."""
        with self.lock:
            if strategy_id not in self.strategies:
                logger.error(f"Strategy {strategy_id} not found")
                return False
                
            deployed_strategy = self.strategies[strategy_id]
            
            # Update config
            if not deployed_strategy.update_config(config_updates):
                logger.error(f"Failed to update config for strategy {strategy_id}")
                return False
                
            logger.info(f"Updated config for strategy {deployed_strategy.config.name} (ID: {strategy_id})")
            return True
    
    def pause_strategy(self, strategy_id: str) -> bool:
        """Pause a running strategy."""
        with self.lock:
            if strategy_id not in self.strategies:
                logger.error(f"Strategy {strategy_id} not found")
                return False
                
            deployed_strategy = self.strategies[strategy_id]
            
            # Pause strategy
            if not deployed_strategy.pause():
                logger.error(f"Failed to pause strategy {strategy_id}")
                return False
                
            logger.info(f"Paused strategy {deployed_strategy.config.name} (ID: {strategy_id})")
            return True
    
    def resume_strategy(self, strategy_id: str) -> bool:
        """Resume a paused strategy."""
        with self.lock:
            if strategy_id not in self.strategies:
                logger.error(f"Strategy {strategy_id} not found")
                return False
                
            deployed_strategy = self.strategies[strategy_id]
            
            # Resume strategy
            if not deployed_strategy.resume():
                logger.error(f"Failed to resume strategy {strategy_id}")
                return False
                
            logger.info(f"Resumed strategy {deployed_strategy.config.name} (ID: {strategy_id})")
            return True
    
    def stop_strategy(self, strategy_id: str) -> bool:
        """Stop a strategy."""
        with self.lock:
            if strategy_id not in self.strategies:
                logger.error(f"Strategy {strategy_id} not found")
                return False
                
            deployed_strategy = self.strategies[strategy_id]
            
            # Stop strategy
            if not deployed_strategy.stop():
                logger.error(f"Failed to stop strategy {strategy_id}")
                return False
                
            logger.info(f"Stopped strategy {deployed_strategy.config.name} (ID: {strategy_id})")
            return True
    
    def get_strategy_status(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a strategy."""
        with self.lock:
            if strategy_id not in self.strategies:
                logger.error(f"Strategy {strategy_id} not found")
                return None
                
            deployed_strategy = self.strategies[strategy_id]
            return deployed_strategy.get_status()
    
    def get_all_strategy_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all strategies."""
        with self.lock:
            return {
                strategy_id: deployed_strategy.get_status()
                for strategy_id, deployed_strategy in self.strategies.items()
            }
    
    def save_strategy_state(self, strategy_id: str, directory: str) -> str:
        """Save strategy state to disk."""
        with self.lock:
            if strategy_id not in self.strategies:
                logger.error(f"Strategy {strategy_id} not found")
                return ""
                
            deployed_strategy = self.strategies[strategy_id]
            return deployed_strategy.save_state(directory)
    
    def load_strategy_state(self, filepath: str, strategy_filepath: str) -> Optional[str]:
        """Load strategy state from disk."""
        with self.lock:
            # Load strategy
            deployed_strategy = DeployedStrategy.load_state(filepath, strategy_filepath)
            if deployed_strategy is None:
                logger.error(f"Failed to load strategy from {filepath}")
                return None
                
            # Add to strategies
            self.strategies[deployed_strategy.config.id] = deployed_strategy
            
            # Schedule run if running
            if deployed_strategy.state == StrategyState.RUNNING:
                self._schedule_strategy_run(deployed_strategy.config.id)
                
            logger.info(f"Loaded strategy {deployed_strategy.config.name} (ID: {deployed_strategy.config.id})")
            return deployed_strategy.config.id
    
    def add_market_data(self, symbol: str, data: pd.DataFrame):
        """Add market data to cache."""
        with self.lock:
            self.market_data_cache[symbol] = data
            logger.info(f"Added market data for {symbol}")
    
    def get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get market data from cache."""
        with self.lock:
            return self.market_data_cache.get(symbol)
    
    def _run_loop(self):
        """Main run loop for production engine."""
        logger.info("Starting production engine run loop")
        
        while self.running:
            try:
                # Check for commands with timeout
                try:
                    command, args = self.queue.get(timeout=1.0)
                    
                    # Handle command
                    if command == 'stop':
                        logger.info("Received stop command")
                        break
                    elif command == 'run_strategy':
                        strategy_id = args
                        self._run_strategy(strategy_id)
                    else:
                        logger.warning(f"Unknown command: {command}")
                        
                    # Mark command as done
                    self.queue.task_done()
                    
                except queue.Empty:
                    # No commands, check if any strategies need to be run
                    self._check_scheduled_runs()
            
            except Exception as e:
                logger.error(f"Error in production engine run loop: {str(e)}")
                
        logger.info("Exiting production engine run loop")
    
    def _check_scheduled_runs(self):
        """Check if any strategies need to be run."""
        with self.lock:
            current_time = datetime.now()
            
            for strategy_id, deployed_strategy in list(self.strategies.items()):
                # Skip if not running
                if deployed_strategy.state != StrategyState.RUNNING:
                    continue
                    
                # Skip if no next run time
                if deployed_strategy.next_run_time is None:
                    continue
                    
                # Check if time to run
                if current_time >= deployed_strategy.next_run_time:
                    # Schedule run
                    self._schedule_strategy_run(strategy_id)
    
    def _schedule_strategy_run(self, strategy_id: str):
        """Schedule a strategy to run."""
        self.queue.put(('run_strategy', strategy_id))
    
    def _run_strategy(self, strategy_id: str):
        """Run a strategy."""
        with self.lock:
            if strategy_id not in self.strategies:
                logger.error(f"Strategy {strategy_id} not found")
                return
                
            deployed_strategy = self.strategies[strategy_id]
            
            # Skip if not running
            if deployed_strategy.state != StrategyState.RUNNING:
                return
                
            # Get market data for strategy
            market_data = self._get_market_data_for_strategy(deployed_strategy)
            if market_data is None or len(market_data) == 0:
                logger.warning(f"No market data for strategy {strategy_id}")
                return
                
            # Run strategy
            result = deployed_strategy.run(market_data)
            
            # Handle result
            if not result:
                logger.warning(f"Strategy {strategy_id} returned empty result")
            
    def _get_market_data_for_strategy(self, deployed_strategy: DeployedStrategy) -> Optional[pd.DataFrame]:
        """Get market data for a strategy."""
        # Get symbol from config
        symbols = deployed_strategy.config.market_data_config.get('symbols', [])
        if not symbols:
            logger.warning(f"No symbols configured for strategy {deployed_strategy.config.name}")
            return None
            
        # Get market data for primary symbol
        primary_symbol = symbols[0]
        market_data = self.get_market_data(primary_symbol)
        if market_data is None:
            logger.warning(f"No market data for symbol {primary_symbol}")
            return None
            
        return market_data