"""
Integration module for ML-based trading strategies with SCEF.
This module provides functions to integrate ML models with the SCEF framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import os
import logging

# Import SCEF components
from ..strategy_dsl import (
    Strategy, create_indicator, create_signal, 
    create_allocation, create_risk_control, 
    create_execution, create_post_trade
)

# Import ML modules
from .features import prepare_ml_features, normalize_features
from .models import MLModel, ml_signal_generator, create_signal_from_predictions
from .reinforcement import rl_signal_generator
from .ensemble import ensemble_signal_generator, EnsembleModel, StrategyEnsemble

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_ml_strategy(name: str, model_type: str = 'classifier', model_name: str = 'random_forest',
                      model_params: Dict[str, Any] = None, prediction_horizon: int = 1, 
                      position_sizing: float = 1.0, risk_control_params: Dict[str, Any] = None) -> Strategy:
    """
    Create a complete trading strategy using a machine learning model.
    
    Args:
        name: Strategy name
        model_type: Type of model ('classifier' or 'regressor')
        model_name: Name of the model ('random_forest', 'logistic_regression', etc.)
        model_params: Parameters for the model
        prediction_horizon: Number of periods ahead to predict
        position_sizing: Maximum position size (as fraction of capital)
        risk_control_params: Parameters for risk control
        
    Returns:
        Strategy object
    """
    # Create strategy
    strategy = Strategy(name)
    
    # Add ML signal generator
    def ml_signal_wrapper(data, context):
        return ml_signal_generator(
            data=data,
            context=context,
            model_type=model_type,
            model_name=model_name,
            model_params=model_params,
            prediction_horizon=prediction_horizon
        )
    
    strategy.add_signal(create_signal("ml_signal", ml_signal_wrapper))
    
    # Add position sizing
    def position_sizer(signal, data, context, max_position=position_sizing):
        return signal * max_position
    
    strategy.add_allocation(create_allocation("position_sizer", position_sizer))
    
    # Add risk control
    def max_position_limit(position, data, context, max_position=position_sizing):
        return np.clip(position, -max_position, max_position)
    
    strategy.add_risk_control(create_risk_control("max_position", max_position_limit))
    
    # Add volatility-based risk control if params provided
    if risk_control_params is not None and 'volatility_lookback' in risk_control_params:
        def volatility_scaling(position, data, context, lookback=risk_control_params.get('volatility_lookback', 20), 
                              target_vol=risk_control_params.get('target_volatility', 0.01),
                              max_leverage=risk_control_params.get('max_leverage', 2.0)):
            # Calculate historical volatility
            returns = data['close'].pct_change().fillna(0)
            vol = returns.rolling(window=lookback).std().iloc[-1]
            
            # Scale position based on volatility
            if vol > 0:
                scale_factor = min(target_vol / vol, max_leverage)
                return position * scale_factor
            else:
                return position
        
        strategy.add_risk_control(create_risk_control("volatility_scaling", volatility_scaling))
    
    # Add execution component
    def slippage_model(trades, data, context, slippage=0.001):
        # Apply slippage to trades
        slipped_trades = np.copy(trades)
        for i in range(len(trades)):
            if trades[i] > 0:  # Buy
                slipped_trades[i] = trades[i] * (1 + slippage)
            elif trades[i] < 0:  # Sell
                slipped_trades[i] = trades[i] * (1 - slippage)
        return slipped_trades
    
    strategy.add_execution(create_execution("slippage_model", slippage_model))
    
    # Add post-trade analysis
    def performance_tracker(trades, data, context, price_col='close'):
        # Get prices
        prices = data[price_col].values
        
        # Initialize arrays
        returns = np.zeros(len(trades))
        equity = np.ones(len(trades)) * 10000.0  # Assuming starting capital of $10,000
        
        # Calculate returns and equity
        position = 0.0
        for i in range(1, len(trades)):
            # Calculate returns based on position
            price_return = prices[i] / prices[i-1] - 1.0
            trade_return = position * price_return
            returns[i] = trade_return
            
            # Update equity
            equity[i] = equity[i-1] * (1 + trade_return)
            
            # Update position after trade
            position += trades[i]
        
        # Store in context
        if 'performance' not in context:
            context['performance'] = {}
        context['performance']['returns'] = returns
        context['performance']['equity'] = equity
        context['performance']['final_equity'] = equity[-1]
        
        # Calculate performance metrics
        if len(returns) > 1:
            annual_return = (equity[-1] / equity[0]) ** (252 / len(returns)) - 1
            volatility = np.std(returns) * np.sqrt(252)
            sharpe = annual_return / volatility if volatility > 0 else 0
            max_dd = 1 - min([equity[i] / max(equity[:i+1]) for i in range(1, len(equity))])
            
            context['performance']['annual_return'] = annual_return
            context['performance']['volatility'] = volatility
            context['performance']['sharpe'] = sharpe
            context['performance']['max_drawdown'] = max_dd
        
        return returns
    
    strategy.add_post_trade(create_post_trade("performance_tracker", performance_tracker))
    
    return strategy


def create_rl_strategy(name: str, window_size: int = 20, action_space: int = 3,
                      model_path: Optional[str] = None, position_sizing: float = 1.0,
                      risk_control_params: Dict[str, Any] = None) -> Strategy:
    """
    Create a trading strategy using reinforcement learning.
    
    Args:
        name: Strategy name
        window_size: Number of past periods to include in state
        action_space: Number of discrete actions
        model_path: Path to pre-trained RL model
        position_sizing: Maximum position size (as fraction of capital)
        risk_control_params: Parameters for risk control
        
    Returns:
        Strategy object
    """
    # Create strategy
    strategy = Strategy(name)
    
    # Add RL signal generator
    def rl_signal_wrapper(data, context):
        return rl_signal_generator(
            data=data,
            context=context,
            window_size=window_size,
            action_space=action_space,
            model_path=model_path
        )
    
    strategy.add_signal(create_signal("rl_signal", rl_signal_wrapper))
    
    # Add position sizing
    def position_sizer(signal, data, context, max_position=position_sizing):
        return signal * max_position
    
    strategy.add_allocation(create_allocation("position_sizer", position_sizer))
    
    # Add risk control
    def max_position_limit(position, data, context, max_position=position_sizing):
        return np.clip(position, -max_position, max_position)
    
    strategy.add_risk_control(create_risk_control("max_position", max_position_limit))
    
    # Add volatility-based risk control if params provided
    if risk_control_params is not None and 'volatility_lookback' in risk_control_params:
        def volatility_scaling(position, data, context, lookback=risk_control_params.get('volatility_lookback', 20), 
                              target_vol=risk_control_params.get('target_volatility', 0.01),
                              max_leverage=risk_control_params.get('max_leverage', 2.0)):
            # Calculate historical volatility
            returns = data['close'].pct_change().fillna(0)
            vol = returns.rolling(window=lookback).std().iloc[-1]
            
            # Scale position based on volatility
            if vol > 0:
                scale_factor = min(target_vol / vol, max_leverage)
                return position * scale_factor
            else:
                return position
        
        strategy.add_risk_control(create_risk_control("volatility_scaling", volatility_scaling))
    
    # Add execution and post-trade components (same as ML strategy)
    def slippage_model(trades, data, context, slippage=0.001):
        # Apply slippage to trades
        slipped_trades = np.copy(trades)
        for i in range(len(trades)):
            if trades[i] > 0:  # Buy
                slipped_trades[i] = trades[i] * (1 + slippage)
            elif trades[i] < 0:  # Sell
                slipped_trades[i] = trades[i] * (1 - slippage)
        return slipped_trades
    
    strategy.add_execution(create_execution("slippage_model", slippage_model))
    
    def performance_tracker(trades, data, context, price_col='close'):
        # Get prices
        prices = data[price_col].values
        
        # Initialize arrays
        returns = np.zeros(len(trades))
        equity = np.ones(len(trades)) * 10000.0  # Assuming starting capital of $10,000
        
        # Calculate returns and equity
        position = 0.0
        for i in range(1, len(trades)):
            # Calculate returns based on position
            price_return = prices[i] / prices[i-1] - 1.0
            trade_return = position * price_return
            returns[i] = trade_return
            
            # Update equity
            equity[i] = equity[i-1] * (1 + trade_return)
            
            # Update position after trade
            position += trades[i]
        
        # Store in context
        if 'performance' not in context:
            context['performance'] = {}
        context['performance']['returns'] = returns
        context['performance']['equity'] = equity
        context['performance']['final_equity'] = equity[-1]
        
        return returns
    
    strategy.add_post_trade(create_post_trade("performance_tracker", performance_tracker))
    
    return strategy


def create_ensemble_strategy(name: str, strategies: List[Strategy], 
                           weights: Optional[List[float]] = None,
                           ensemble_method: str = 'weighted_average',
                           adaptation_method: Optional[str] = None) -> Strategy:
    """
    Create an ensemble strategy combining multiple strategies.
    
    Args:
        name: Strategy name
        strategies: List of strategy objects
        weights: List of weights for each strategy
        ensemble_method: Method to combine strategies
        adaptation_method: Method for adapting weights
        
    Returns:
        Strategy object
    """
    # Create strategy
    strategy = Strategy(name)
    
    # Initialize ensemble
    ensemble = StrategyEnsemble(
        strategies=strategies,
        weights=weights,
        ensemble_method=ensemble_method,
        adaptation_method=adaptation_method
    )
    
    # Add ensemble signal generator
    def ensemble_signal_wrapper(data, context):
        if 'ensemble' not in context:
            context['ensemble'] = {}
        context['ensemble']['strategies'] = strategies
        
        # Generate ensemble signal
        return ensemble.generate_signal(data, context)
    
    strategy.add_signal(create_signal("ensemble_signal", ensemble_signal_wrapper))
    
    # Add position sizing
    def position_sizer(signal, data, context, max_position=1.0):
        return signal * max_position
    
    strategy.add_allocation(create_allocation("position_sizer", position_sizer))
    
    # Add risk control
    def max_position_limit(position, data, context, max_position=1.0):
        return np.clip(position, -max_position, max_position)
    
    strategy.add_risk_control(create_risk_control("max_position", max_position_limit))
    
    # Add execution
    def slippage_model(trades, data, context, slippage=0.001):
        # Apply slippage to trades
        slipped_trades = np.copy(trades)
        for i in range(len(trades)):
            if trades[i] > 0:  # Buy
                slipped_trades[i] = trades[i] * (1 + slippage)
            elif trades[i] < 0:  # Sell
                slipped_trades[i] = trades[i] * (1 - slippage)
        return slipped_trades
    
    strategy.add_execution(create_execution("slippage_model", slippage_model))
    
    # Add post-trade analysis
    def performance_tracker(trades, data, context, price_col='close'):
        # Get prices
        prices = data[price_col].values
        
        # Initialize arrays
        returns = np.zeros(len(trades))
        equity = np.ones(len(trades)) * 10000.0  # Assuming starting capital of $10,000
        
        # Calculate returns and equity
        position = 0.0
        for i in range(1, len(trades)):
            # Calculate returns based on position
            price_return = prices[i] / prices[i-1] - 1.0
            trade_return = position * price_return
            returns[i] = trade_return
            
            # Update equity
            equity[i] = equity[i-1] * (1 + trade_return)
            
            # Update position after trade
            position += trades[i]
        
        # Store in context
        if 'performance' not in context:
            context['performance'] = {}
        context['performance']['returns'] = returns
        context['performance']['equity'] = equity
        context['performance']['final_equity'] = equity[-1]
        
        # Update ensemble weights based on performance
        if adaptation_method is not None:
            ensemble.update_performance(data, context)
        
        return returns
    
    strategy.add_post_trade(create_post_trade("performance_tracker", performance_tracker))
    
    return strategy


def register_ml_components():
    """Register ML-based components with the SCEF framework."""
    from .. import strategy_dsl
    
    # Register ML-based signal generators
    strategy_dsl.register_signal_component("ml_signal", ml_signal_generator)
    strategy_dsl.register_signal_component("rl_signal", rl_signal_generator)
    strategy_dsl.register_signal_component("ensemble_signal", ensemble_signal_generator)
    
    # Register other ML-based components as needed
    logger.info("ML components registered with SCEF framework.")
