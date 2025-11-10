"""
Ensemble strategies for machine learning in trading.
This module provides implementations for combining multiple ML models and strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import os
import logging

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class EnsembleModel:
    """Class for combining multiple ML models."""
    
    def __init__(self, models: List[Any], weights: Optional[List[float]] = None,
                 ensemble_method: str = 'weighted_average'):
        """
        Initialize the ensemble model.
        
        Args:
            models: List of model objects
            weights: List of weights for each model (must sum to 1)
            ensemble_method: Method to combine models ('weighted_average', 'voting', 'stacking')
        """
        self.models = models
        self.num_models = len(models)
        
        # Validate and normalize weights
        if weights is None:
            self.weights = np.ones(self.num_models) / self.num_models
        else:
            if len(weights) != self.num_models:
                raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({self.num_models})")
            
            # Normalize weights to sum to 1
            self.weights = np.array(weights) / sum(weights)
        
        self.ensemble_method = ensemble_method
        self.meta_model = None  # For stacking
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the ensemble.
        
        Args:
            features: DataFrame with features
            
        Returns:
            Array of predictions
        """
        if self.num_models == 0:
            logger.warning("No models in ensemble. Returning zeros.")
            return np.zeros(len(features))
        
        # Get predictions from each model
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict'):
                pred = model.predict(features)
                predictions.append(pred)
            else:
                logger.warning(f"Model {model} does not have predict method. Skipping.")
        
        # Combine predictions
        if self.ensemble_method == 'weighted_average':
            # Weighted average of predictions
            ensemble_pred = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                ensemble_pred += pred * self.weights[i]
        
        elif self.ensemble_method == 'voting':
            # Majority vote (for classification)
            predictions = np.array(predictions)
            ensemble_pred = np.zeros_like(predictions[0])
            
            # Convert probabilities to binary predictions if needed
            if np.issubdtype(predictions.dtype, np.float):
                binary_preds = (predictions > 0.5).astype(int)
            else:
                binary_preds = predictions
            
            # Count votes for each class
            ensemble_pred = np.mean(binary_preds, axis=0)
            
            # Convert back to binary predictions
            ensemble_pred = (ensemble_pred > 0.5).astype(float)
        
        elif self.ensemble_method == 'stacking':
            # Use meta-model for stacking (if available)
            if self.meta_model is not None and hasattr(self.meta_model, 'predict'):
                # Create meta-features
                meta_features = np.column_stack(predictions)
                
                # Generate predictions from meta-model
                ensemble_pred = self.meta_model.predict(meta_features)
            else:
                # Fall back to weighted average
                logger.warning("Meta-model not available for stacking. Using weighted average.")
                ensemble_pred = np.zeros_like(predictions[0])
                for i, pred in enumerate(predictions):
                    ensemble_pred += pred * self.weights[i]
        
        else:
            logger.warning(f"Unknown ensemble method: {self.ensemble_method}. Using weighted average.")
            # Fall back to weighted average
            ensemble_pred = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                ensemble_pred += pred * self.weights[i]
        
        return ensemble_pred
    
    def fit_meta_model(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
                       meta_model: Any = None):
        """
        Train meta-model for stacking ensemble.
        
        Args:
            X: Features for base models
            y: Target labels
            meta_model: Model to use for stacking (default: LogisticRegression)
        """
        if self.ensemble_method != 'stacking':
            logger.warning(f"Ensemble method is {self.ensemble_method}, not stacking. Meta-model will not be used.")
            return
        
        # Get predictions from each base model
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict'):
                pred = model.predict(X)
                predictions.append(pred)
            else:
                logger.warning(f"Model {model} does not have predict method. Skipping.")
        
        # Create meta-features
        meta_features = np.column_stack(predictions)
        
        # Initialize meta-model if not provided
        if meta_model is None:
            try:
                from sklearn.linear_model import LogisticRegression
                meta_model = LogisticRegression()
            except ImportError:
                logger.error("scikit-learn not available. Cannot create meta-model.")
                return
        
        # Train meta-model
        meta_model.fit(meta_features, y)
        
        # Store meta-model
        self.meta_model = meta_model
        
        logger.info("Meta-model trained successfully.")
    
    def update_weights(self, weights: List[float]):
        """
        Update ensemble weights.
        
        Args:
            weights: New weights for each model
        """
        if len(weights) != self.num_models:
            raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({self.num_models})")
        
        # Normalize weights to sum to 1
        self.weights = np.array(weights) / sum(weights)
        
        logger.info("Ensemble weights updated.")


class StrategyEnsemble:
    """Class for combining multiple trading strategies."""
    
    def __init__(self, strategies: List[Any], weights: Optional[List[float]] = None,
                 ensemble_method: str = 'weighted_average', adaptation_method: Optional[str] = None):
        """
        Initialize the strategy ensemble.
        
        Args:
            strategies: List of strategy objects
            weights: List of weights for each strategy (must sum to 1)
            ensemble_method: Method to combine strategies ('weighted_average', 'voting', 'dynamic')
            adaptation_method: Method for adapting weights ('fixed', 'momentum', 'performance')
        """
        self.strategies = strategies
        self.num_strategies = len(strategies)
        
        # Validate and normalize weights
        if weights is None:
            self.weights = np.ones(self.num_strategies) / self.num_strategies
        else:
            if len(weights) != self.num_strategies:
                raise ValueError(f"Number of weights ({len(weights)}) must match number of strategies ({self.num_strategies})")
            
            # Normalize weights to sum to 1
            self.weights = np.array(weights) / sum(weights)
        
        self.ensemble_method = ensemble_method
        self.adaptation_method = adaptation_method or 'fixed'
        self.performance_history = []
        
        # Initialize weight adaptation parameters
        self.momentum = 0.9
        self.learning_rate = 0.1
        self.performance_window = 20
    
    def generate_signal(self, data: pd.DataFrame, context: Dict[str, Any]) -> np.ndarray:
        """
        Generate trading signals using the ensemble.
        
        Args:
            data: DataFrame with OHLCV data
            context: Strategy context dictionary
            
        Returns:
            Array of trading signals
        """
        if self.num_strategies == 0:
            logger.warning("No strategies in ensemble. Returning zeros.")
            return np.zeros(len(data))
        
        # Get signals from each strategy
        signals = []
        for strategy in self.strategies:
            # Check if strategy has execute method
            if hasattr(strategy, 'execute'):
                # Execute strategy
                signal = strategy.execute(data, context)
            elif hasattr(strategy, 'generate_signal'):
                # Generate signal
                signal = strategy.generate_signal(data, context)
            elif callable(strategy):
                # Call strategy function
                signal = strategy(data, context)
            else:
                logger.warning(f"Strategy {strategy} does not have execute or generate_signal method. Skipping.")
                continue
            
            signals.append(signal)
        
        # Combine signals based on ensemble method
        if self.ensemble_method == 'weighted_average':
            # Weighted average of signals
            ensemble_signal = np.zeros_like(signals[0])
            for i, signal in enumerate(signals):
                ensemble_signal += signal * self.weights[i]
        
        elif self.ensemble_method == 'voting':
            # Majority vote (for classification-like signals)
            signals = np.array(signals)
            
            # Count votes for each direction (positive, zero, negative)
            positive_votes = np.sum(signals > 0, axis=0)
            negative_votes = np.sum(signals < 0, axis=0)
            
            # Generate ensemble signal based on votes
            ensemble_signal = np.zeros_like(signals[0])
            ensemble_signal[positive_votes > negative_votes] = 1.0
            ensemble_signal[negative_votes > positive_votes] = -1.0
        
        elif self.ensemble_method == 'dynamic':
            # Dynamic weighting based on recent performance
            if len(self.performance_history) >= self.performance_window:
                # Calculate recent performance for each strategy
                recent_performance = np.zeros(self.num_strategies)
                for i in range(self.num_strategies):
                    recent_performance[i] = np.mean([perf[i] for perf in self.performance_history[-self.performance_window:]])
                
                # Adjust weights based on performance
                if np.sum(recent_performance) > 0:
                    # Normalize performance to sum to 1
                    self.weights = recent_performance / np.sum(recent_performance)
                    logger.debug(f"Dynamic weights updated: {self.weights}")
            
            # Weighted average with dynamic weights
            ensemble_signal = np.zeros_like(signals[0])
            for i, signal in enumerate(signals):
                ensemble_signal += signal * self.weights[i]
        
        else:
            logger.warning(f"Unknown ensemble method: {self.ensemble_method}. Using weighted average.")
            # Fall back to weighted average
            ensemble_signal = np.zeros_like(signals[0])
            for i, signal in enumerate(signals):
                ensemble_signal += signal * self.weights[i]
        
        # Store signals in context
        if context is not None:
            if 'ensemble' not in context:
                context['ensemble'] = {}
            context['ensemble']['signals'] = signals
            context['ensemble']['weights'] = self.weights
            context['ensemble']['ensemble_signal'] = ensemble_signal
        
        return ensemble_signal
    
    def update_performance(self, data: pd.DataFrame, context: Dict[str, Any]):
        """
        Update performance history for dynamic weighting.
        
        Args:
            data: DataFrame with OHLCV data
            context: Strategy context dictionary
        """
        if 'ensemble' not in context or 'signals' not in context['ensemble']:
            logger.warning("No ensemble signals in context. Cannot update performance.")
            return
        
        # Get signals from context
        signals = context['ensemble']['signals']
        
        # Calculate performance for each strategy
        performance = []
        for i, signal in enumerate(signals):
            # Calculate returns
            returns = data['close'].pct_change()
            
            # Shift signal to align with returns (signal today affects tomorrow's return)
            aligned_signal = np.roll(signal, 1)
            aligned_signal[0] = 0  # No signal for first day
            
            # Calculate strategy return
            strategy_return = np.sum(aligned_signal * returns)
            
            # Store performance
            performance.append(strategy_return)
        
        # Add to performance history
        self.performance_history.append(performance)
        
        # Update weights if using performance-based adaptation
        if self.adaptation_method == 'performance':
            self._adapt_weights_performance()
        elif self.adaptation_method == 'momentum':
            self._adapt_weights_momentum()
    
    def _adapt_weights_performance(self):
        """Adapt weights based on recent performance."""
        if len(self.performance_history) < self.performance_window:
            return
        
        # Calculate recent performance for each strategy
        recent_performance = np.zeros(self.num_strategies)
        for i in range(self.num_strategies):
            recent_performance[i] = np.mean([perf[i] for perf in self.performance_history[-self.performance_window:]])
        
        # Adjust weights based on performance
        if np.sum(recent_performance) > 0:
            # Normalize performance to sum to 1
            new_weights = recent_performance / np.sum(recent_performance)
            
            # Update weights with learning rate
            self.weights = (1 - self.learning_rate) * self.weights + self.learning_rate * new_weights
            
            # Normalize weights to sum to 1
            self.weights = self.weights / np.sum(self.weights)
            
            logger.debug(f"Weights updated based on performance: {self.weights}")
    
    def _adapt_weights_momentum(self):
        """Adapt weights using momentum method."""
        if len(self.performance_history) < 2:
            return
        
        # Get recent and previous performance
        recent_perf = self.performance_history[-1]
        prev_perf = self.performance_history[-2]
        
        # Calculate momentum
        momentum = np.array([recent_perf[i] - prev_perf[i] for i in range(self.num_strategies)])
        
        # Update weights based on momentum
        momentum_adj = np.exp(momentum) / np.sum(np.exp(momentum))
        self.weights = self.momentum * self.weights + (1 - self.momentum) * momentum_adj
        
        # Normalize weights to sum to 1
        self.weights = self.weights / np.sum(self.weights)
        
        logger.debug(f"Weights updated using momentum: {self.weights}")


def ensemble_signal_generator(data: pd.DataFrame, context: Dict[str, Any],
                              models: Optional[List[Any]] = None,
                              model_weights: Optional[List[float]] = None,
                              ensemble_method: str = 'weighted_average') -> np.ndarray:
    """
    Generate trading signals using an ensemble of ML models.
    
    Args:
        data: DataFrame with OHLCV data
        context: Strategy context dictionary
        models: List of model objects
        model_weights: List of weights for each model
        ensemble_method: Method to combine models
        
    Returns:
        Array of trading signals
    """
    from .features import prepare_ml_features, normalize_features
    from .models import create_signal_from_predictions
    
    # Check if models are provided or in context
    if models is None:
        if 'ensemble_models' in context:
            models = context['ensemble_models']
        else:
            logger.warning("No models provided or found in context. Cannot generate signals.")
            return np.zeros(len(data))
    
    # Create ensemble if not in context
    if 'ml_ensemble' not in context:
        ensemble = EnsembleModel(
            models=models,
            weights=model_weights,
            ensemble_method=ensemble_method
        )
        context['ml_ensemble'] = ensemble
    else:
        ensemble = context['ml_ensemble']
    
    # Prepare features
    features = prepare_ml_features(data, context)
    normalized_features = normalize_features(features, context)
    
    # Generate predictions
    predictions = ensemble.predict(normalized_features)
    
    # Convert predictions to signals
    signals = create_signal_from_predictions(predictions, 0.5, len(data), normalized_features.index)
    
    return signals
