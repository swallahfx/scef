"""
Machine learning models for trading strategies.
This module provides implementations of ML models for signal generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import os
import pickle
import joblib
import datetime
import logging

# Optional imports for ML libraries
try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.svm import SVC, SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MLModel:
    """Base class for machine learning models in trading strategies."""
    
    def __init__(self, model_type: str = 'classifier', model_name: str = 'random_forest',
                 model_params: Dict[str, Any] = None, prediction_horizon: int = 1):
        """
        Initialize the ML model.
        
        Args:
            model_type: Type of model ('classifier' or 'regressor')
            model_name: Name of the model ('random_forest', 'logistic_regression', etc.)
            model_params: Parameters for the model
            prediction_horizon: Number of periods ahead to predict
        """
        self.model_type = model_type
        self.model_name = model_name
        self.model_params = model_params or {}
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False
        
        # Check if required libraries are available
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. ML models won't work properly.")
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the machine learning model based on type and name."""
        if not SKLEARN_AVAILABLE:
            logger.error("Cannot initialize model: scikit-learn is not available.")
            return
        
        if self.model_type == 'classifier':
            if self.model_name == 'random_forest':
                self.model = RandomForestClassifier(**self.model_params)
            elif self.model_name == 'logistic_regression':
                self.model = LogisticRegression(**self.model_params)
            elif self.model_name == 'svm':
                self.model = SVC(**self.model_params)
            elif self.model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                self.model = lgb.LGBMClassifier(**self.model_params)
            else:
                logger.error(f"Unsupported classifier model: {self.model_name}")
                self.model = None
        
        elif self.model_type == 'regressor':
            if self.model_name == 'random_forest':
                self.model = RandomForestRegressor(**self.model_params)
            elif self.model_name == 'linear_regression':
                self.model = LinearRegression(**self.model_params)
            elif self.model_name == 'svm':
                self.model = SVR(**self.model_params)
            elif self.model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                self.model = lgb.LGBMRegressor(**self.model_params)
            else:
                logger.error(f"Unsupported regressor model: {self.model_name}")
                self.model = None
        
        else:
            logger.error(f"Unsupported model type: {self.model_type}")
            self.model = None
        
        # Initialize scaler
        self.scaler = StandardScaler()
    
    def prepare_data(self, features: pd.DataFrame, data: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training or prediction.
        
        Args:
            features: DataFrame with features
            data: DataFrame with OHLCV data (needed to create labels)
            
        Returns:
            X: Features array
            y: Labels array (or None if data is None)
        """
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Drop rows with NaN values
        features = features.dropna()
        
        # Scale features
        if self.scaler is not None:
            X = self.scaler.fit_transform(features)
        else:
            X = features.values
        
        if data is not None:
            # Create labels based on future returns
            if self.prediction_horizon > 0:
                future_returns = data['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
                future_returns = future_returns.loc[features.index]
                
                if self.model_type == 'classifier':
                    # Create binary labels (1 for positive returns, 0 for negative)
                    y = (future_returns > 0).astype(int).values
                else:
                    # Use raw returns for regression
                    y = future_returns.values
            else:
                logger.warning("prediction_horizon must be positive. Using 1.")
                future_returns = data['close'].pct_change(1).shift(-1)
                future_returns = future_returns.loc[features.index]
                
                if self.model_type == 'classifier':
                    y = (future_returns > 0).astype(int).values
                else:
                    y = future_returns.values
            
            return X, y
        
        return X, None
    
    def fit(self, features: pd.DataFrame, data: pd.DataFrame) -> None:
        """
        Train the model.
        
        Args:
            features: DataFrame with features
            data: DataFrame with OHLCV data (needed to create labels)
        """
        if self.model is None:
            logger.error("Model not initialized. Cannot fit.")
            return
        
        # Prepare data
        X, y = self.prepare_data(features, data)
        
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            logger.error("No valid data for training. Cannot fit.")
            return
        
        # Fit the model
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Log model info
        logger.info(f"Model trained successfully with {len(X)} samples.")
        
        # For classifiers, log class balance
        if self.model_type == 'classifier':
            class_counts = np.bincount(y)
            class_percentages = class_counts / len(y) * 100
            logger.info(f"Class distribution: {class_percentages[0]:.1f}% negative, {class_percentages[1]:.1f}% positive")
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            features: DataFrame with features
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Returning zeros.")
            return np.zeros(len(features))
        
        # Prepare data
        X, _ = self.prepare_data(features)
        
        if len(X) == 0:
            logger.warning("No valid data for prediction. Returning empty array.")
            return np.array([])
        
        # Generate predictions
        if self.model_type == 'classifier':
            # For classification, return the raw probability of positive class
            if hasattr(self.model, 'predict_proba'):
                predictions = self.model.predict_proba(X)[:, 1]
            else:
                predictions = self.model.predict(X)
        else:
            # For regression, return raw predictions
            predictions = self.model.predict(X)
        
        return predictions
    
    def evaluate(self, features: pd.DataFrame, data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            features: DataFrame with features
            data: DataFrame with OHLCV data (needed to create labels)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Cannot evaluate.")
            return {}
        
        # Prepare data
        X, y = self.prepare_data(features, data)
        
        if len(X) == 0 or len(y) == 0:
            logger.warning("No valid data for evaluation. Returning empty metrics.")
            return {}
        
        # Generate predictions
        if self.model_type == 'classifier':
            y_pred = self.model.predict(X)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1': f1_score(y, y_pred)
            }
        else:
            y_pred = self.model.predict(X)
            
            # Calculate metrics
            metrics = {
                'mse': mean_squared_error(y, y_pred),
                'mae': mean_absolute_error(y, y_pred),
                'r2': r2_score(y, y_pred)
            }
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Nothing to save.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save the model and metadata
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'model_name': self.model_name,
            'model_params': self.model_params,
            'prediction_horizon': self.prediction_horizon,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'saved_at': datetime.datetime.now().isoformat()
        }
        
        try:
            # Save using joblib for better support of scikit-learn objects
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load(self, filepath: str) -> None:
        """
        Load the model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        try:
            # Load using joblib
            model_data = joblib.load(filepath)
            
            # Load model attributes
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler')
            self.model_type = model_data.get('model_type', self.model_type)
            self.model_name = model_data.get('model_name', self.model_name)
            self.model_params = model_data.get('model_params', self.model_params)
            self.prediction_horizon = model_data.get('prediction_horizon', self.prediction_horizon)
            self.feature_names = model_data.get('feature_names', None)
            self.is_fitted = model_data.get('is_fitted', False)
            
            logger.info(f"Model loaded successfully from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")


def create_signal_from_predictions(predictions: np.ndarray, threshold: float = 0.5, 
                                   data_length: int = None, tail_idx: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert model predictions to trading signals.
    
    Args:
        predictions: Array of model predictions
        threshold: Threshold for converting probabilities to signals
        data_length: Length of the original data (for zero padding)
        tail_idx: Indices of the predictions in the original data
        
    Returns:
        Array of trading signals
    """
    if len(predictions) == 0:
        return np.array([])
    
    # Convert predictions to signals
    if np.issubdtype(predictions.dtype, np.number):
        # For probabilistic predictions
        signals = np.zeros_like(predictions)
        signals[predictions > threshold] = 1
        signals[predictions < (1 - threshold)] = -1
    else:
        # For categorical predictions
        signals = np.where(predictions == 1, 1, -1)
    
    # Map signals back to original data length if needed
    if data_length is not None and tail_idx is not None:
        full_signals = np.zeros(data_length)
        full_signals[tail_idx] = signals
        return full_signals
    
    return signals


def ml_signal_generator(data: pd.DataFrame, context: Dict[str, Any],
                        model_type: str = 'classifier', model_name: str = 'random_forest',
                        model_params: Dict[str, Any] = None, prediction_horizon: int = 1,
                        threshold: float = 0.5) -> np.ndarray:
    """
    Generate trading signals using machine learning models.
    
    Args:
        data: DataFrame with OHLCV data
        context: Strategy context dictionary
        model_type: Type of model ('classifier' or 'regressor')
        model_name: Name of the model ('random_forest', 'logistic_regression', etc.)
        model_params: Parameters for the model
        prediction_horizon: Number of periods ahead to predict
        threshold: Threshold for converting probabilities to signals
        
    Returns:
        Array of trading signals
    """
    from .features import prepare_ml_features, normalize_features
    
    # Initialize or retrieve model
    if 'ml_model' not in context:
        model_params = model_params or {
            'n_estimators': 100,
            'random_state': 42
        }
        context['ml_model'] = MLModel(
            model_type=model_type,
            model_name=model_name,
            model_params=model_params,
            prediction_horizon=prediction_horizon
        )
    
    model = context['ml_model']
    
    # Prepare features
    features = prepare_ml_features(data, context)
    normalized_features = normalize_features(features, context)
    
    # Check if we have enough data for training
    min_training_samples = 100  # Arbitrary threshold
    
    if len(normalized_features.dropna()) < min_training_samples:
        logger.warning(f"Not enough data for training ({len(normalized_features.dropna())} < {min_training_samples}). Returning zeros.")
        return np.zeros(len(data))
    
    # Train and predict based on current state
    if not model.is_fitted:
        # Use up to the last 10% of data for validation
        validation_start = int(len(normalized_features) * 0.9)
        train_features = normalized_features.iloc[:validation_start]
        
        # Train the model
        model.fit(train_features, data.iloc[:validation_start])
    
    # Get predictions for all data
    predictions = model.predict(normalized_features)
    
    # Convert predictions to signals
    signals = create_signal_from_predictions(predictions, threshold, len(data), normalized_features.index)
    
    return signals
