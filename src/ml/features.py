"""
Feature engineering for machine learning-based trading strategies.
This module provides functions to extract features from market data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

def create_technical_features(data: pd.DataFrame, 
                              windows: List[int] = [5, 10, 20, 50, 100]) -> pd.DataFrame:
    """
    Create features based on technical indicators.
    
    Args:
        data: DataFrame with OHLCV data
        windows: List of lookback periods for indicators
    
    Returns:
        DataFrame with technical features
    """
    features = pd.DataFrame(index=data.index)
    price = data['close']
    
    # Price-based features
    features['returns_1d'] = price.pct_change(1)
    
    # Create features for each window
    for window in windows:
        # Moving averages
        features[f'sma_{window}'] = price.rolling(window).mean()
        features[f'ema_{window}'] = price.ewm(span=window, adjust=False).mean()
        
        # Relative position to moving average
        features[f'sma_dist_{window}'] = (price - features[f'sma_{window}']) / price
        features[f'ema_dist_{window}'] = (price - features[f'ema_{window}']) / price
        
        # Moving average crossovers
        if window > min(windows):
            min_window = min(windows)
            features[f'ma_cross_{min_window}_{window}'] = np.where(
                features[f'sma_{min_window}'] > features[f'sma_{window}'], 1, -1)
        
        # Volatility
        features[f'volatility_{window}'] = features['returns_1d'].rolling(window).std()
        
        # Momentum
        features[f'roc_{window}'] = price.pct_change(window)
        
        # Range features
        features[f'range_{window}'] = (data['high'].rolling(window).max() - 
                                      data['low'].rolling(window).min()) / price
        
        # Volume features
        if 'volume' in data.columns:
            features[f'volume_sma_{window}'] = data['volume'].rolling(window).mean()
            features[f'volume_ratio_{window}'] = data['volume'] / features[f'volume_sma_{window}']
    
    # RSI
    delta = price.diff()
    up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
    
    for window in windows:
        # Calculate RSI
        avg_gain = up.rolling(window).mean()
        avg_loss = down.rolling(window).mean()
        
        rs = avg_gain / avg_loss
        features[f'rsi_{window}'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    for window in windows:
        std = price.rolling(window).std()
        middle = features[f'sma_{window}']
        features[f'bb_upper_{window}'] = middle + 2 * std
        features[f'bb_lower_{window}'] = middle - 2 * std
        features[f'bb_width_{window}'] = (features[f'bb_upper_{window}'] - 
                                         features[f'bb_lower_{window}']) / middle
        features[f'bb_position_{window}'] = (price - features[f'bb_lower_{window}']) / (
            features[f'bb_upper_{window}'] - features[f'bb_lower_{window}'])
    
    # Drop NaN values
    return features


def create_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar-based features.
    
    Args:
        data: DataFrame with datetime index
    
    Returns:
        DataFrame with time features
    """
    features = pd.DataFrame(index=data.index)
    
    # Extract date components
    features['day_of_week'] = data.index.dayofweek
    features['day_of_month'] = data.index.day
    features['week_of_year'] = data.index.isocalendar().week
    features['month'] = data.index.month
    features['quarter'] = data.index.quarter
    features['year'] = data.index.year
    
    # Create cyclical features
    features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
    features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
    features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
    
    # Day type features
    features['is_month_start'] = data.index.is_month_start.astype(int)
    features['is_month_end'] = data.index.is_month_end.astype(int)
    features['is_quarter_start'] = data.index.is_quarter_start.astype(int)
    features['is_quarter_end'] = data.index.is_quarter_end.astype(int)
    
    return features


def prepare_ml_features(data: pd.DataFrame, 
                        context: Dict[str, Any], 
                        windows: List[int] = [5, 10, 20, 50, 100],
                        include_time_features: bool = True,
                        lookback: int = 1) -> pd.DataFrame:
    """
    Prepare features for machine learning models.
    
    Args:
        data: DataFrame with OHLCV data
        context: Strategy context dictionary
        windows: List of lookback periods for indicators
        include_time_features: Whether to include time-based features
        lookback: Number of lags to include for each feature
    
    Returns:
        DataFrame with all features
    """
    # Extract technical features
    tech_features = create_technical_features(data, windows)
    
    # Add time features if requested
    if include_time_features:
        time_features = create_time_features(data)
        features = pd.concat([tech_features, time_features], axis=1)
    else:
        features = tech_features
    
    # Add lagged features
    if lookback > 1:
        original_cols = features.columns.tolist()
        for lag in range(1, lookback):
            for col in original_cols:
                features[f'{col}_lag_{lag}'] = features[col].shift(lag)
    
    # Store in context if needed
    if context is not None:
        context['ml_features'] = features
    
    return features


def normalize_features(features: pd.DataFrame, 
                       context: Dict[str, Any],
                       method: str = 'z_score',
                       training_window: int = 252) -> pd.DataFrame:
    """
    Normalize features for machine learning models.
    
    Args:
        features: DataFrame with features
        context: Strategy context dictionary
        method: Normalization method ('z_score', 'min_max', or 'robust')
        training_window: Number of days to use for normalization parameters
    
    Returns:
        DataFrame with normalized features
    """
    normalized = pd.DataFrame(index=features.index)
    
    for col in features.columns:
        if features[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            # Get historical data for normalization
            historical = features[col].iloc[-training_window:] if len(features) > training_window else features[col]
            
            if method == 'z_score':
                mean = historical.mean()
                std = historical.std()
                if std > 0:
                    normalized[col] = (features[col] - mean) / std
                else:
                    normalized[col] = features[col] - mean
                    
            elif method == 'min_max':
                min_val = historical.min()
                max_val = historical.max()
                if max_val > min_val:
                    normalized[col] = (features[col] - min_val) / (max_val - min_val)
                else:
                    normalized[col] = features[col] - min_val
                    
            elif method == 'robust':
                q25 = historical.quantile(0.25)
                q75 = historical.quantile(0.75)
                iqr = q75 - q25
                if iqr > 0:
                    normalized[col] = (features[col] - q25) / iqr
                else:
                    normalized[col] = features[col] - q25
                    
            else:
                normalized[col] = features[col]
        else:
            normalized[col] = features[col]
    
    # Store normalization parameters in context
    if context is not None:
        if 'normalization' not in context:
            context['normalization'] = {}
        context['normalization'][method] = {
            'features': normalized,
            'params': {col: {
                'mean': features[col].mean(),
                'std': features[col].std(),
                'min': features[col].min(),
                'max': features[col].max(),
                'q25': features[col].quantile(0.25),
                'q75': features[col].quantile(0.75)
            } for col in features.columns if features[col].dtype in [np.float64, np.float32, np.int64, np.int32]}
        }
    
    return normalized
