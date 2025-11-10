"""
Technical indicators library with efficient numerical implementations.

This module provides optimized implementations of common technical indicators
used in trading strategies.
"""
import numpy as np
import pandas as pd
from numba import jit
from typing import Optional, Tuple, Dict, Any


@jit(nopython=True)
def _sma_numba(array: np.ndarray, window: int) -> np.ndarray:
    """Numba-optimized Simple Moving Average."""
    n = len(array)
    result = np.full(n, np.nan)
    
    if n < window:
        return result
        
    # Compute the first average
    sum_val = 0.0
    for i in range(window):
        sum_val += array[i]
    
    result[window-1] = sum_val / window
    
    # Update moving average for remaining elements
    for i in range(window, n):
        sum_val = sum_val + array[i] - array[i-window]
        result[i] = sum_val / window
        
    return result


def sma(data: pd.DataFrame, context: Dict[str, Any], column: str = 'close', window: int = 20) -> np.ndarray:
    """Simple Moving Average."""
    if isinstance(data, pd.DataFrame):
        array = data[column].values
    else:
        array = data
        
    return _sma_numba(array, window)


@jit(nopython=True)
def _ema_numba(array: np.ndarray, window: int) -> np.ndarray:
    """Numba-optimized Exponential Moving Average."""
    n = len(array)
    result = np.full(n, np.nan)
    
    if n < window:
        return result
        
    # Start with SMA for first window elements
    sum_val = 0.0
    for i in range(window):
        sum_val += array[i]
    
    result[window-1] = sum_val / window
    
    # Calculate EMA for remaining elements
    alpha = 2.0 / (window + 1)
    for i in range(window, n):
        result[i] = array[i] * alpha + result[i-1] * (1 - alpha)
        
    return result


def ema(data: pd.DataFrame, context: Dict[str, Any], column: str = 'close', window: int = 20) -> np.ndarray:
    """Exponential Moving Average."""
    if isinstance(data, pd.DataFrame):
        array = data[column].values
    else:
        array = data
        
    return _ema_numba(array, window)


@jit(nopython=True)
def _macd_numba(array: np.ndarray, fast_window: int = 12, slow_window: int = 26, signal_window: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numba-optimized MACD implementation."""
    n = len(array)
    macd_line = np.full(n, np.nan)
    signal_line = np.full(n, np.nan)
    histogram = np.full(n, np.nan)
    
    if n < slow_window:
        return macd_line, signal_line, histogram
    
    # Calculate fast EMA
    fast_ema = _ema_numba(array, fast_window)
    
    # Calculate slow EMA
    slow_ema = _ema_numba(array, slow_window)
    
    # Calculate MACD line
    for i in range(slow_window-1, n):
        macd_line[i] = fast_ema[i] - slow_ema[i]
    
    # Calculate signal line (EMA of MACD line)
    # First need to handle NANs
    macd_for_signal = np.copy(macd_line)
    start_idx = slow_window-1
    macd_for_signal[:start_idx] = macd_for_signal[start_idx]  # Forward fill
    
    signal_line = _ema_numba(macd_for_signal, signal_window)
    
    # Calculate histogram
    for i in range(slow_window + signal_window - 2, n):
        histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram


def macd(data: pd.DataFrame, context: Dict[str, Any], column: str = 'close', 
         fast_window: int = 12, slow_window: int = 26, signal_window: int = 9) -> Dict[str, np.ndarray]:
    """Moving Average Convergence Divergence (MACD)."""
    if isinstance(data, pd.DataFrame):
        array = data[column].values
    else:
        array = data
        
    macd_line, signal_line, histogram = _macd_numba(array, fast_window, slow_window, signal_window)
    
    return {
        'macd_line': macd_line,
        'signal_line': signal_line,
        'histogram': histogram
    }


@jit(nopython=True)
def _rsi_numba(array: np.ndarray, window: int = 14) -> np.ndarray:
    """Numba-optimized Relative Strength Index implementation."""
    n = len(array)
    rsi = np.full(n, np.nan)
    
    if n < window + 1:
        return rsi
    
    # Calculate price changes
    changes = np.zeros(n)
    for i in range(1, n):
        changes[i] = array[i] - array[i-1]
    
    # Separate gains and losses
    gains = np.zeros(n)
    losses = np.zeros(n)
    
    for i in range(1, n):
        if changes[i] > 0:
            gains[i] = changes[i]
            losses[i] = 0
        else:
            gains[i] = 0
            losses[i] = -changes[i]
    
    # Calculate initial averages
    avg_gain = 0
    avg_loss = 0
    
    for i in range(1, window+1):
        avg_gain += gains[i]
        avg_loss += losses[i]
    
    avg_gain /= window
    avg_loss /= window
    
    # Calculate first RSI
    if avg_loss == 0:
        rsi[window] = 100
    else:
        rs = avg_gain / avg_loss
        rsi[window] = 100 - (100 / (1 + rs))
    
    # Calculate remaining RSIs
    for i in range(window + 1, n):
        avg_gain = ((avg_gain * (window - 1)) + gains[i]) / window
        avg_loss = ((avg_loss * (window - 1)) + losses[i]) / window
        
        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def rsi(data: pd.DataFrame, context: Dict[str, Any], column: str = 'close', window: int = 14) -> np.ndarray:
    """Relative Strength Index."""
    if isinstance(data, pd.DataFrame):
        array = data[column].values
    else:
        array = data
        
    return _rsi_numba(array, window)


@jit(nopython=True)
def _bollinger_bands_numba(array: np.ndarray, window: int = 20, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numba-optimized Bollinger Bands implementation."""
    n = len(array)
    middle_band = np.full(n, np.nan)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    
    if n < window:
        return middle_band, upper_band, lower_band
    
    # Calculate middle band (SMA)
    middle_band = _sma_numba(array, window)
    
    # Calculate standard deviation
    std = np.full(n, np.nan)
    for i in range(window-1, n):
        sum_squared_diff = 0
        for j in range(i-(window-1), i+1):
            sum_squared_diff += (array[j] - middle_band[i]) ** 2
        std[i] = np.sqrt(sum_squared_diff / window)
    
    # Calculate upper and lower bands
    for i in range(window-1, n):
        upper_band[i] = middle_band[i] + (std[i] * num_std)
        lower_band[i] = middle_band[i] - (std[i] * num_std)
    
    return middle_band, upper_band, lower_band


def bollinger_bands(data: pd.DataFrame, context: Dict[str, Any], column: str = 'close', 
                    window: int = 20, num_std: float = 2.0) -> Dict[str, np.ndarray]:
    """Bollinger Bands."""
    if isinstance(data, pd.DataFrame):
        array = data[column].values
    else:
        array = data
        
    middle, upper, lower = _bollinger_bands_numba(array, window, num_std)
    
    return {
        'middle': middle,
        'upper': upper,
        'lower': lower
    }


@jit(nopython=True)
def _atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
    """Numba-optimized Average True Range implementation."""
    n = len(high)
    tr = np.full(n, np.nan)
    atr = np.full(n, np.nan)
    
    if n < 2:
        return atr
    
    # Calculate True Range for the first bar
    tr[0] = high[0] - low[0]
    
    # Calculate True Range for remaining bars
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Calculate first ATR
    sum_tr = 0
    for i in range(min(window, n)):
        sum_tr += tr[i]
    
    atr[window-1] = sum_tr / window
    
    # Calculate remaining ATRs using Wilder's smoothing method
    for i in range(window, n):
        atr[i] = ((atr[i-1] * (window - 1)) + tr[i]) / window
    
    return atr


def atr(data: pd.DataFrame, context: Dict[str, Any], window: int = 14) -> np.ndarray:
    """Average True Range."""
    if isinstance(data, pd.DataFrame):
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
    else:
        raise ValueError("ATR requires DataFrame with high, low, close columns")
        
    return _atr_numba(high, low, close, window)
