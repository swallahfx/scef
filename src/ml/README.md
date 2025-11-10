# SCEF Machine Learning Module

This module provides machine learning components for the Strategy Composition and Evaluation Framework (SCEF). It allows you to create, backtest, and deploy trading strategies powered by machine learning algorithms.

## Features

- **Feature Engineering**: Extract meaningful features from market data for ML models
- **Machine Learning Models**: Implement classification and regression models for signal generation
- **Reinforcement Learning**: Train agents to make trading decisions based on market state
- **Ensemble Strategies**: Combine multiple models or strategies for improved performance
- **Seamless Integration**: Works with the existing SCEF framework for backtesting and deployment

## Getting Started

### Prerequisites

In addition to the base SCEF requirements, you'll need these packages for ML functionality:

```bash
pip install scikit-learn numpy pandas matplotlib tensorflow keras lightgbm
```

### Basic Usage

Here's a quick example of creating and testing an ML-based strategy:

```python
from src.ml.integration import create_ml_strategy
from src.backtester import Backtester, BacktestConfig

# Create an ML strategy
strategy = create_ml_strategy(
    name="Random Forest Strategy",
    model_type="classifier",
    model_name="random_forest",
    prediction_horizon=1,
    position_sizing=1.0
)

# Create backtester
backtester = Backtester(BacktestConfig())

# Run backtest
result = backtester.backtest(strategy, data)

# Analyze results
print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
```

## Module Structure

- **features.py**: Functions for feature engineering and data preparation
- **models.py**: Implementation of ML models for trading signals
- **reinforcement.py**: Reinforcement learning components
- **ensemble.py**: Tools for combining multiple models or strategies
- **integration.py**: Integration with the SCEF framework
- **demo.py**: Demonstration of ML-based trading strategies

## Creating ML Strategies

### Machine Learning Strategies

```python
from src.ml.integration import create_ml_strategy

# Create a random forest classifier strategy
ml_strategy = create_ml_strategy(
    name="ML Strategy",
    model_type="classifier",  # or "regressor"
    model_name="random_forest",  # or "logistic_regression", "svm", etc.
    model_params={"n_estimators": 100, "random_state": 42},
    prediction_horizon=1,  # Number of periods ahead to predict
    position_sizing=1.0  # Maximum position size
)
```

### Reinforcement Learning Strategies

```python
from src.ml.integration import create_rl_strategy

# Create a reinforcement learning strategy
rl_strategy = create_rl_strategy(
    name="RL Strategy",
    window_size=20,  # Number of past periods to include in state
    action_space=3,  # Number of discrete actions (e.g., 3 for short/neutral/long)
    position_sizing=1.0  # Maximum position size
)
```

### Ensemble Strategies

```python
from src.ml.integration import create_ensemble_strategy

# Create an ensemble strategy
ensemble_strategy = create_ensemble_strategy(
    name="Ensemble Strategy",
    strategies=[strategy1, strategy2, strategy3],
    weights=[0.5, 0.3, 0.2],
    ensemble_method="weighted_average"  # or "voting", "dynamic"
)
```

## Demo

The `demo.py` script provides a complete example of creating and testing ML-based strategies. Run it to see the ML module in action:

```bash
cd scef
python -m src.ml.demo
```

## Advanced Usage

### Custom Feature Engineering

You can create custom features for your ML models:

```python
from src.ml.features import prepare_ml_features, normalize_features

# Custom feature engineering function
def create_custom_features(data, context=None):
    features = prepare_ml_features(data, context)
    
    # Add custom features
    features['custom_indicator'] = calculate_custom_indicator(data)
    
    # Normalize features
    return normalize_features(features, context)
```

### Model Persistence

Save and load trained models:

```python
from src.ml.models import MLModel

# Create and train model
model = MLModel(model_type="classifier", model_name="random_forest")
model.fit(features, data)

# Save model
model.save("models/my_model.joblib")

# Load model
loaded_model = MLModel(model_type="classifier", model_name="random_forest")
loaded_model.load("models/my_model.joblib")
```

## Integration with Web Interface

The ML module integrates with the SCEF web interface, allowing you to create and manage ML-based strategies through the browser. The web interface provides:

- ML strategy creation and configuration
- Feature visualization
- Model training and evaluation
- Performance comparison with traditional strategies

## Contributing

Contributions to the ML module are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
