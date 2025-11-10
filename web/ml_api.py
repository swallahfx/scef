"""
Web API integration for ML-based trading strategies.
This module extends the SCEF web interface to support machine learning components.
"""


import os
import json
import uuid
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from fastapi import APIRouter, Request, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import ML components
from src.ml.features import prepare_ml_features, normalize_features
from src.ml.models import MLModel, ml_signal_generator
from src.ml.reinforcement import rl_signal_generator
from src.ml.ensemble import ensemble_signal_generator
from src.ml.integration import (
    create_ml_strategy, create_rl_strategy, 
    create_ensemble_strategy, register_ml_components
)

# Define API router
ml_router = APIRouter(prefix="/api/ml", tags=["ml"])

# Define storage paths
ML_MODELS_DIR = "storage/ml_models"
os.makedirs(ML_MODELS_DIR, exist_ok=True)

# In-memory storage
ml_models = {}
ml_features = {}


# API Models
class MLModelConfig(BaseModel):
    model_type: str
    model_name: str
    model_params: Optional[Dict[str, Any]] = None
    prediction_horizon: int = 1


class TrainModelRequest(BaseModel):
    model_id: Optional[str] = None
    model_config_: MLModelConfig
    data_id: str
    train_test_split: float = 0.7
    feature_params: Optional[Dict[str, Any]] = None


class EvaluateModelRequest(BaseModel):
    model_id: str
    data_id: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class CreateMLStrategyRequest(BaseModel):
    name: str
    model_id: str
    position_sizing: float = 1.0
    risk_control_params: Optional[Dict[str, Any]] = None


class CreateRLStrategyRequest(BaseModel):
    name: str
    window_size: int = 20
    action_space: int = 3
    model_path: Optional[str] = None
    position_sizing: float = 1.0
    risk_control_params: Optional[Dict[str, Any]] = None


class CreateEnsembleStrategyRequest(BaseModel):
    name: str
    strategy_ids: List[str]
    weights: Optional[List[float]] = None
    ensemble_method: str = "weighted_average"
    adaptation_method: Optional[str] = None


# Helper functions
def get_market_data(data_id: str) -> pd.DataFrame:
    """Get market data from storage."""
    from web.app import market_data
    
    if data_id not in market_data:
        raise HTTPException(status_code=404, detail=f"Market data with ID {data_id} not found")
    
    return market_data[data_id]["data"]


def get_strategy(strategy_id: str) -> Any:
    """Get strategy from storage."""
    from web.app import strategies
    
    if strategy_id not in strategies:
        raise HTTPException(status_code=404, detail=f"Strategy with ID {strategy_id} not found")
    
    return strategies[strategy_id]["strategy_obj"]


# API endpoints

@ml_router.post("/models", response_class=JSONResponse)
async def create_model(model_config: MLModelConfig):
    """Create a new ML model."""
    try:
        # Generate ID for model
        model_id = str(uuid.uuid4())
        
        # Create model
        model = MLModel(
            model_type=model_config.model_type,
            model_name=model_config.model_name,
            model_params=model_config.model_params or {},
            prediction_horizon=model_config.prediction_horizon
        )
        
        # Store model
        ml_models[model_id] = {
            "id": model_id,
            "config": model_config.dict(),
            "model": model,
            "is_trained": False,
            "created_at": pd.Timestamp.now().isoformat()
        }
        
        return {
            "status": "success",
            "message": "Model created successfully",
            "model_id": model_id
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error creating model: {str(e)}"
        }


@ml_router.get("/models", response_class=JSONResponse)
async def get_models():
    """Get all ML models."""
    try:
        models = []
        
        for model_id, model_info in ml_models.items():
            models.append({
                "id": model_id,
                "model_type": model_info["config"]["model_type"],
                "model_name": model_info["config"]["model_name"],
                "is_trained": model_info["is_trained"],
                "created_at": model_info["created_at"]
            })
        
        return models
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting models: {str(e)}"
        }


@ml_router.get("/models/{model_id}", response_class=JSONResponse)
async def get_model(model_id: str):
    """Get ML model by ID."""
    try:
        if model_id not in ml_models:
            return {
                "status": "error",
                "message": f"Model with ID {model_id} not found"
            }
        
        model_info = ml_models[model_id]
        
        return {
            "id": model_id,
            "config": model_info["config"],
            "is_trained": model_info["is_trained"],
            "created_at": model_info["created_at"]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting model: {str(e)}"
        }


@ml_router.post("/models/train", response_class=JSONResponse)
async def train_model(request: TrainModelRequest):
    """Train an ML model."""
    try:
        # Get model
        model_id = request.model_id
        
        if model_id is not None and model_id not in ml_models:
            return {
                "status": "error",
                "message": f"Model with ID {model_id} not found"
            }
        
        if model_id is None:
            # Create a new model
            model_id = str(uuid.uuid4())
            model = MLModel(
                model_type=request.model_config.model_type,
                model_name=request.model_config.model_name,
                model_params=request.model_config.model_params or {},
                prediction_horizon=request.model_config.prediction_horizon
            )
            
            ml_models[model_id] = {
                "id": model_id,
                "config": request.model_config.dict(),
                "model": model,
                "is_trained": False,
                "created_at": pd.Timestamp.now().isoformat()
            }
        
        model_info = ml_models[model_id]
        model = model_info["model"]
        
        # Get market data
        data = get_market_data(request.data_id)
        
        # Prepare features
        context = {}
        
        features = prepare_ml_features(
            data, 
            context,
            windows=[5, 10, 20, 50] if request.feature_params is None else request.feature_params.get("windows", [5, 10, 20, 50]),
            include_time_features=True if request.feature_params is None else request.feature_params.get("include_time_features", True),
            lookback=1 if request.feature_params is None else request.feature_params.get("lookback", 1)
        )
        
        normalized_features = normalize_features(
            features, 
            context,
            method="z_score" if request.feature_params is None else request.feature_params.get("normalization_method", "z_score")
        )
        
        # Split data for training
        split_idx = int(len(data) * request.train_test_split)
        train_data = data.iloc[:split_idx]
        train_features = normalized_features.iloc[:split_idx]
        
        # Train model
        model.fit(train_features, train_data)
        
        # Update model info
        model_info["is_trained"] = True
        model_info["trained_at"] = pd.Timestamp.now().isoformat()
        model_info["training_data_id"] = request.data_id
        
        # Save features (for demonstration)
        feature_id = str(uuid.uuid4())
        ml_features[feature_id] = {
            "id": feature_id,
            "model_id": model_id,
            "data_id": request.data_id,
            "features": features,
            "normalized_features": normalized_features,
            "created_at": pd.Timestamp.now().isoformat()
        }
        
        # Evaluate model
        test_data = data.iloc[split_idx:]
        test_features = normalized_features.iloc[split_idx:]
        
        metrics = model.evaluate(test_features, test_data)
        
        return {
            "status": "success",
            "message": "Model trained successfully",
            "model_id": model_id,
            "feature_id": feature_id,
            "metrics": metrics,
            "train_size": len(train_data),
            "test_size": len(test_data)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        return {
            "status": "error",
            "message": f"Error training model: {str(e)}"
        }


@ml_router.post("/models/{model_id}/evaluate", response_class=JSONResponse)
async def evaluate_model(model_id: str, request: EvaluateModelRequest):
    """Evaluate an ML model."""
    try:
        # Get model
        if model_id not in ml_models:
            return {
                "status": "error",
                "message": f"Model with ID {model_id} not found"
            }
        
        model_info = ml_models[model_id]
        model = model_info["model"]
        
        # Check if model is trained
        if not model_info["is_trained"]:
            return {
                "status": "error",
                "message": "Model is not trained"
            }
        
        # Get market data
        data = get_market_data(request.data_id)
        
        # Filter data by date range if provided
        if request.start_date is not None:
            start_date = pd.Timestamp(request.start_date)
            data = data[data.index >= start_date]
        
        if request.end_date is not None:
            end_date = pd.Timestamp(request.end_date)
            data = data[data.index <= end_date]
        
        # Prepare features
        context = {}
        features = prepare_ml_features(data, context)
        normalized_features = normalize_features(features, context)
        
        # Evaluate model
        metrics = model.evaluate(normalized_features, data)
        
        # Generate predictions
        predictions = model.predict(normalized_features)
        
        # Create signals from predictions
        signals = np.zeros(len(data))
        if len(normalized_features) > 0:
            signal_values = predictions > 0.5 if model_info["config"]["model_type"] == "classifier" else predictions
            signals[normalized_features.index] = signal_values
        
        return {
            "status": "success",
            "metrics": metrics,
            "predictions": predictions.tolist() if len(predictions) > 0 else [],
            "signals": signals.tolist()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error evaluating model: {str(e)}"
        }


@ml_router.post("/models/{model_id}/save", response_class=JSONResponse)
async def save_model(model_id: str):
    """Save an ML model to disk."""
    try:
        # Get model
        if model_id not in ml_models:
            return {
                "status": "error",
                "message": f"Model with ID {model_id} not found"
            }
        
        model_info = ml_models[model_id]
        model = model_info["model"]
        
        # Check if model is trained
        if not model_info["is_trained"]:
            return {
                "status": "error",
                "message": "Model is not trained"
            }
        
        # Save model
        filepath = os.path.join(ML_MODELS_DIR, f"{model_id}.joblib")
        model.save(filepath)
        
        return {
            "status": "success",
            "message": "Model saved successfully",
            "filepath": filepath
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error saving model: {str(e)}"
        }


@ml_router.post("/models/{model_id}/load", response_class=JSONResponse)
async def load_model(model_id: str):
    """Load an ML model from disk."""
    try:
        # Check if model exists in storage
        filepath = os.path.join(ML_MODELS_DIR, f"{model_id}.joblib")
        if not os.path.exists(filepath):
            return {
                "status": "error",
                "message": f"Model file not found at {filepath}"
            }
        
        # Create model if not in memory
        if model_id not in ml_models:
            model = MLModel()
            
            ml_models[model_id] = {
                "id": model_id,
                "config": {
                    "model_type": "unknown",
                    "model_name": "unknown",
                    "prediction_horizon": 1
                },
                "model": model,
                "is_trained": False,
                "created_at": pd.Timestamp.now().isoformat()
            }
        
        model_info = ml_models[model_id]
        model = model_info["model"]
        
        # Load model
        model.load(filepath)
        
        # Update model info
        model_info["is_trained"] = True
        model_info["loaded_at"] = pd.Timestamp.now().isoformat()
        
        return {
            "status": "success",
            "message": "Model loaded successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error loading model: {str(e)}"
        }


@ml_router.post("/strategies/ml", response_class=JSONResponse)
async def create_ml_strategy_endpoint(request: CreateMLStrategyRequest):
    """Create an ML-based trading strategy."""
    try:
        from web.app import strategies
        
        # Get model
        if request.model_id not in ml_models:
            return {
                "status": "error",
                "message": f"Model with ID {request.model_id} not found"
            }
        
        model_info = ml_models[request.model_id]
        
        # Check if model is trained
        if not model_info["is_trained"]:
            return {
                "status": "error",
                "message": "Model is not trained"
            }
        
        # Create strategy
        strategy = create_ml_strategy(
            name=request.name,
            model_type=model_info["config"]["model_type"],
            model_name=model_info["config"]["model_name"],
            model_params=model_info["config"].get("model_params", {}),
            prediction_horizon=model_info["config"]["prediction_horizon"],
            position_sizing=request.position_sizing,
            risk_control_params=request.risk_control_params
        )
        
        # Add model to strategy context
        context = {"ml_model": model_info["model"]}
        
        # Generate ID for strategy
        strategy_id = str(uuid.uuid4())
        
        # Store strategy
        strategies[strategy_id] = {
            "id": strategy_id,
            "name": request.name,
            "strategy_obj": strategy,
            "context": context,
            "created_at": pd.Timestamp.now().isoformat(),
            "type": "ml",
            "model_id": request.model_id
        }
        
        return {
            "status": "success",
            "message": "Strategy created successfully",
            "strategy_id": strategy_id
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error creating strategy: {str(e)}"
        }


@ml_router.post("/strategies/rl", response_class=JSONResponse)
async def create_rl_strategy_endpoint(request: CreateRLStrategyRequest):
    """Create a reinforcement learning-based trading strategy."""
    try:
        from web.app import strategies
        
        # Create strategy
        strategy = create_rl_strategy(
            name=request.name,
            window_size=request.window_size,
            action_space=request.action_space,
            model_path=request.model_path,
            position_sizing=request.position_sizing,
            risk_control_params=request.risk_control_params
        )
        
        # Generate ID for strategy
        strategy_id = str(uuid.uuid4())
        
        # Store strategy
        strategies[strategy_id] = {
            "id": strategy_id,
            "name": request.name,
            "strategy_obj": strategy,
            "context": {},
            "created_at": pd.Timestamp.now().isoformat(),
            "type": "rl",
            "window_size": request.window_size,
            "action_space": request.action_space
        }
        
        return {
            "status": "success",
            "message": "Strategy created successfully",
            "strategy_id": strategy_id
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error creating strategy: {str(e)}"
        }


@ml_router.post("/strategies/ensemble", response_class=JSONResponse)
async def create_ensemble_strategy_endpoint(request: CreateEnsembleStrategyRequest):
    """Create an ensemble trading strategy."""
    try:
        from web.app import strategies
        
        # Get component strategies
        component_strategies = []
        
        for strategy_id in request.strategy_ids:
            if strategy_id not in strategies:
                return {
                    "status": "error",
                    "message": f"Strategy with ID {strategy_id} not found"
                }
            
            component_strategies.append(strategies[strategy_id]["strategy_obj"])
        
        # Create strategy
        strategy = create_ensemble_strategy(
            name=request.name,
            strategies=component_strategies,
            weights=request.weights,
            ensemble_method=request.ensemble_method,
            adaptation_method=request.adaptation_method
        )
        
        # Generate ID for strategy
        strategy_id = str(uuid.uuid4())
        
        # Store strategy
        strategies[strategy_id] = {
            "id": strategy_id,
            "name": request.name,
            "strategy_obj": strategy,
            "context": {
                "ensemble": {
                    "strategy_ids": request.strategy_ids,
                    "weights": request.weights,
                    "ensemble_method": request.ensemble_method,
                    "adaptation_method": request.adaptation_method
                }
            },
            "created_at": pd.Timestamp.now().isoformat(),
            "type": "ensemble",
            "component_strategy_ids": request.strategy_ids
        }
        
        return {
            "status": "success",
            "message": "Strategy created successfully",
            "strategy_id": strategy_id
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error creating strategy: {str(e)}"
        }


@ml_router.get("/features/{feature_id}", response_class=JSONResponse)
async def get_features(feature_id: str):
    """Get feature data by ID."""
    try:
        if feature_id not in ml_features:
            return {
                "status": "error",
                "message": f"Features with ID {feature_id} not found"
            }
        
        feature_info = ml_features[feature_id]
        
        # Convert to lists for JSON serialization
        features_sample = feature_info["features"].head(10).to_dict()
        normalized_sample = feature_info["normalized_features"].head(10).to_dict()
        
        return {
            "id": feature_id,
            "model_id": feature_info["model_id"],
            "data_id": feature_info["data_id"],
            "features_sample": features_sample,
            "normalized_sample": normalized_sample,
            "num_features": len(feature_info["features"].columns),
            "num_samples": len(feature_info["features"]),
            "created_at": feature_info["created_at"]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting features: {str(e)}"
        }


# Initialize ML components when app starts
def initialize_ml_components():
    """Register ML components with the SCEF framework."""
    try:
        register_ml_components()
        print("ML components registered successfully")
    except Exception as e:
        print(f"Error registering ML components: {str(e)}")


# Function to include in app startup
def include_ml_router(app):
    """Include ML router in the FastAPI app."""
    app.include_router(ml_router)
    app.add_event_handler("startup", initialize_ml_components)
