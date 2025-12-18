"""
Model Loader Utility

This module provides functions to load pre-trained models and their associated scalers.
Falls back to training if no pre-trained model is found.
"""

import os
import json
import joblib
from pathlib import Path
from datetime import datetime, timedelta


# Path to trained models directory
TRAINED_MODELS_DIR = Path(__file__).parent / "trained_models"


class ModelNotFoundError(Exception):
    """Raised when a requested model is not found."""
    pass


def get_model_path(ticker, model_type):
    """Get the file path for a saved model."""
    ticker_dir = TRAINED_MODELS_DIR / ticker.upper()
    return ticker_dir / f"{model_type}.pkl"


def get_scaler_path(ticker, model_type):
    """Get the file path for a saved scaler."""
    ticker_dir = TRAINED_MODELS_DIR / ticker.upper()
    return ticker_dir / f"{model_type}_scaler.pkl"


def get_metadata_path(ticker, model_type):
    """Get the file path for model metadata."""
    ticker_dir = TRAINED_MODELS_DIR / ticker.upper()
    return ticker_dir / f"{model_type}_metadata.json"


def load_model_metadata(ticker, model_type):
    """
    Load model metadata.

    Returns:
        dict: Metadata dictionary or None if not found
    """
    metadata_path = get_metadata_path(ticker, model_type)

    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        print(f"Warning: Could not load metadata: {e}")
        return None


def is_model_stale(ticker, model_type, max_age_days=30):
    """
    Check if a model is stale (too old).

    Args:
        ticker: Stock ticker symbol
        model_type: Type of model ("linear_regression" or "random_forest")
        max_age_days: Maximum age in days before model is considered stale

    Returns:
        bool: True if model is stale or metadata not found
    """
    metadata = load_model_metadata(ticker, model_type)

    if metadata is None:
        return True

    try:
        trained_at = datetime.fromisoformat(metadata['trained_at'])
        age = datetime.now() - trained_at
        return age.days > max_age_days
    except Exception:
        return True


def load_model(ticker, model_type):
    """
    Load a pre-trained model and its scaler.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        model_type: Type of model ("linear_regression" or "random_forest")

    Returns:
        dict: Dictionary containing:
            - model: The loaded model
            - scaler: The loaded scaler
            - metadata: Model metadata
            - loaded: True if successfully loaded

    Raises:
        ModelNotFoundError: If model files are not found
    """
    ticker = ticker.upper()
    model_path = get_model_path(ticker, model_type)
    scaler_path = get_scaler_path(ticker, model_type)

    # Check if model exists
    if not model_path.exists():
        raise ModelNotFoundError(
            f"No trained model found for {ticker} ({model_type}).\n"
            f"Expected path: {model_path}\n"
            f"Run: python train_models.py {ticker}"
        )

    if not scaler_path.exists():
        raise ModelNotFoundError(
            f"No scaler found for {ticker} ({model_type}).\n"
            f"Expected path: {scaler_path}\n"
            f"Run: python train_models.py {ticker}"
        )

    try:
        # Load model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # Load metadata
        metadata = load_model_metadata(ticker, model_type)

        # Check if model is stale
        if is_model_stale(ticker, model_type):
            trained_at = metadata.get('trained_at', 'unknown') if metadata else 'unknown'
            print(f"Warning: Model for {ticker} is more than 30 days old (trained: {trained_at})")
            print(f"Consider retraining: python train_models.py --retrain {ticker}")

        print(f"Loaded {model_type} model for {ticker}")
        if metadata:
            print(f"  Trained: {metadata.get('trained_at', 'N/A')}")
            print(f"  R² Score: {metadata.get('metrics', {}).get('r2_score', 'N/A'):.4f}")
            print(f"  MAE: ${metadata.get('metrics', {}).get('mae', 'N/A'):.2f}")

        return {
            'model': model,
            'scaler': scaler,
            'metadata': metadata,
            'loaded': True
        }

    except Exception as e:
        raise ModelNotFoundError(
            f"Error loading model for {ticker} ({model_type}): {str(e)}\n"
            f"Try retraining: python train_models.py --retrain {ticker}"
        )


def load_or_train_model(ticker, model_type, train_function, df=None):
    """
    Load a pre-trained model, or train a new one if not found.

    Args:
        ticker: Stock ticker symbol
        model_type: Type of model ("linear_regression" or "random_forest")
        train_function: Function to call if training is needed
        df: DataFrame with data (optional, will be fetched if not provided)

    Returns:
        dict: Dictionary containing model, scaler, and metadata
    """
    try:
        # Try to load pre-trained model
        result = load_model(ticker, model_type)
        result['trained_now'] = False
        return result

    except ModelNotFoundError as e:
        # Model not found, train a new one
        print(f"Pre-trained model not found for {ticker}")
        print(f"Training new {model_type} model...")

        if df is None:
            # Fetch data if not provided
            if model_type == "linear_regression":
                from linear_reg.fetch_data import fetch_stock_data
            else:
                from random_forest.fetch_data import fetch_stock_data

            df = fetch_stock_data(ticker, period="2y")

        # Train model
        train_result = train_function(df)

        if train_result is None:
            raise Exception(f"Training failed for {ticker}")

        print(f"Model trained successfully (this is a one-time training)")
        print(f"To avoid retraining in the future, run: python train_models.py {ticker}")

        return {
            'model': train_result['model'],
            'scaler': train_result['scaler'],
            'metadata': {
                'r2_score': train_result.get('r2_score', 0),
                'mae': train_result.get('mae', 0),
                'rmse': train_result.get('rmse', 0)
            },
            'loaded': False,
            'trained_now': True
        }


def list_available_models():
    """
    List all available pre-trained models.

    Returns:
        dict: Dictionary mapping tickers to lists of available model types
    """
    if not TRAINED_MODELS_DIR.exists():
        return {}

    available = {}

    for ticker_dir in TRAINED_MODELS_DIR.iterdir():
        if ticker_dir.is_dir():
            ticker = ticker_dir.name
            models = []

            if (ticker_dir / "linear_regression.pkl").exists():
                models.append("linear_regression")

            if (ticker_dir / "random_forest.pkl").exists():
                models.append("random_forest")

            if models:
                available[ticker] = models

    return available


def get_model_info(ticker, model_type):
    """
    Get information about a saved model without loading it.

    Args:
        ticker: Stock ticker symbol
        model_type: Type of model

    Returns:
        dict: Model information or None if not found
    """
    ticker = ticker.upper()
    model_path = get_model_path(ticker, model_type)

    if not model_path.exists():
        return None

    metadata = load_model_metadata(ticker, model_type)

    info = {
        'ticker': ticker,
        'model_type': model_type,
        'model_path': str(model_path),
        'exists': True,
        'stale': is_model_stale(ticker, model_type)
    }

    if metadata:
        info.update({
            'trained_at': metadata.get('trained_at'),
            'metrics': metadata.get('metrics'),
            'feature_count': metadata.get('feature_count'),
            'training_samples': metadata.get('training_samples')
        })

    return info


if __name__ == "__main__":
    # Demo: List all available models
    print("Available Pre-trained Models:")
    print("=" * 60)

    available = list_available_models()

    if not available:
        print("No pre-trained models found.")
        print("\nTo train models, run:")
        print("  python train_models.py AAPL")
        print("  python train_models.py --popular")
    else:
        for ticker, models in available.items():
            print(f"\n{ticker}:")
            for model_type in models:
                info = get_model_info(ticker, model_type)
                if info:
                    status = "STALE" if info['stale'] else "FRESH"
                    print(f"  [{status}] {model_type}")
                    if info.get('metrics'):
                        print(f"         R²: {info['metrics'].get('r2_score', 'N/A'):.4f}")
                        print(f"        MAE: ${info['metrics'].get('mae', 'N/A'):.2f}")
                    if info.get('trained_at'):
                        print(f"    Trained: {info['trained_at']}")
