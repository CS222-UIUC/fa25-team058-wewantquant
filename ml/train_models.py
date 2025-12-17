"""
Training Script for Stock Prediction Models

This script trains both Linear Regression and Random Forest models on 5 years of historical data
and saves them as .pkl files for efficient reuse. This avoids retraining on every prediction.

Usage:
    python train_models.py AAPL                    # Train models for a single ticker
    python train_models.py AAPL GOOGL MSFT        # Train models for multiple tickers
    python train_models.py --popular               # Train on popular stocks
    python train_models.py --retrain AAPL         # Force retrain existing models
"""

import os
import sys
import json
import joblib
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from linear_reg.predict import train_linear_regression_model
from random_forest.predict import train_random_forest_model
from linear_reg.fetch_data import fetch_stock_data
from random_forest.fetch_data import add_technical_indicators

# Configuration
TRAINED_MODELS_DIR = Path(__file__).parent / "trained_models"
TRAINING_PERIOD = "5y"  # 5 years of historical data
TEST_SIZE = 0.2

# Popular stocks for batch training
POPULAR_TICKERS = [
    "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA",
    "META", "NVDA", "AMD", "NFLX", "DIS"
]


def ensure_model_directory():
    """Create the trained_models directory if it doesn't exist."""
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Model directory: {TRAINED_MODELS_DIR}")


def get_model_path(ticker, model_type):
    """Get the file path for a saved model."""
    ticker_dir = TRAINED_MODELS_DIR / ticker.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)
    return ticker_dir / f"{model_type}.pkl"


def get_scaler_path(ticker, model_type):
    """Get the file path for a saved scaler."""
    ticker_dir = TRAINED_MODELS_DIR / ticker.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)
    return ticker_dir / f"{model_type}_scaler.pkl"


def get_metadata_path(ticker, model_type):
    """Get the file path for model metadata."""
    ticker_dir = TRAINED_MODELS_DIR / ticker.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)
    return ticker_dir / f"{model_type}_metadata.json"


def save_model_metadata(ticker, model_type, metrics, feature_count, training_samples):
    """Save model training metadata."""
    metadata = {
        "ticker": ticker.upper(),
        "model_type": model_type,
        "trained_at": datetime.now().isoformat(),
        "training_period": TRAINING_PERIOD,
        "test_size": TEST_SIZE,
        "metrics": {
            "r2_score": float(metrics.get("r2", 0)),
            "mae": float(metrics.get("mae", 0)),
            "rmse": float(metrics.get("rmse", 0)),
            "mse": float(metrics.get("mse", 0))
        },
        "feature_count": feature_count,
        "training_samples": training_samples
    }

    metadata_path = get_metadata_path(ticker, model_type)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


def train_and_save_linear_regression(ticker, force_retrain=False):
    """Train and save a Linear Regression model for a ticker."""
    print(f"\n{'='*60}")
    print(f"Training Linear Regression Model for {ticker.upper()}")
    print(f"{'='*60}")

    model_path = get_model_path(ticker, "linear_regression")
    scaler_path = get_scaler_path(ticker, "linear_regression")

    # Check if model already exists
    if model_path.exists() and not force_retrain:
        print(f"Model already exists at {model_path}")
        print("Use --retrain flag to force retraining")
        return False

    try:
        # Fetch data
        print(f"Fetching {TRAINING_PERIOD} of data for {ticker}...")
        df = fetch_stock_data(ticker, period=TRAINING_PERIOD)

        if df is None or len(df) < 100:
            print(f"ERROR: Insufficient data for {ticker}")
            return False

        print(f"Data fetched: {len(df)} rows")

        # Add technical indicators and prepare features
        print("Adding technical indicators...")
        from linear_reg.fetch_data import add_technical_indicators
        from linear_reg.predict import prepare_features
        df = add_technical_indicators(df)
        df = prepare_features(df)

        # Train model
        print("Training Linear Regression model...")
        result = train_linear_regression_model(df, test_size=TEST_SIZE)

        if result is None:
            print(f"ERROR: Training failed for {ticker}")
            return False

        model = result['model']
        scaler = result['scaler']
        metrics = {
            'r2': result['r2'],
            'mae': result['mae'],
            'rmse': result['rmse'],
            'mse': result['mse']
        }

        # Save model and scaler
        print(f"Saving model to {model_path}...")
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        # Save metadata
        feature_count = len(result['feature_cols']) if 'feature_cols' in result else 0
        training_samples = len(df)
        metadata = save_model_metadata(ticker, "linear_regression", metrics,
                                       feature_count, training_samples)

        # Print results
        print(f"\n{'-'*60}")
        print(f"Training Complete for {ticker.upper()} - Linear Regression")
        print(f"{'-'*60}")
        print(f"R² Score:  {metrics['r2']:.4f}")
        print(f"MAE:       ${metrics['mae']:.2f}")
        print(f"RMSE:      ${metrics['rmse']:.2f}")
        print(f"Features:  {feature_count}")
        print(f"Samples:   {training_samples}")
        print(f"Model:     {model_path}")
        print(f"{'-'*60}\n")

        return True

    except Exception as e:
        print(f"ERROR training Linear Regression for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def train_and_save_random_forest(ticker, force_retrain=False):
    """Train and save a Random Forest model for a ticker."""
    print(f"\n{'='*60}")
    print(f"Training Random Forest Model for {ticker.upper()}")
    print(f"{'='*60}")

    model_path = get_model_path(ticker, "random_forest")
    scaler_path = get_scaler_path(ticker, "random_forest")

    # Check if model already exists
    if model_path.exists() and not force_retrain:
        print(f"Model already exists at {model_path}")
        print("Use --retrain flag to force retraining")
        return False

    try:
        # Fetch data
        print(f"Fetching {TRAINING_PERIOD} of data for {ticker}...")
        df = fetch_stock_data(ticker, period=TRAINING_PERIOD)

        if df is None or len(df) < 100:
            print(f"ERROR: Insufficient data for {ticker}")
            return False

        print(f"Data fetched: {len(df)} rows")

        # Add technical indicators and prepare features
        print("Adding technical indicators...")
        from random_forest.fetch_data import add_technical_indicators
        from random_forest.predict import prepare_features
        df = add_technical_indicators(df)
        df = prepare_features(df)

        # Train model with optimized parameters
        print("Training Random Forest model (300 trees)...")
        result = train_random_forest_model(
            df,
            test_size=TEST_SIZE,
            n_estimators=300,
            use_optimized_params=True
        )

        if result is None:
            print(f"ERROR: Training failed for {ticker}")
            return False

        model = result['model']
        scaler = result['scaler']
        metrics = {
            'r2': result['r2'],
            'mae': result['mae'],
            'rmse': result['rmse'],
            'mse': result['mse']
        }

        # Save model and scaler
        print(f"Saving model to {model_path}...")
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        # Save metadata
        feature_count = len(result['feature_cols']) if 'feature_cols' in result else 0
        training_samples = len(df)
        metadata = save_model_metadata(ticker, "random_forest", metrics,
                                       feature_count, training_samples)

        # Print results
        print(f"\n{'-'*60}")
        print(f"Training Complete for {ticker.upper()} - Random Forest")
        print(f"{'-'*60}")
        print(f"R² Score:  {metrics['r2']:.4f}")
        print(f"MAE:       ${metrics['mae']:.2f}")
        print(f"RMSE:      ${metrics['rmse']:.2f}")
        print(f"Features:  {feature_count}")
        print(f"Samples:   {training_samples}")
        print(f"Model:     {model_path}")
        print(f"{'-'*60}\n")

        return True

    except Exception as e:
        print(f"ERROR training Random Forest for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def train_ticker(ticker, force_retrain=False, models=None):
    """Train both models for a single ticker."""
    if models is None:
        models = ["linear_regression", "random_forest"]

    results = {}

    if "linear_regression" in models:
        results["linear_regression"] = train_and_save_linear_regression(ticker, force_retrain)

    if "random_forest" in models:
        results["random_forest"] = train_and_save_random_forest(ticker, force_retrain)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train stock prediction models and save them as .pkl files"
    )
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Stock ticker symbols (e.g., AAPL GOOGL MSFT)"
    )
    parser.add_argument(
        "--popular",
        action="store_true",
        help="Train models on popular stocks"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retrain even if models exist"
    )
    parser.add_argument(
        "--model",
        choices=["linear_regression", "random_forest", "both"],
        default="both",
        help="Which model to train (default: both)"
    )

    args = parser.parse_args()

    # Determine which tickers to train
    tickers = []
    if args.popular:
        tickers = POPULAR_TICKERS
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        parser.print_help()
        print("\nERROR: Please specify tickers or use --popular flag")
        sys.exit(1)

    # Determine which models to train
    models = []
    if args.model == "both":
        models = ["linear_regression", "random_forest"]
    else:
        models = [args.model]

    # Setup
    ensure_model_directory()

    print(f"\n{'#'*60}")
    print(f"# Stock Prediction Model Training")
    print(f"{'#'*60}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Models: {', '.join(models)}")
    print(f"Training Period: {TRAINING_PERIOD}")
    print(f"Force Retrain: {args.retrain}")
    print(f"{'#'*60}\n")

    # Train models
    overall_results = {}
    for ticker in tickers:
        print(f"\n\n{'*'*60}")
        print(f"* Processing {ticker.upper()}")
        print(f"{'*'*60}")
        results = train_ticker(ticker, force_retrain=args.retrain, models=models)
        overall_results[ticker] = results

    # Summary
    print(f"\n\n{'#'*60}")
    print(f"# Training Summary")
    print(f"{'#'*60}")
    for ticker, results in overall_results.items():
        print(f"\n{ticker.upper()}:")
        for model_type, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED/SKIPPED"
            print(f"  {model_type:20s}: {status}")
    print(f"\n{'#'*60}\n")


if __name__ == "__main__":
    main()
