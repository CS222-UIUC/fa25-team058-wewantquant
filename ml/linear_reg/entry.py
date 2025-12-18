"""
Entry point for Linear Regression model integration with the Flask backend.
This module provides a clean interface for the web application to use the Linear Regression model.
"""

import sys
import os

# Add parent directory to path to allow imports
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.linear_reg.predict import predict_stock_for_api_with_pretrained, predict_stock_for_api
from ml.linear_reg.fetch_data import fetch_stock_data
import pandas as pd
from datetime import datetime


def run_prediction(ticker, days_ahead=7, period="2y", use_pretrained=True):
    """
    Run Linear Regression prediction and return results in API-compatible format.

    This function now uses PRE-TRAINED MODELS by default for faster predictions
    and better accuracy. Models are trained on 5 years of data and saved as .pkl files.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        days_ahead (int): Number of days to predict into the future
        period (str): Historical data period to fetch (default: "2y")
                     NOTE: Only used if use_pretrained=False
        use_pretrained (bool): Use pre-trained model (default: True, RECOMMENDED)

    Returns:
        dict: Prediction results in the format expected by the frontend:
            {
                'ticker': str,
                'model_name': str,
                'current_price': float,
                'predictions': [
                    {'date': str, 'price': float},
                    ...
                ],
                'confidence': int,
                'accuracy': float,
                'using_pretrained': bool
            }

    Raises:
        Exception: If prediction fails

    Notes:
        - Pre-trained models are MUCH faster (no training time)
        - Pre-trained models are trained on 5 years of data (better accuracy)
        - If no pre-trained model exists, will train on-the-fly and suggest saving
        - To train models: python train_models.py TICKER
    """
    try:
        if use_pretrained:
            # Use pre-trained model (FASTER, RECOMMENDED)
            result = predict_stock_for_api_with_pretrained(
                ticker=ticker,
                days_ahead=days_ahead
            )
        else:
            # Train new model each time (SLOWER, legacy behavior)
            result = predict_stock_for_api(
                ticker=ticker,
                days_ahead=days_ahead,
                period=period
            )

        return result

    except Exception as e:
        raise Exception(f"Prediction failed for {ticker}: {str(e)}")


def get_current_price(ticker):
    """
    Fetch the current stock price.

    Args:
        ticker (str): Stock ticker symbol

    Returns:
        float: Current stock price
    """
    try:
        df = fetch_stock_data(ticker, period="5d")
        return float(df['Close'].iloc[-1])
    except Exception as e:
        raise Exception(f"Failed to fetch current price for {ticker}: {str(e)}")


if __name__ == "__main__":
    # Example usage for testing
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 7

        print(f"\nRunning prediction for {ticker} ({days} days)...\n")
        result = run_prediction(ticker, days)

        print(f"Ticker: {result['ticker']}")
        print(f"Model: {result['model_name']}")
        print(f"Current Price: ${result['current_price']:.2f}")
        print(f"Confidence: {result['confidence']}%")
        print(f"R² Score: {result['accuracy']:.2f}")
        print(f"\nPredictions:")
        for pred in result['predictions']:
            print(f"  {pred['date']}: ${pred['price']:.2f}")
    else:
        print("Usage: python entry.py TICKER [DAYS_AHEAD]")
        print("Example: python entry.py AAPL 7")
