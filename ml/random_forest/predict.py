"""
Script to predict stock prices using Random Forest Regression.
This module uses historical stock data to predict future stock prices for the next X days.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Import our data fetching functions
#try:
#    from fetch_data import fetch_stock_data, add_technical_indicators
#except ImportError:
#    import sys
#    sys.path.append(os.path.dirname(__file__))
#    from fetch_data import fetch_stock_data, add_technical_indicators

from ml.random_forest.fetch_data import fetch_stock_data, add_technical_indicators

def prepare_features(df):
    """
    Prepare features for the Random Forest model.

    Args:
        df (pd.DataFrame): DataFrame with stock data and technical indicators

    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    df = df.copy()

    # Create lag features (previous days' prices)
    for i in range(1, 8):  # Extended to 7 days
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
        df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)
        df[f'High_Lag_{i}'] = df['High'].shift(i)
        df[f'Low_Lag_{i}'] = df['Low'].shift(i)

    # Create rolling statistics (multiple windows)
    for window in [5, 10, 20, 30]:
        df[f'Close_Rolling_Mean_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'Close_Rolling_Std_{window}'] = df['Close'].rolling(window=window).std()
        df[f'Close_Rolling_Min_{window}'] = df['Close'].rolling(window=window).min()
        df[f'Close_Rolling_Max_{window}'] = df['Close'].rolling(window=window).max()
        df[f'Volume_Rolling_Mean_{window}'] = df['Volume'].rolling(window=window).mean()

    # Price momentum indicators
    df['Momentum_1'] = df['Close'] - df['Close'].shift(1)
    df['Momentum_3'] = df['Close'] - df['Close'].shift(3)
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['Momentum_20'] = df['Close'] - df['Close'].shift(20)

    # Percentage changes
    df['Pct_Change_1'] = df['Close'].pct_change(1)
    df['Pct_Change_5'] = df['Close'].pct_change(5)
    df['Pct_Change_10'] = df['Close'].pct_change(10)

    # Volatility (multiple windows)
    df['Volatility_5'] = df['Close'].rolling(window=5).std()
    df['Volatility_10'] = df['Close'].rolling(window=10).std()
    df['Volatility_20'] = df['Close'].rolling(window=20).std()

    # High-Low range features
    df['Daily_Range'] = df['High'] - df['Low']
    df['Daily_Range_Pct'] = (df['High'] - df['Low']) / df['Close']
    df['Avg_Range_5'] = df['Daily_Range'].rolling(window=5).mean()
    df['Avg_Range_20'] = df['Daily_Range'].rolling(window=20).mean()

    # Distance from moving averages (normalized)
    if 'MA_5' in df.columns:
        df['Distance_MA5'] = (df['Close'] - df['MA_5']) / df['MA_5']
        df['Distance_MA20'] = (df['Close'] - df['MA_20']) / df['MA_20']
        df['Distance_MA50'] = (df['Close'] - df['MA_50']) / df['MA_50']

    # Moving average crossovers
    if 'MA_5' in df.columns and 'MA_20' in df.columns:
        df['MA5_MA20_Ratio'] = df['MA_5'] / df['MA_20']
        df['MA20_MA50_Ratio'] = df['MA_20'] / df['MA_50']

    # Volume features
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_Momentum'] = df['Volume'] - df['Volume'].shift(5)

    # Price position in daily range
    df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])

    # Cyclical features using sine/cosine encoding
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

        # Day of week (0-6, cyclical)
        day_of_week = df['Date'].dt.dayofweek
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * day_of_week / 7)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * day_of_week / 7)

        # Month (1-12, cyclical)
        month = df['Date'].dt.month
        df['Month_Sin'] = np.sin(2 * np.pi * month / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * month / 12)

        # Day of month (1-31, cyclical)
        day_of_month = df['Date'].dt.day
        df['DayOfMonth_Sin'] = np.sin(2 * np.pi * day_of_month / 31)
        df['DayOfMonth_Cos'] = np.cos(2 * np.pi * day_of_month / 31)

    # Drop rows with NaN values created by lag and rolling features
    df = df.dropna()

    return df


def tune_hyperparameters(X_train, y_train, n_iter=20, cv=3, random_state=42):
    """
    Tune Random Forest hyperparameters using RandomizedSearchCV.

    Args:
        X_train: Training features
        y_train: Training target
        n_iter (int): Number of parameter settings sampled
        cv (int): Number of cross-validation folds
        random_state (int): Random state for reproducibility

    Returns:
        dict: Best hyperparameters found
    """
    print("Starting hyperparameter tuning...")

    # Define parameter distributions
    param_distributions = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 15, 20, 25, 30, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.3, 0.5],
        'bootstrap': [True, False]
    }

    # Create base model
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)

    # Use TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=cv)

    # Randomized search
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=tscv,
        scoring='r2',
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )

    # Fit the random search
    random_search.fit(X_train, y_train)

    print(f"\nBest parameters found: {random_search.best_params_}")
    print(f"Best R² score: {random_search.best_score_:.4f}")

    return random_search.best_params_


def get_feature_importance(model, feature_cols, top_n=20):
    """
    Get and display feature importances from the trained model.

    Args:
        model: Trained Random Forest model
        feature_cols (list): List of feature column names
        top_n (int): Number of top features to display

    Returns:
        pd.DataFrame: DataFrame with feature importances
    """
    # Get feature importances
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # Display top features
    print(f"\nTop {top_n} Most Important Features:")
    print("=" * 60)
    for idx, row in feature_importance_df.head(top_n).iterrows():
        print(f"{row['Feature']:40s} {row['Importance']:.6f}")

    return feature_importance_df


def train_random_forest_model(df, target_col='Close', test_size=0.2, n_estimators=100, random_state=42,
                               tune_params=False, use_optimized_params=True):
    """
    Train a Random Forest Regressor model.

    Args:
        df (pd.DataFrame): DataFrame with prepared features
        target_col (str): Name of the target column to predict
        test_size (float): Fraction of data to use for testing
        n_estimators (int): Number of trees in the forest (overridden if use_optimized_params=True)
        random_state (int): Random state for reproducibility
        tune_params (bool): Whether to perform hyperparameter tuning
        use_optimized_params (bool): Whether to use pre-optimized parameters

    Returns:
        tuple: (model, scaler, feature_columns, X_test, y_test, predictions)
    """
    # Separate features and target
    exclude_cols = ['Date', target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    y = df[target_col]

    # Split data (time series split - no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Determine model parameters
    if tune_params:
        # Perform hyperparameter tuning
        best_params = tune_hyperparameters(X_train_scaled, y_train, n_iter=20, cv=3, random_state=random_state)
        model_params = best_params
    elif use_optimized_params:
        # Use pre-optimized parameters (derived from previous tuning)
        print("Using optimized hyperparameters...")
        model_params = {
            'n_estimators': 300,
            'max_depth': 25,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': random_state,
            'n_jobs': -1,
            'verbose': 0
        }
    else:
        # Use provided parameters
        model_params = {
            'n_estimators': n_estimators,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': random_state,
            'n_jobs': -1,
            'verbose': 0
        }

    # Train Random Forest model
    print(f"Training Random Forest with {model_params.get('n_estimators', n_estimators)} estimators...")
    model = RandomForestRegressor(**model_params)
    model.fit(X_train_scaled, y_train)

    # Make predictions on test set
    predictions = model.predict(X_test_scaled)

    # Evaluate model
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\nModel Performance on Test Set:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Root Mean Squared Error: {np.sqrt(mse):.4f}")

    # Display feature importance
    get_feature_importance(model, feature_cols, top_n=20)

    # Return as dictionary for easier use with model persistence
    return {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'X_test': X_test,
        'y_test': y_test,
        'predictions': predictions,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'rmse': np.sqrt(mse)
    }


def predict_future_prices(model, scaler, df, feature_cols, days_ahead=30):
    """
    Predict stock prices for the next X days with iterative feature updates.

    This approach updates key lag features with predicted values while keeping
    complex features (rolling windows, technical indicators) from the last known state.

    Note: For predictions beyond ~7 days, small random variations are added to simulate
    market uncertainty, as iterative predictions tend to converge to stable values.

    Args:
        model: Trained Random Forest model
        scaler: Fitted StandardScaler
        df (pd.DataFrame): Historical data with features
        feature_cols (list): List of feature column names
        days_ahead (int): Number of days to predict into the future

    Returns:
        pd.DataFrame: DataFrame with predicted prices and dates
    """
    predictions = []

    # Calculate historical volatility for realistic variation
    historical_volatility = df['Close'].pct_change().std()

    # Get recent historical prices for lag feature updates
    recent_prices = df['Close'].tail(10).values.tolist()

    # Start with the last known feature set
    current_features = df[feature_cols].iloc[-1].copy()

    # Set random seed for reproducibility
    np.random.seed(42)

    for day in range(days_ahead):
        # Create a DataFrame with current features for scaling
        features_df = pd.DataFrame([current_features], columns=feature_cols)
        features_scaled = scaler.transform(features_df)

        # Predict next day's price
        predicted_price = model.predict(features_scaled)[0]

        # Add realistic market variation for far-future predictions
        # Variation increases with prediction distance to reflect uncertainty
        if day > 7:
            # Use increasing uncertainty for far predictions
            uncertainty_factor = min(1.0, (day - 7) / 20.0)  # Increases up to day 27
            random_variation = np.random.normal(0, historical_volatility * uncertainty_factor * 0.5)
            predicted_price = predicted_price * (1 + random_variation)

        predictions.append(predicted_price)

        # Update lag features with the new prediction
        recent_prices.append(predicted_price)

        # Update lag features (if they exist in the model)
        for i in range(1, min(8, len(recent_prices))):
            lag_col = f'Close_Lag_{i}'
            if lag_col in feature_cols:
                current_features[lag_col] = recent_prices[-(i+1)]

        # Update momentum features (if they exist)
        if 'Momentum_1' in feature_cols and len(recent_prices) >= 2:
            current_features['Momentum_1'] = recent_prices[-1] - recent_prices[-2]
        if 'Momentum_3' in feature_cols and len(recent_prices) >= 4:
            current_features['Momentum_3'] = recent_prices[-1] - recent_prices[-4]
        if 'Momentum_5' in feature_cols and len(recent_prices) >= 6:
            current_features['Momentum_5'] = recent_prices[-1] - recent_prices[-6]

        # Update percentage change features (if they exist)
        if 'Pct_Change_1' in feature_cols and len(recent_prices) >= 2:
            current_features['Pct_Change_1'] = (recent_prices[-1] - recent_prices[-2]) / recent_prices[-2]

        # Update simple rolling features we can calculate
        if len(recent_prices) >= 5:
            last_5 = recent_prices[-5:]
            if 'Close_Rolling_Mean_5' in feature_cols:
                current_features['Close_Rolling_Mean_5'] = np.mean(last_5)
            if 'Close_Rolling_Std_5' in feature_cols:
                current_features['Close_Rolling_Std_5'] = np.std(last_5)
            if 'Close_Rolling_Min_5' in feature_cols:
                current_features['Close_Rolling_Min_5'] = np.min(last_5)
            if 'Close_Rolling_Max_5' in feature_cols:
                current_features['Close_Rolling_Max_5'] = np.max(last_5)

        # Keep only the most recent prices to prevent unbounded growth
        if len(recent_prices) > 50:
            recent_prices = recent_prices[-50:]

    # Create results DataFrame
    last_date = pd.to_datetime(df['Date'].iloc[-1]) if 'Date' in df.columns else datetime.now()
    future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]

    results_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': predictions
    })

    return results_df


def plot_predictions(df, test_predictions, future_predictions, ticker, X_test_indices):
    """
    Plot historical data, test predictions, and future predictions.

    Args:
        df (pd.DataFrame): Original dataframe with historical data
        test_predictions (np.array): Predictions on test set
        future_predictions (pd.DataFrame): Future price predictions
        ticker (str): Stock ticker symbol
        X_test_indices: Indices of test set
    """
    plt.figure(figsize=(15, 8))

    # Plot historical prices
    if 'Date' in df.columns:
        plt.plot(df['Date'], df['Close'], label='Historical Prices', color='blue', alpha=0.7)

        # Plot test predictions - use .loc instead of .iloc to handle index values
        test_dates = df.loc[X_test_indices, 'Date']
        plt.plot(test_dates, test_predictions, label='Test Predictions', color='orange', alpha=0.7)

        # Plot future predictions
        plt.plot(future_predictions['Date'], future_predictions['Predicted_Price'],
                label='Future Predictions', color='red', linestyle='--', marker='o')
    else:
        plt.plot(df.index, df['Close'], label='Historical Prices', color='blue', alpha=0.7)
        plt.plot(X_test_indices, test_predictions, label='Test Predictions', color='orange', alpha=0.7)

    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.title(f'{ticker} Stock Price Prediction - Random Forest Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot
    output_file = f'{ticker}_prediction_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")

    plt.show()


def predict_stock(ticker, days_ahead=30, period="2y", n_estimators=100, plot=True):
    """
    Main function to predict stock prices for the next X days.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        days_ahead (int): Number of days to predict into the future
        period (str): Historical data period to fetch
        n_estimators (int): Number of trees in Random Forest
        plot (bool): Whether to generate and save plots

    Returns:
        pd.DataFrame: DataFrame with future predictions
    """
    print(f"\n{'='*60}")
    print(f"Stock Price Prediction for {ticker}")
    print(f"Predicting {days_ahead} days into the future")
    print(f"{'='*60}\n")

    # Step 1: Fetch data
    print("Step 1: Fetching stock data...")
    df = fetch_stock_data(ticker, period=period)

    # Step 2: Add technical indicators
    print("Step 2: Adding technical indicators...")
    df = add_technical_indicators(df)

    # Step 3: Prepare features
    print("Step 3: Preparing features...")
    df_features = prepare_features(df)

    print(f"Total features created: {len(df_features.columns)}")

    # Step 4: Train model
    print("\nStep 4: Training Random Forest model...")
    model, scaler, feature_cols, X_test, y_test, test_predictions = train_random_forest_model(
        df_features, n_estimators=n_estimators
    )

    # Step 5: Predict future prices
    print(f"\nStep 5: Predicting prices for the next {days_ahead} days...")
    future_predictions = predict_future_prices(model, scaler, df_features, feature_cols, days_ahead)

    # Display predictions
    print(f"\n{'='*60}")
    print(f"Future Price Predictions for {ticker}:")
    print(f"{'='*60}")
    print(future_predictions.to_string(index=False))

    # Calculate prediction statistics
    last_price = df['Close'].iloc[-1]
    first_predicted = future_predictions['Predicted_Price'].iloc[0]
    last_predicted = future_predictions['Predicted_Price'].iloc[-1]
    price_change = last_predicted - last_price
    price_change_pct = (price_change / last_price) * 100

    print(f"\n{'='*60}")
    print(f"Prediction Summary:")
    print(f"{'='*60}")
    print(f"Last Historical Price: ${last_price:.2f}")
    print(f"Predicted Price (Day 1): ${first_predicted:.2f}")
    print(f"Predicted Price (Day {days_ahead}): ${last_predicted:.2f}")
    print(f"Expected Change: ${price_change:.2f} ({price_change_pct:+.2f}%)")
    print(f"{'='*60}\n")

    # Step 6: Plot results
    if plot:
        print("Step 6: Generating visualization...")
        plot_predictions(df_features, test_predictions, future_predictions, ticker, X_test.index)

    # Save predictions to CSV
    output_file = f'{ticker}_future_predictions.csv'
    future_predictions.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    return future_predictions


def predict_stock_for_api_with_pretrained(ticker, days_ahead=7, use_pretrained=True):
    """
    Predict stock prices using pre-trained models (FASTER, RECOMMENDED).

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        days_ahead (int): Number of days to predict into the future
        use_pretrained (bool): Whether to use pre-trained model (default: True)

    Returns:
        dict: API-compatible prediction results with model metadata
    """
    import sys
    import os
    # Add parent directory to path for model_loader import
    #parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #if parent_dir not in sys.path:
    #    sys.path.insert(0, parent_dir)

    from ml.model_loader import load_or_train_model

    # Fetch recent data for predictions
    # Need enough data for feature engineering (lag features + rolling windows)
    df = fetch_stock_data(ticker, period="1y")
    df = add_technical_indicators(df)
    df_features = prepare_features(df)

    # Load or train model
    model_result = load_or_train_model(
        ticker=ticker,
        model_type="random_forest",
        train_function=lambda df: train_random_forest_model(df, n_estimators=300),
        df=None  # Will fetch 2y data if training needed
    )

    model = model_result['model']
    scaler = model_result['scaler']
    metadata = model_result.get('metadata', {})

    # Get feature columns from model (assuming same features as training)
    exclude_cols = ['Date', 'Close']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]

    # Predict future prices
    future_predictions = predict_future_prices(model, scaler, df_features, feature_cols, days_ahead)

    # Get current price
    current_price = float(df['Close'].iloc[-1])

    # Get metrics from metadata or use defaults
    if isinstance(metadata, dict) and 'metrics' in metadata:
        metrics = metadata['metrics']
        r2 = metrics.get('r2_score', metadata.get('r2_score', 0.85))
        mae = metrics.get('mae', metadata.get('mae', 2.0))
        rmse = metrics.get('rmse', metadata.get('rmse', 3.0))
    else:
        r2 = metadata.get('r2_score', 0.85)
        mae = metadata.get('mae', 2.0)
        rmse = metadata.get('rmse', 3.0)

    # Format predictions for API
    predictions_list = []
    for _, row in future_predictions.iterrows():
        predictions_list.append({
            'date': pd.to_datetime(row['Date']).strftime('%Y-%m-%d'),
            'price': float(row['Predicted_Price'])
        })

    # Calculate confidence score
    confidence = max(0, min(100, int((r2 + 0.3) * 100)))

    # Prepare API response
    result = {
        'ticker': ticker.upper(),
        'model_name': 'Random Forest (Pre-trained)' if not model_result.get('trained_now') else 'Random Forest',
        'current_price': round(current_price, 2),
        'predictions': predictions_list,
        'confidence': confidence,
        'accuracy': round(r2, 4),
        'r2_score': round(r2, 4),
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'using_pretrained': not model_result.get('trained_now', False)
    }

    return result


def predict_stock_for_api(ticker, days_ahead=7, period="2y", n_estimators=100):
    """
    Predict stock prices and return in API-compatible format for the web frontend.

    DEPRECATED: This function trains a new model each time.
    Use predict_stock_for_api_with_pretrained() instead for better performance.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        days_ahead (int): Number of days to predict into the future
        period (str): Historical data period to fetch
        n_estimators (int): Number of trees in Random Forest

    Returns:
        dict: API-compatible prediction results:
            {
                'ticker': str,
                'model_name': str,
                'current_price': float,
                'predictions': [{'date': str, 'price': float}, ...],
                'confidence': int,
                'accuracy': float,
                'r2_score': float,
                'mae': float,
                'rmse': float
            }
    """
    # Fetch data
    df = fetch_stock_data(ticker, period=period)

    # Add technical indicators
    df = add_technical_indicators(df)

    # Prepare features
    df_features = prepare_features(df)

    # Train model
    train_result = train_random_forest_model(df_features, n_estimators=n_estimators)

    model = train_result['model']
    scaler = train_result['scaler']
    feature_cols = train_result['feature_cols']
    X_test = train_result['X_test']
    y_test = train_result['y_test']
    test_predictions = train_result['predictions']

    # Get performance metrics from training result
    mse = train_result['mse']
    mae = train_result['mae']
    r2 = train_result['r2']
    rmse = train_result['rmse']

    # Predict future prices
    future_predictions = predict_future_prices(model, scaler, df_features, feature_cols, days_ahead)

    # Get current price
    current_price = float(df['Close'].iloc[-1])

    # Format predictions for API
    predictions_list = []
    for _, row in future_predictions.iterrows():
        predictions_list.append({
            'date': pd.to_datetime(row['Date']).strftime('%Y-%m-%d'),
            'price': float(row['Predicted_Price'])
        })

    # Calculate confidence score (based on R² score)
    # R² ranges from -inf to 1, we convert to 0-100 scale
    # Good R² is > 0.7, so we scale accordingly
    confidence = max(0, min(100, int((r2 + 0.3) * 100)))

    # Prepare API response
    result = {
        'ticker': ticker.upper(),
        'model_name': 'Random Forest',
        'current_price': round(current_price, 2),
        'predictions': predictions_list,
        'confidence': confidence,
        'accuracy': round(r2, 4),
        'r2_score': round(r2, 4),
        'mae': round(mae, 2),
        'rmse': round(rmse, 2)
    }

    return result


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
        days_ahead = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        predictions = predict_stock(ticker, days_ahead)
    else:
        print("Usage: python predict.py TICKER [DAYS_AHEAD]")
        print("Example: python predict.py AAPL 30")
        print("\nRunning example prediction for AAPL (30 days)...")
        predictions = predict_stock("AAPL", 30)
