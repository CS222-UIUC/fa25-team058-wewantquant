import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

import torch
from ml.baseTorchLSTM.variableLSTM import StockLSTM
import pandas as pd

def getdata(n = 10, ticker = "AAPL"):
    #ticker = "AAPL"
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=n*1.5)).strftime('%Y-%m-%d')

    print(f"Fetching {ticker} data from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    data = data[-n:]
    data.columns = [f"{col[0]}" if isinstance(col, tuple) else col for col in data.columns]
    data = data.reset_index()

    return data

def get_indicators(df):
    """Add momentum and trend features to the data"""
    print("Adding technical indicators...")
    df = df.copy()
    
    # Moving Averages (Trend)
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD (Momentum)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # RSI (Momentum)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (Volatility/Trend)
    bb_middle = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Middle'] = bb_middle
    df['BB_Upper'] = bb_middle + (bb_std * 2)
    df['BB_Lower'] = bb_middle - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    
    # Rate of Change (Momentum)
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    
    # Volume indicators
    volume_ma = df['Volume'].rolling(window=20).mean()
    df['Volume_MA_20'] = volume_ma
    df['Volume_Ratio'] = df['Volume'] / volume_ma
    
    # Price momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(4)
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=10).std()
    
    # Drop rows with NaN values (from rolling calculations)
    df = df.dropna()
    
    print(f"Added {len(df.columns) - 5} technical indicators")
    return df

def predict(ticker = "AAPL", horizon = 7):
    print("ENTRY LSTM ticker", ticker)
    filepath = "./models/LSTM_best.pth"
    ckpt = torch.load(filepath, map_location="cpu", weights_only=False)
    scaler = ckpt['scaler']
    model_state_dict = ckpt['model_state_dict']
    #ticker = ckpt['ticker']            WE DONT CARE ABOUT THIS
    lookback = ckpt['lookback']
    max_horizon = ckpt['max_horizon']
    raw_data = ckpt['raw_data']
    feature_data = ckpt['feature_data']

    data = getdata(lookback + 50, ticker) 

    df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])

    prep_data = get_indicators(df)

    lookback_data = prep_data.iloc[-lookback:].values
    scaled_data = scaler.transform(lookback_data)

    X = scaled_data.reshape(1, lookback, -1)
    
    device = torch.device('cpu')
    model = StockLSTM(
        input_size=24,  # 5 OHLCV + 18 technical indicators BUG FIX 23 TO 24
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        max_horizon=max_horizon
    ).to(device)
    
    # Load weights
    model.load_state_dict(model_state_dict)
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor)  # Shape: (1, 30)
        predictions = predictions[:, :horizon].cpu().numpy()  # Shape: (1, horizon)
    
    # Inverse transform to get dollar prices
    num_features = scaler.n_features_in_
    dummy = np.zeros((1, horizon, num_features))
    dummy[:, :, 3] = predictions  # Close price at index 3
    
    dummy_reshaped = dummy.reshape(-1, num_features)
    inv_transformed = scaler.inverse_transform(dummy_reshaped)
    predicted_prices = inv_transformed[:, 3].reshape(1, horizon)

    current_close = prep_data['Close'].iloc[-1]
    print("CURRENT CLOSE", current_close)
    
    return current_close, predicted_prices[0]

if __name__ == "__main__":
    result = predict()
    print(result)