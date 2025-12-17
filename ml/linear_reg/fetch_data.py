"""
Script to fetch stock data from Yahoo Finance.
This module provides functions to download historical stock data and save it to CSV files.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def fetch_stock_data(ticker, period="2y", interval="1d", start_date=None, end_date=None):
    """
    Fetch historical stock data from Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        period (str): Time period to fetch (valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Data interval (valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        start_date (str): Start date in 'YYYY-MM-DD' format (optional, overrides period)
        end_date (str): End date in 'YYYY-MM-DD' format (optional)

    Returns:
        pd.DataFrame: DataFrame with columns [Open, High, Low, Close, Volume, Adj Close]
    """
    try:
        stock = yf.Ticker(ticker)

        if start_date and end_date:
            df = stock.history(start=start_date, end=end_date, interval=interval)
        elif start_date:
            df = stock.history(start=start_date, interval=interval)
        else:
            df = stock.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")

        # Reset index to make Date a column
        df.reset_index(inplace=True)

        return df

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        raise


def add_technical_indicators(df):
    """
    Add technical indicators to the stock data for better predictions.

    Args:
        df (pd.DataFrame): DataFrame with stock data

    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    # Moving Averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    # Exponential Moving Average
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    # Volume moving average
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

    # Price change percentage
    df['Price_Change'] = df['Close'].pct_change()

    # Average True Range (ATR) - Volatility indicator
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()

    # Average Directional Index (ADX) - Trend strength
    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr14 = true_range.rolling(window=14).sum()
    plus_di14 = 100 * (plus_dm.rolling(window=14).sum() / tr14)
    minus_di14 = 100 * (minus_dm.rolling(window=14).sum() / tr14)

    dx = 100 * abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    df['ADX'] = dx.rolling(window=14).mean()
    df['Plus_DI'] = plus_di14
    df['Minus_DI'] = minus_di14

    # On-Balance Volume (OBV) - Volume indicator
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    df['OBV_MA'] = df['OBV'].rolling(window=20).mean()

    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stochastic_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()

    # Commodity Channel Index (CCI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = typical_price.rolling(window=20).mean()
    mad = typical_price.rolling(window=20).apply(lambda x: abs(x - x.mean()).mean())
    df['CCI'] = (typical_price - sma_tp) / (0.015 * mad)

    # Williams %R
    df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))

    # Money Flow Index (MFI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = money_flow.where(df['Close'] > df['Close'].shift(1), 0).rolling(window=14).sum()
    negative_flow = money_flow.where(df['Close'] < df['Close'].shift(1), 0).rolling(window=14).sum()
    mfi_ratio = positive_flow / negative_flow
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))

    # Rate of Change (ROC)
    df['ROC'] = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12)) * 100

    # Drop NaN values created by rolling calculations
    df = df.dropna()

    return df


def save_to_csv(df, filename):
    """
    Save DataFrame to CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str): Output filename
    """
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


def fetch_and_save(ticker, filename=None, period="2y", interval="1d",
                   add_indicators=True, start_date=None, end_date=None):
    """
    Fetch stock data and save to CSV file.

    Args:
        ticker (str): Stock ticker symbol
        filename (str): Output filename (defaults to {ticker}_data.csv)
        period (str): Time period to fetch
        interval (str): Data interval
        add_indicators (bool): Whether to add technical indicators
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
        pd.DataFrame: The fetched and processed data
    """
    if filename is None:
        filename = f"{ticker}_data.csv"

    print(f"Fetching data for {ticker}...")
    df = fetch_stock_data(ticker, period, interval, start_date, end_date)

    if add_indicators:
        print("Adding technical indicators...")
        df = add_technical_indicators(df)

    save_to_csv(df, filename)

    print(f"Successfully fetched {len(df)} rows of data")
    print(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")

    return df


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        ticker = sys.argv[1]
        filename = sys.argv[2] if len(sys.argv) > 2 else None
        df = fetch_and_save(ticker, filename)
    else:
        print("Usage: python fetch_data.py TICKER [output_filename.csv]")
        print("Example: python fetch_data.py AAPL apple_data.csv")
        print("\nFetching example data for AAPL...")
        df = fetch_and_save("AAPL", "AAPL_data.csv")
