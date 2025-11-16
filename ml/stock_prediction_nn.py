"""
Simple Neural Network for Stock Price Prediction

This script implements a basic feedforward neural network to predict
next-day stock closing prices using historical OHLCV (Open, high, Low, Close, Volume) data.

Our stock data files contain: 
8,507 symbols with historical data from 1962-2017
High-liquidity stocks: AAPL, MSFT, SPY, QQQ (good for training)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ----------------------------------------------------------------------------
# DATA LOADING AND PREPROCESSING
# ----------------------------------------------------------------------------

def load_stock_data(file_path: str) -> pd.DataFrame:
    """
    Function to load stock data from a text file.

    Args:
        file_path: Path to the stock data file

    Returns:
        DataFrame with Date index and OHLCV columns
    """
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df.sort_values('Date')
    df = df.set_index('Date')
    return df


def create_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Function to create technical indicators and rolling features.

    Args:
        df: DataFrame with OHLCV (open, high, low, close, volume) data
        lookback: Number of days to look back for rolling features

    Returns:
        DataFrame with additional feature columns on top of OHLCV 

    Took the help of an LLM to generate this function to get an idea of what stock features we should look at and predict
    """
    df = df.copy()

    # Daily returns
    # Captures daily % change
    df['returns'] = df['Close'].pct_change()

    # Logarithmic returns
    # This data is used for stabilizing volatility
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1)) 

    # Moving averages
    # This data is used for smoothening noise and learning trends
    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_10'] = df['Close'].rolling(window=10).mean()
    df['ma_20'] = df['Close'].rolling(window=20).mean() 

    # Volatility
    # Baiscally the standard deviation of returns
    # Used for measuring uncertainty / risk
    df['volatility'] = df['returns'].rolling(window=lookback).std()

    # Price momentum (rate of change)
    # Price % change over time window (here 30 days based on lookback argument passed to function)
    # Purpose is to capture price acceleration
    df['momentum'] = df['Close'] / df['Close'].shift(lookback) - 1

    # Volume features
    df['volume_ma'] = df['Volume'].rolling(window=lookback).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma']

    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # High-Low spread
    df['hl_spread'] = (df['High'] - df['Low']) / df['Close']

    # Drop rows with NaN values created by rolling operations
    df = df.dropna()

    return df


def create_sequences(data: np.ndarray, seq_length: int = 30) -> tuple:
    """
    Function to create sequences for time series prediction.

    Args:
        data: Feature array
        seq_length: Number of time steps to look back

    Returns:
        Tuple of (X, y) where X is sequences and y is targets (next day close)
    """
    X, y = [], []

    for i in range(seq_length, len(data)):
        # Use past seq_length days as features
        X.append(data[i-seq_length:i, :-1])  # All columns except target
        # Predict the next day's closing price (last column)
        y.append(data[i, -1])

    return np.array(X), np.array(y)


def prepare_data(file_path: str, seq_length: int = 30, test_size: float = 0.2):
    """
    Function to complete data preparation pipeline.

    Args:
        file_path: Path to stock data file
        seq_length: Sequence length for lookback window
        test_size: Fraction of data to use for testing

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler)
    """
    # Load and create features from the functions we defined above
    df = load_stock_data(file_path)
    df = create_features(df)

    # Select features for training (OHLC)
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'returns', 'log_returns', 'ma_5', 'ma_10', 'ma_20',
                    'volatility', 'momentum', 'volume_ratio', 'rsi', 'hl_spread']

    # Add target column (next day's close price)
    df['target'] = df['Close'].shift(-1)
    df = df.dropna()

    # Prepare data array by extracting the features we selected from training above
    data = df[feature_cols + ['target']].values 

    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Create sequences
    # Convert time series data into sliding window using the function we defined above 
    X, y = create_sequences(data_scaled, seq_length)

    # Train-test split (keep temporal order)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test, scaler


# ----------------------------------------------------------------------------
# PYTORCH DATASET
# ----------------------------------------------------------------------------

class StockDataset(Dataset):
    """PyTorch Dataset for stock sequences."""

    """
    Stores our our input data (X) and target labels (y) in tensor format.
    Lets PyTorch retrieve samples efficiently when training.
    """

    def __init__(self, X, y):
        """
        Initialize dataset.

        Args:
            X: Feature sequences (3D array: samples x seq_length x features)
            y: Target values (1D array)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    # This converts our data into Pytorch Tensors
    def __len__(self):
        return len(self.X)

    # Tells PyTorch how many samples are in the dataset
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ----------------------------------------------------------------------------
# NEURAL NETWORK MODEL
# ----------------------------------------------------------------------------

class StockPredictor(nn.Module):
    """
    Simple feedforward neural network for stock prediction.

    Architecture:
        - Flatten layer to convert sequences to vectors
        - 3 fully connected hidden layers with ReLU activation
        - Dropout for regularization
        - Single output neuron for price prediction
    """

    def __init__(self, input_size: int, hidden_sizes: list = [128, 64, 32], dropout: float = 0.2):
        """
        Initialize the model.

        Args:
            input_size: Total number of input features (seq_length * num_features)
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout probability for regularization
        """
        super(StockPredictor, self).__init__()

        layers = []
        prev_size = input_size

        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        # Output layer (single value: predicted price)
        layers.append(nn.Linear(prev_size, 1))

        # Combine all layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, seq_length, num_features)

        Returns:
            Predicted prices of shape (batch_size, 1)
        """
        # Flatten the sequence dimension
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        # Pass through network
        return self.network(x)


# ----------------------------------------------------------------------------
# TRAINING AND EVALUATION
# ----------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, epochs: int = 50,
                learning_rate: float = 0.001, device: str = 'cpu'):
    """
    Train the neural network model.

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')

    Returns:
        Tuple of (training_losses, validation_losses)
    """
    model = model.to(device)

    # Loss function: Mean Squared Error for regression
    criterion = nn.MSELoss()

    # Optimizer: Adam
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler: reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.5, patience=5)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            # Forward pass
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)

                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Update learning rate
        scheduler.step(val_loss)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    return train_losses, val_losses


def evaluate_model(model, test_loader, scaler, device: str = 'cpu'):
    """
    Evaluate model performance on test set.

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        scaler: MinMaxScaler used for normalization
        device: Device to evaluate on

    Returns:
        Dictionary with evaluation metrics
    """
    model = model.to(device)
    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)

            predictions.extend(pred.cpu().numpy())
            actuals.extend(y_batch.numpy())

    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    # Denormalize predictions (inverse transform)
    # Create dummy array with all features to use scaler
    dummy = np.zeros((len(predictions), scaler.n_features_in_))
    dummy[:, -1] = predictions  # Put predictions in last column (target)
    predictions_denorm = scaler.inverse_transform(dummy)[:, -1]

    dummy[:, -1] = actuals
    actuals_denorm = scaler.inverse_transform(dummy)[:, -1]

    # Calculate metrics
    mse = np.mean((predictions_denorm - actuals_denorm) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_denorm - actuals_denorm))
    mape = np.mean(np.abs((actuals_denorm - predictions_denorm) / actuals_denorm)) * 100

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'predictions': predictions_denorm,
        'actuals': actuals_denorm
    }


# ----------------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------------

def main():
    """
    Main function to run the complete pipeline.
    """
    # Configuration parameters for the NN
    STOCK_FILE = '../notebooks/data_analysis_1/stock_data/Stocks/aapl.us.txt' # Path to the data being used (Apple stock for now)
    # To change the stock to predict, just change the name at the end example - /msft.us.txt
    SEQ_LENGTH = 30  # Use 30 days of history
    BATCH_SIZE = 32 # Number of training examples used in one iteration to update a model's internal weights
    EPOCHS = 100 # The model passes through ALL the data 100 times and learns from all the samples 
    LEARNING_RATE = 0.001 # Learning rate is a hyperparameter that governs how much a machine learning model adjusts its parameters at each step of its optimization algorithm.
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {DEVICE}")
    print(f"Loading data from {STOCK_FILE}...")

    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        STOCK_FILE,
        seq_length=SEQ_LENGTH,
        test_size=0.2 # 80/20 Train/Test Split (Can be changed)
    )

    print(f"Training samples: {len(X_train)}") # number of training samples we have
    print(f"Test samples: {len(X_test)}") # Number of test samples we have
    print(f"Feature dimensions: {X_train.shape}") 

    # Create datasets and dataloaders
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test) 

    # The function of a data loader is to basically take our data and chunk it down into smaller sizes (BATCH_SIZE 32) to be fed into the model one at a time 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # The shuffle parameter randomly shuffles the data every epoch (epochs = 100 here)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize our Neural Netowrk model
    input_size = X_train.shape[1] * X_train.shape[2]  # seq_length * num_features
    model = StockPredictor(input_size=input_size, hidden_sizes=[128, 64, 32], dropout=0.2)

    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

    # Train our model
    print("\nTraining model...")
    train_losses, val_losses = train_model(
        model,
        train_loader,
        test_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE
    )

    # Evaluate our model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, test_loader, scaler, device=DEVICE)

    print("\n=== Test Set Results ===")
    print(f"RMSE: ${metrics['rmse']:.2f}") # On average, our prediction is about $X away from the real closing price.
    print(f"MAE: ${metrics['mae']:.2f}") # This is similar to MAE, but it penalizes big mistakes more.
    print(f"MAPE: {metrics['mape']:.2f}%") # Our predictions are wrong by about X% of the stock price, on average.

    # Save our model and its weights 
    torch.save(model.state_dict(), 'stock_predictor.pth')
    print("\nModel saved to 'stock_predictor.pth'")


if __name__ == '__main__':
    main()
