import tensorflow as tf
import keras
from keras import layers

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Embedding

def getData(csv):
    df = pd.read_csv(csv)

    df_clean = df.dropna()
    features = ['open', 'high', 'low', 'close', 'volume']
    data = df_clean[features].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length, 3] - data[i+seq_length-1, 3]) # changed
        return np.array(X), np.array(y)

    seq_length = 60
    X, y = create_sequences(scaled_data, seq_length)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return data, scaler, X_train, X_test, y_train, y_test

def train():
    csvFile = "./data/all_stocks_5yr.csv"
    data, scaler, X_train, X_test, y_train, y_test = getData(csvFile)

    model = keras.models.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    layers.Dropout(0.2),
    layers.LSTM(50, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(25),
    layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    history = model.fit(X_train, y_train, 
                        epochs=1, 
                        batch_size=32, 
                        validation_split=0.1)
    
    model.save("./models/kerasRNN.keras")

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(
        np.concatenate((np.zeros((predictions.shape[0], data.shape[1]-1)), predictions), axis=1)
    )[:, -1]
    y_test_rescaled = scaler.inverse_transform(
        np.concatenate((np.zeros((y_test.shape[0], data.shape[1]-1)), y_test.reshape(-1, 1)), axis=1)
    )[:, -1]

    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_test_rescaled, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions))
    r2 = r2_score(y_test_rescaled, predictions)

    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation R²: {r2:.4f}")

if __name__ == "main":
    train()
