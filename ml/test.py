import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# ========== CONFIG ==========
MODEL_PATH = "./models/kerasRNN.keras"   # path to your saved model
DATA_PATH = "./data/all_stocks_5yr.csv"     # path to your data file
SEQ_LENGTH = 60                             # same as during training
TEST_SPLIT = 0.8                            # same split as before
# ============================

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 3])  # column 3 = 'close'
    return np.array(X), np.array(y)

def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()

    features = ['open', 'high', 'low', 'close', 'volume']
    data = df[features].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = create_sequences(scaled_data, SEQ_LENGTH)

    split = int(TEST_SPLIT * len(X))
    X_test, y_test = X[split:], y[split:]

    # Save scalers for inverse transform
    return X_test, y_test, scaler, data

def evaluate_directional_accuracy():
    print(f"Loading model from {MODEL_PATH} ...")
    model = load_model(MODEL_PATH)

    X_test, y_test, scaler, data = load_and_prepare_data()

    print("Running model predictions...")
    predictions = model.predict(X_test, verbose=0)

    # Inverse scale predictions and true values
    predictions_rescaled = scaler.inverse_transform(
        np.concatenate((np.zeros((predictions.shape[0], data.shape[1]-1)), predictions), axis=1)
    )[:, -1]

    y_test_rescaled = scaler.inverse_transform(
        np.concatenate((np.zeros((y_test.shape[0], data.shape[1]-1)), y_test.reshape(-1, 1)), axis=1)
    )[:, -1]

    # Compute direction (1 = up, -1 = down, 0 = flat)
    y_true_dir = np.sign(np.diff(y_test_rescaled, prepend=y_test_rescaled[0]))
    y_pred_dir = np.sign(np.diff(predictions_rescaled, prepend=predictions_rescaled[0]))

    directional_acc = accuracy_score(y_true_dir, y_pred_dir)
    print(f"\n📈 Directional Accuracy: {directional_acc:.4f}")

if __name__ == "__main__":
    evaluate_directional_accuracy()
