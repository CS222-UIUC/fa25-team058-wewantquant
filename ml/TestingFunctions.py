from keras.models import load_model
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

import joblib

def directionalAccuracy(modelName, X_test, y_test):
    print(f"Loading model : {modelName}...")

    # We need the exact scaler for the model
    model = load_model("./models/" + modelName + ".keras")
    scaler = joblib.load("./models/" + modelName + "_scaler.pkl")   

    num_features = X_test.shape[2]

    print("Running model predictions...")
    predictions = model.predict(X_test, verbose=0)

    # Inverse scale predictions and true values
    # The scaler expects <num_feature> values, so we must add blank columns, then grab the column we care about.
    predictions_rescaled = scaler.inverse_transform(
        np.concatenate((np.zeros((predictions.shape[0], num_features - 1)), predictions), axis=1))[:, -1]

    y_test_rescaled = scaler.inverse_transform(
        np.concatenate((np.zeros((y_test.shape[0], num_features - 1)), y_test.reshape(-1, 1)),axis=1,))[:, -1]

    # Compute direction (1 = up, -1 = down, 0 = flat)
    y_true_dir = np.sign(np.diff(y_test_rescaled, prepend=y_test_rescaled[0]))
    y_pred_dir = np.sign(np.diff(predictions_rescaled, prepend=predictions_rescaled[0]))

    directional_acc = accuracy_score(y_true_dir, y_pred_dir)
    print(f"Directional Accuracy: {directional_acc:.4f}")
    return directional_acc