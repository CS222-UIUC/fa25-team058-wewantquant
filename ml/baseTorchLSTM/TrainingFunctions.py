import tensorflow as tf
import torch

import keras
from keras import layers

import numpy as np
import pandas as pd

import joblib
import os

import DataFunctions
import TestingFunctions

def getGPUstatus():

    print("TensorFlow version:", tf.__version__)
    print("GPUs Available:", tf.config.list_physical_devices("GPU"))

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

def trainTorchLSTM(df, name, pred_col, epochs, seq_length = 60, train_split = 0.8):

    # assume df has correct columns, and is cleaned. 

    scaled_data, scaler = DataFunctions.scaleData(df)
    X, y = DataFunctions.breakDataIntoSequences(scaled_data, seq_length, pred_col)
    X_train, X_test, y_train, y_test = DataFunctions.generateSplit(X, y, train_split)

    #data, scaler, X_train, X_test, y_train, y_test = getData(csvFile)

    model = keras.models.Sequential(
        [
            layers.LSTM(
                50,
                return_sequences=True,
                input_shape=(X_train.shape[1], X_train.shape[2]),
            ),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1)

    os.makedirs("./models", exist_ok=True)
    model.save("./models/" + name + ".keras")
    joblib.dump(scaler, "./models/" + name + "_scaler.pkl")

    directional_acc = TestingFunctions.directionalAccuracy(name, X_test, y_test)

    return history
